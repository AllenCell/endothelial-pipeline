"""Methods specific to the DiffusionAutoEncoder model, including deterministic inference and visualization."""

import logging
from copy import deepcopy

import numpy as np
import torch
import tqdm
from bioio.writers import OmeTiffWriter
from cyto_dl.models.im2im.diffusion_autoencoder import DiffusionAutoEncoder as _BaseDiffAE
from monai.utils import convert_to_tensor

logger = logging.getLogger(__name__)


def detach(img: torch.Tensor | np.ndarray) -> np.ndarray:
    """Detach tensor from graph and move to CPU, or return numpy array as is."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dtype == torch.bfloat16:
            img = img.half()
        return img.numpy()
    return img


class _AverageAccumulator:
    """Accumulator that computes the average of added samples."""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self._accumulator: torch.Tensor | None = None

    def add(self, sample: torch.Tensor) -> None:
        self._accumulator = sample if self._accumulator is None else (self._accumulator + sample)

    def result(self) -> torch.Tensor:
        if self._accumulator is None:
            raise ValueError("No samples were added to the accumulator.")
        return self._accumulator / self.n_samples


class _AppendAccumulator:
    """Accumulator that appends samples along a new dimension."""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self._accumulator: list[torch.Tensor] = []

    def add(self, sample: torch.Tensor) -> None:
        self._accumulator.append(sample)

    def result(self) -> torch.Tensor:
        return torch.cat(self._accumulator, dim=-1)


def _get_accumulator(average: bool, n_samples: int):
    """Return the appropriate accumulator based on whether averaging or appending is desired."""
    if average:
        return _AverageAccumulator(n_samples)
    else:
        return _AppendAccumulator(n_samples)


class DiffusionAutoEncoder(_BaseDiffAE):
    """Enhanced Diffusion AutoEncoder with deterministic inference and visualization capabilities.

    The added functionality in this class allows for deterministic inference by providing
    control over the conditioning image, diffusion image, and noise generation, ensuring
    reproducible outputs for analysis and visualization.
    """

    def __init__(
        self,
        *,
        noise_cons: bool = False,
        autoencoder,
        image_shape,
        condition_key,
        noise_scheduler,
        diffusion_inferer,
        spatial_inferer=None,
        loss=None,
        semantic_encoder=None,
        diffusion_key=None,
        n_inference_steps: int = 50,
        save_dir: str = "./",
        save_images_every_n_epochs: int = 1,
        n_noise_samples: int = 1,
        train_encoder: bool = True,
        gamma: float = -1.0,
        fixed_sample_seed: int | None = 42,
        **base_kwargs,
    ):
        """Initialize the DiffusionAutoEncoder with options for deterministic noise and fixed samples."""
        self.fixed_noise: torch.Tensor | None = None
        self.noise_cons = noise_cons
        # Store fixed samples for consistent visualization
        self.fixed_sample_seed = fixed_sample_seed
        self.fixed_samples = None
        super().__init__(
            autoencoder=autoencoder,
            image_shape=image_shape,
            condition_key=condition_key,
            noise_scheduler=noise_scheduler,
            diffusion_inferer=diffusion_inferer,
            spatial_inferer=spatial_inferer,
            loss=loss,
            semantic_encoder=semantic_encoder,
            diffusion_key=diffusion_key,
            n_inference_steps=n_inference_steps,
            save_dir=save_dir,
            save_images_every_n_epochs=save_images_every_n_epochs,
            n_noise_samples=n_noise_samples,
            train_encoder=train_encoder,
            gamma=gamma,
            **base_kwargs,
        )

    def _get_seed_for_sample(self, global_sample_idx, extra=0):
        return int(global_sample_idx) + int(extra)

    def _generate_image(self, noise, cond):
        if cond.ndim != 3 or cond.shape[1] != 1:
            raise ValueError(f"condition must be (B, 1, D), got {cond.shape}")
        self.scheduler.set_timesteps(num_inference_steps=self.hparams.n_inference_steps)
        with torch.no_grad():
            sample, intermediates = self.inferer.sample(
                input_noise=noise,
                diffusion_model=self.autoencoder,
                scheduler=self.scheduler,
                save_intermediates=True,
                conditioning=cond,
                verbose=False,
                intermediate_steps=1,
            )
        # during training, final image is all-nan, try to return last non-nan image
        if torch.any(torch.isnan(sample)):
            while torch.any(torch.isnan(sample)) and len(intermediates) > 0:
                sample = intermediates.pop(-1)
        return sample

    def _get_samplewise_noise(self, ref_tensor, global_sample_idx):
        shape, device, dtype = ref_tensor.shape, ref_tensor.device, ref_tensor.dtype
        seed = self._get_seed_for_sample(global_sample_idx, extra=999)
        # Draw on the CPU with a CPU generator for device-independent noise,
        # then move (and cast) to the reference tensor's device/dtype.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        return torch.randn(shape, generator=gen).to(device=device, dtype=dtype)

    def _get_fixed_samples(self, batch):
        """Get fixed samples for consistent visualization across epochs.

        This function does not generate the noise sample directly,
        but it does set the random seed for reproducibility if
        `fixed_sample_seed` is provided.
        """
        if self.fixed_samples is None and self.fixed_sample_seed is not None:
            # Set seed for reproducible sample selection
            generator = torch.Generator()
            generator.manual_seed(self.fixed_sample_seed)

            # Get first sample from batch
            cond_img = batch[self.hparams.condition_key][:1]
            diff_img = batch[self.diffusion_key][:1]

            # Store fixed samples
            self.fixed_samples = {
                "cond": cond_img.clone().detach(),
                "diff": diff_img.clone().detach(),
            }

        return self.fixed_samples

    def _make_generator(self, seed: int | None, device: str):
        """Create a torch generator object with an optional seed."""
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        return gen

    def _generate_noise(
        self,
        shape,
        dtype,
        device: str,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate noise from an optional seed.

        When a seed is provided, the noise is always drawn on the CPU with a CPU
        generator and then moved to ``device``. A CUDA generator and a CPU
        generator produce different sequences from the same seed, so drawing on
        the CPU is what keeps the sampled noise -- and therefore diffusion
        reconstructions -- identical across CPU and GPU.
        """
        if seed is not None:
            # Draw seeded noise on the CPU for device independence, then move
            # (and cast) to the target device.
            cpu_generator = self._make_generator(seed, "cpu")
            noise = torch.randn(shape, generator=cpu_generator).to(device)
            return noise.to(dtype) if dtype is not None else noise

        # Unseeded: fall back to the global RNG on the target device.
        return torch.randn(shape, device=device, dtype=dtype)

    def _get_fixed_noise(self, shape, dtype, device: str) -> torch.Tensor:
        """Generate or retrieve fixed noise for `noise_cons` mode.

        Only generates noise once and caches it if `noise_cons=True`.
        This ensures the exact same noise vector is used across all epochs.
        """
        if self.fixed_noise is None:
            # Fixed noise is always seeded if a fixed_sample_seed exists
            seed = self.fixed_sample_seed  # can be None → truly random but still cached
            self.fixed_noise = self._generate_noise(
                shape=shape, dtype=dtype, device=device, seed=seed
            )
        return self.fixed_noise

    def save_example(self, stage, cond_img, diff_img):
        """Save the sequence of denoising steps."""
        # Use fixed samples if available
        if self.fixed_samples is not None:
            cond_img = self.fixed_samples["cond"]
            diff_img = self.fixed_samples["diff"]

        with torch.no_grad():
            cond = self.semantic_encoder(cond_img).unsqueeze(1)

        # Generate noise
        device = self.device
        shape = diff_img.shape
        dtype = diff_img.dtype

        if self.noise_cons:
            # Same exact noise vector every epoch → isolates latent changes
            noise = self._get_fixed_noise(shape, dtype, device)
        else:
            # New noise every epoch, but reproducible if a base seed is given
            if self.fixed_sample_seed is not None:
                # Offset seed by epoch so we get different (but deterministic) noise each epoch
                seed = self.fixed_sample_seed + self.trainer.current_epoch
            else:
                seed = None  # truly random each call

            noise = self._generate_noise(shape, dtype, device, seed=seed)

        # ==================================================
        sample = self._generate_image(noise, cond)

        for img, name in [(cond_img, "cond"), (diff_img, "diff"), (sample, "recon")]:
            OmeTiffWriter.save(
                uri=f"{self.hparams.save_dir}/{self.trainer.current_epoch}_{stage}_{name}.tiff",
                data=detach(img).astype(float),
            )

    def model_step(self, stage, batch, batch_idx):
        """Perform a training/validation step and save example images at specified intervals."""
        batch = convert_to_tensor(batch)
        cond_img = batch[self.hparams.condition_key]
        diff_img = batch[self.diffusion_key]
        noise, noise_pred, latent, loss_weight = self.forward(cond_img, diff_img)

        # Only save on rank 0 and for the first batch of validation
        if (
            (self.trainer.current_epoch + 1) % self.hparams.save_images_every_n_epochs == 0
            and batch_idx == 0
            and stage == "val"
            and (not hasattr(self.trainer, "local_rank") or self.trainer.local_rank == 0)
        ):
            # Initialize fixed samples on first call
            if self.fixed_samples is None:
                self._get_fixed_samples(batch)
            self.save_example(stage, cond_img[:1], diff_img[:1])

        diffusion_loss = self.loss(noise, noise_pred)
        if loss_weight is not None:
            diffusion_loss = torch.mean(diffusion_loss * loss_weight)
        if diffusion_loss.numel() > 1:
            raise ValueError(
                f"Diffusion loss should be a scalar, got {diffusion_loss.shape}. Ensure `gamma` is provided if your loss has no reduction."
            )
        return {"loss": diffusion_loss}, latent, None

    def generate_from_latent(
        self,
        cond: torch.Tensor,
        save_name: str = "generated_image",
        n_noise_samples: int | None = None,
        average: bool = True,
        save: bool = True,
        batch_size: int = 3,
        random_seed: int | None = None,
    ) -> torch.Tensor:
        """Generate images from latent features.

        Parameters
        ----------
        cond
            A tensor of shape (N, D) or (N, 1, D) containing the conditioning vectors.
        save_name
            The base name for saving the generated image (without extension).
        n_noise_samples
            The number of noise samples to generate for each conditioning vector.
            If None, uses the default value from the model's hyperparameters.
        average
            If True, averages the generated samples for each conditioning vector.
            If False, appends the samples along a new dimension.
        save
            Whether to save the generated image as a TIFF file.
        batch_size
            The batch size to use when generating images.
            This controls how many conditioning vectors are processed at a time.
        random_seed
            An optional random seed for noise generation to ensure reproducibility.

        Returns
        -------
        torch.Tensor
            Detached tensor on CPU (shape: [N, C, H, W]).

        """
        if batch_size <= 0:
            raise ValueError("Batch size must be at least 1")

        if cond.ndim == 3 and cond.shape[1] == 1:
            pass
        elif cond.ndim == 2:
            cond = cond.unsqueeze(1)  # (N, D) → (N, 1, D)
        else:
            raise ValueError(f"cond must be (N, D) or (N, 1, D), got {cond.shape}")

        batch_indices = [
            (i, min(i + batch_size, cond.shape[0])) for i in range(0, cond.shape[0], batch_size)
        ]
        n_noise_samples = int(n_noise_samples or self.hparams.n_noise_samples)
        accumulator = _get_accumulator(average, n_noise_samples)

        with torch.no_grad():
            for i in tqdm.tqdm(range(n_noise_samples), desc="Sampling noise"):
                seed_i = random_seed + i if random_seed is not None else None
                noise = torch.stack(
                    [
                        self._generate_noise(
                            self.hparams.image_shape, None, device=self.device, seed=seed_i
                        )
                    ]
                    * cond.shape[0]
                )
                sample = torch.cat(
                    [
                        self._generate_image(noise[start:stop], cond[start:stop]).squeeze(1)
                        for start, stop in batch_indices
                    ],
                    dim=0,
                )
                accumulator.add(sample)

            recon_tensor = accumulator.result()

        recon_tensor = detach(recon_tensor)
        if isinstance(recon_tensor, np.ndarray):
            recon_tensor = torch.from_numpy(recon_tensor)
        recon_tensor = recon_tensor.cpu()
        if save:
            recon_np = recon_tensor.numpy().astype(float)
            OmeTiffWriter.save(
                uri=f"{self.hparams.save_dir}/{save_name}.tiff",
                data=recon_np,
            )
        return recon_tensor

    def generate_from_latent_and_noised_image(
        self,
        conditioning_vector: torch.Tensor,
        noised_image: torch.Tensor,
        batch_size: int = 3,
    ) -> np.ndarray:
        """Generate image by denoising a noised image conditioned on a latent vector.

        Allows for batch processing to handle a series of conditioning vectors.

        **Input tensor shapes**

        The input conditioning vector tensor should have shape ``(num_vecs, num_dims)``, where
        ``num_vecs`` is the number of conditioning vectors and ``num_dims`` is the dimensionality
        of the latent space. This allows for generating multiple images corresponding
        to ``num_vecs`` different conditioning vectors.

        The noised image tensor should have shape ``(num_channels, num_pixels_y, num_pixels_x)``,
        where ``num_channels`` is the number of channels, ``num_pixels_y`` is the height of the image
        (number of pixels in Y), and ``num_pixels_x`` is the width of the image (number of pixels in X).
        Note that this shape should be the same as ``model.image_shape`` in the model's configuration.

        Parameters
        ----------
        conditioning_vector
            A tensor holding the denoising conditioning vector(s).
        noised_image
            A tensor holding the noised image to use for image generation.
        batch_size
            The batch size for processing.

        Returns
        -------
        :
            A detached tensor on CPU containing the generated image(s) corresponding
            to the input conditioning vector(s).

        """
        if tuple(noised_image.shape) != tuple(self.hparams.image_shape):
            logger.error(
                "Noised image shape [ %s ] does not match model image shape [ %s ]",
                noised_image.shape,
                self.hparams.image_shape,
            )
            raise ValueError(
                f"Noised image shape [ {noised_image.shape} ] does not match model image shape [ {self.hparams.image_shape} ]"
            )

        if batch_size <= 0:
            logger.error("Batch size must be at least 1, got [ %d ]", batch_size)
            raise ValueError("Batch size must be at least 1")
        batch_indices = [
            (i, i + batch_size) for i in range(0, conditioning_vector.shape[0], batch_size)
        ]
        with torch.no_grad():
            noise_stacked = torch.stack([noised_image] * conditioning_vector.shape[0])
            reconstructed_image = torch.cat(
                [
                    self._generate_image(
                        noise_stacked[start:stop], conditioning_vector[start:stop].unsqueeze(1)
                    ).squeeze(1)
                    for start, stop in tqdm.tqdm(batch_indices, desc="Generating batch")
                ],
                0,
            )

        reconstructed_image = detach(reconstructed_image)
        if isinstance(reconstructed_image, np.ndarray):
            reconstructed_image = torch.from_numpy(reconstructed_image)
        reconstructed_image = reconstructed_image.cpu()

        return detach(reconstructed_image)

    # The forward method is modified to make this work with both cross-attention and AdaGN!

    def forward(self, x_cond: torch.Tensor, x_diff: torch.Tensor):
        """Perform a forward pass through the model, encoding the conditioning image and predicting noise."""
        B = x_diff.shape[0]
        device = x_diff.device

        noise = torch.randn_like(x_diff, device=device)
        timesteps = torch.randint(
            0,
            self.inferer.scheduler.num_train_timesteps,
            (B,),
            device=device,
            dtype=torch.long,
        )
        loss_weight = self._get_loss_weight(timesteps)

        # Encode condition: (B, lat_dim)
        latent = self.semantic_encoder(x_cond)

        # (B, 1, lat_dim) for cross-attention
        # AdaGN will internally .squeeze(1)
        condition = latent.unsqueeze(1)

        noise_pred = self.inferer(
            inputs=x_diff,
            diffusion_model=self.autoencoder,
            noise=noise,
            timesteps=timesteps,
            condition=condition,
        )

        return noise, noise_pred, latent, loss_weight

    def predict_step(self, batch, batch_idx):
        """Perform a single prediction step."""
        meta = batch[self.hparams.condition_key].meta
        dtype = torch.bfloat16 if self.trainer.precision == "bf16-mixed" else torch.float16
        self.to(dtype)
        batch = convert_to_tensor(batch, dtype=dtype)
        z, loc = self.encode_image(batch[self.hparams.condition_key])
        meta.update(loc)
        return detach(z), deepcopy(meta)
