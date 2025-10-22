import numpy as np
import torch
import tqdm
from bioio.writers import OmeTiffWriter
from cyto_dl.models.im2im.diffusion_autoencoder import DiffusionAutoEncoder as _BaseDiffAE
from cyto_dl.models.im2im.utils import detach
from monai.utils import convert_to_tensor


class DiffusionAutoEncoder(_BaseDiffAE):
    """
    This class is subclassed from `cyto_dl.models.im2im.diffusion_autoencoder.DiffusionAutoEncoder`.

    The added functionality allows for deterministic inference by providing control
    over the conditioning image, diffusion image, and noise generation, ensuring
    reproducible outputs for analysis and visualization.
    """

    def __init__(
        self,
        *,
        noise_cons=False,
        autoencoder,
        image_shape,
        condition_key,
        noise_scheduler,
        diffusion_inferer,
        spatial_inferer=None,
        loss=None,
        semantic_encoder=None,
        diffusion_key=None,
        n_inference_steps=50,
        save_dir="./",
        save_images_every_n_epochs=1,
        n_noise_samples=1,
        train_encoder=True,
        gamma=-1.0,
        fixed_sample_seed: int | None = 42,
        **base_kwargs,
    ):
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

    def _get_samplewise_noise(self, ref_tensor, global_sample_idx):
        shape, device, dtype = ref_tensor.shape, ref_tensor.device, ref_tensor.dtype
        seed = self._get_seed_for_sample(global_sample_idx, extra=999)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return torch.randn(shape, device=device, dtype=dtype, generator=gen)

    def _get_fixed_samples(self, batch):
        """Get fixed samples for consistent visualization across epochs"""
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

    def save_example(self, stage, cond_img, diff_img):
        """Save the sequence of denoising steps."""
        # Use fixed samples if available
        if self.fixed_samples is not None:
            cond_img = self.fixed_samples["cond"]
            diff_img = self.fixed_samples["diff"]

        with torch.no_grad():
            cond = self.semantic_encoder(cond_img).unsqueeze(2)

        # Use fixed seed for consistent noise generation across epochs
        if self.fixed_sample_seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.fixed_sample_seed)
            # Fix: Use torch.randn with generator, then move to correct device if needed
            noise = torch.randn(
                diff_img.shape, generator=generator, device=self.device, dtype=diff_img.dtype
            )
        else:
            noise = torch.randn_like(diff_img, device=self.device)

        sample = self._generate_image(noise, cond)

        for img, name in zip([cond_img, diff_img, sample], ["cond", "diff", "recon"]):
            OmeTiffWriter.save(
                uri=f"{self.hparams.save_dir}/{self.trainer.current_epoch}_{stage}_{name}.tiff",
                data=detach(img).astype(float),
            )

    def model_step(self, stage, batch, batch_idx):
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

    def generate_from_latent_and_noised_image(
        self,
        conditioning_vector: torch.Tensor,
        noised_image: torch.Tensor,
        batch_size: int = 3,
    ) -> np.ndarray:
        """
        Generate image by denoising a noised image conditioned on a latent vector.

        Allows for batch processing to handle a series of conditioning vectors.

        **Input tensor shapes**:

        The input conditioning vector array should have shape (B, D), where
        B is the number of conditioning vectors and D is the dimensionality
        of the latent space. This allows for generating multiple images corresponding
        to B different conditioning vectors.

        The noised image tensor should have shape (C, H, W), where C is the number of channels,
        H is the height, and W is the width of the image to be denoised. Note that this shape
        should be the same as `model.image_shape`.

        Parameters
        ----------
        conditioning_vector
            A tensor of shape (B, D) representing the denoising conditioning vector(s).
        noised_image
            A tensor of shape (C, H, W) representing the noised image to use for image generation.
        batch_size
            The batch size for processing.
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be at least 1")
        batch_indices = [
            (i, i + batch_size) for i in range(0, conditioning_vector.shape[0], batch_size)
        ]
        with torch.no_grad():
            noise_stacked = torch.stack([noised_image] * conditioning_vector.shape[0])
            reconstructed_image = torch.cat(
                [
                    self._generate_image(
                        noise_stacked[start:stop], conditioning_vector[start:stop].unsqueeze(2)
                    ).squeeze(1)
                    for start, stop in tqdm.tqdm(batch_indices, desc="Generating batch")
                ],
                0,
            )
        reconstructed_image_numpy = detach(reconstructed_image).astype(float)
        return reconstructed_image_numpy
