from collections.abc import Sequence
from warnings import warn

import cv2
import numpy as np
import torch
from bioio.writers import OmeTiffWriter
from cyto_dl.callbacks.latent_walk_diffae import DiffAELatentWalk

from .diffusion_autoencoder import detach


class DiffAELatentWalkRank0(DiffAELatentWalk):
    """
    Subclass of DiffAELatentWalk from Cyto-DL that only performs latent walk computation on rank 0
    during distributed training to avoid redundant computation and file conflicts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_feats: list[np.ndarray] = []

    @staticmethod
    def get_core_model(model):
        # Returns the "real" model regardless of DDP or not
        return getattr(model, "module", model)

    @staticmethod
    def _write_pc_vals(walk_img: np.ndarray, ranges: Sequence[np.ndarray]) -> np.ndarray:
        """Write PC index and value on image. Expects and returns NumPy."""
        idx = 0
        for i, range_ in enumerate(ranges):
            for val in range_:
                img = walk_img[idx]
                if torch.is_tensor(img):
                    img = img.detach().cpu().numpy()
                # Add text
                img = DiffAELatentWalkRank0._write_text(img, f"PC{i+1}:{val:.1f}")
                walk_img[idx] = img
                idx += 1
        return walk_img

    def _is_rank_zero(self, trainer):
        """Check if current process is rank 0"""
        return not hasattr(trainer, "local_rank") or trainer.local_rank == 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # Only collect features on rank 0
        if self._is_rank_zero(trainer):
            super().on_validation_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        This callback will be executed at the end of each validation epoch.
        Generated latent space walks along PC axes and saves them out as TIFF
        files for visualization.

        Only executed on rank 0 in distributed training setups to avoid
        redundant computations.

        Parameters
        ----------
        trainer
            The PyTorch Lightning Trainer instance managing the training loop.
        pl_module
            The PyTorch Lightning Module being trained, which should have access
            to the model and its hyperparameters.
        """
        # Skip execution on non-primary ranks in distributed training
        if not self._is_rank_zero(trainer):
            return

        # Only perform latent walk every N epochs to reduce computational overhead
        if (trainer.current_epoch + 1) % self.every_n_epoch != 0:
            return

        # Concatenate all validation features collected during this epoch
        feats = np.concatenate(self.val_feats)
        save_path = f"{pl_module.hparams.save_dir}/{trainer.current_epoch+1}_latent_walk.tiff"

        # Validate that sufficient data is available for PCA decomposition
        if len(feats.shape) == 1 or feats.shape[0] < self.num_pcs:
            warn(
                f"Insufficient data for latent walk with {self.num_pcs} PCs. Skipping...",
                stacklevel=2,
            )
            return

        pca_data = self.pca.fit_transform(feats)
        print(f"Explained variance ratio: {self.pca.named_steps['pca'].explained_variance_ratio_}")

        walk_list: list[np.ndarray] = []
        ranges = []

        # Latent Walks!
        for pc in range(self.num_pcs):
            std = pca_data[:, pc].std()
            if self.sigma_range is None:
                min_val = pca_data[:, pc].min() / std
                max_val = pca_data[:, pc].max() / std
                range_ = np.linspace(min_val, max_val, self.n_steps)
            else:
                range_ = np.linspace(-self.sigma_range, self.sigma_range, self.n_steps)
            print(f"PC{pc} range: {range_}")

            # Create latent values by varying a single PC while keeping others at zero
            for i in range_:
                array = np.zeros(self.num_pcs)
                array[pc] = i * std
                walk_list.append(array)
            ranges.append(range_)

        # Stack all latent codes and inverse transform from PCA space to feature space
        walk_array = np.stack(walk_list)
        walk_pca = self.pca.inverse_transform(walk_array)

        # Get the model and prepare the tensors
        model = DiffAELatentWalkRank0.get_core_model(trainer.model)
        walk = torch.from_numpy(walk_pca).float().to(model.device)

        # Generate images from latent values using the model
        walk_img = model.generate_from_latent(
            walk,
            n_noise_samples=self.n_noise_samples,
            average=self.average,
            save=False,
            batch_size=self.batch_size,
        )

        # Post-process generated images: detach from computation graph and reshape
        walk_img = detach(walk_img).astype(np.float32)
        walk_img = walk_img.reshape(walk_img.shape[0], -1, walk_img.shape[-1])

        # Annotate the generated images with the PC values
        walk_img = self._write_pc_vals(walk_img, ranges)

        # Save out these images
        OmeTiffWriter.save(uri=save_path, data=walk_img)

        self.val_feats = []

    def on_predict_epoch_end(self, trainer, pl_module):
        # Only perform latent walk on rank 0
        if self._is_rank_zero(trainer):
            super().on_predict_epoch_end(trainer, pl_module)

    @staticmethod
    def _write_text(img, text):
        """
        Renders text annotation onto an image at the upper right corner.

        Handles format conversions (torch tensors → numpy, channel dimensions, dtypes)
        and automatically selects appropriate text color based on image intensity.

        Parameters
        ----------
        img
            Input image in (h,w,c), (c,h,w), or (h,w) format.
        text
            Text string to render on the image.
        """

        # Convert torch.Tensor to numpy array
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        # Ensure shape is HWC for OpenCV
        if img.ndim == 3 and img.shape[0] < img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 2:  # grayscale
            img = img[..., None]
        # Convert float to uint8
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        # Remove channel dimension if 1-channel
        if img.shape[2] == 1:
            img = img.squeeze(2)

        # Color for putText: grayscale or RGB
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (int(img.max()),) * (3 if img.ndim == 3 and img.shape[2] == 3 else 1)
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = img.shape[1] - text_size[0] - 3
        text_y = text_size[1] + 3
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

        return img
