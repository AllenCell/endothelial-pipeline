from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from cyto_dl.models.im2im.diffusion_autoencoder import DiffusionAutoEncoder
from matplotlib import pyplot as plt
from monai.utils import convert_to_tensor
from torchmetrics import MeanMetric


class DropoutWrapper(nn.Module):
    """Wrapper module that applies dropout to encoder outputs.

    Args:
        encoder: The encoder module to wrap
        dropout_rate: Dropout probability (default: 0.3)
    """

    def __init__(self, encoder, dropout_rate=0.3):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        features = self.encoder(x)
        if self.training:
            features = self.dropout(features)
        return features


class DiffAEFinetune(DiffusionAutoEncoder):
    """Class for finetuning a DiffAE model using paired data."""

    def __init__(
        self,
        paired_condition_key: str,
        use_separate_encoders: bool = True,
        infer_with_fixed: bool = True,
        **base_kwargs: dict,
    ) -> None:
        """
        Finetune a DiffAE model using paired data (e.g. fixed vs. live).
        A checkpoint should be provided when using this class, as it will
        initialize the semantic encoder and autoencoder from the checkpoint.

        Parameters
        ----------
        paired_condition_key: str
            The key in the batch that contains the images paired
            to the reference (i.e. `condition_key`) images
        use_separate_encoders : bool
            If True, use a separate encoder for the fixed semantic encoder.
            This encoder will be initialized using the weights of the semantic
            encoder. If False, the semantic encoder and diffusion UNet will
            both be trained to minimize the diffusion and matching losses.
        infer_with_fixed : bool
            If True, the inference will be done using the fixed semantic
            encoder. If False, the inference will be done using the semantic encoder.
        base_kwargs : dict
            Additional keyword arguments to pass to the parent class.
        """
        _default_metrics = {
            "train/loss": MeanMetric(),
            "train/loss/diffusion": MeanMetric(),
            "train/loss/mse": MeanMetric(),
            "val/loss": MeanMetric(),
            "val/loss/diffusion": MeanMetric(),
            "val/loss/mse": MeanMetric(),
            "test/loss": MeanMetric(),
        }
        metrics = base_kwargs.pop("metrics", _default_metrics)
        super().__init__(metrics=metrics, **base_kwargs)

        if use_separate_encoders:
            # First create the encoder copy
            encoder_copy = deepcopy(self.semantic_encoder)

            # Freeze everything in the encoder
            for param in encoder_copy.parameters():
                param.requires_grad = False

            # Selectively unfreeze specific layers
            for name, param in encoder_copy.named_parameters():
                # Only train the last 2 sub-blocks of features.7 + classifier
                if any(
                    x in name for x in ["features.7.2", "classifier"]
                ):  # Remove features.7.1 if still too many params
                    param.requires_grad = True
                    print(f"✓ Training: {name} - {param.numel():,} params")

            # NOW wrap with dropout
            self.fixed_semantic_encoder = DropoutWrapper(encoder_copy, dropout_rate=0.3)

            # Original freezing
            for p in self.semantic_encoder.parameters():
                p.requires_grad = False
            for p in self.autoencoder.parameters():
                p.requires_grad = False

            # Print trainable parameter count
            trainable = sum(
                p.numel() for p in self.fixed_semantic_encoder.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self.fixed_semantic_encoder.parameters())
            print(f"\n{'='*60}")
            print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
            print(f"{'='*60}\n")
        else:
            self.fixed_semantic_encoder = self.semantic_encoder

    def configure_optimizers(self) -> tuple[list, list]:
        """Configure optimizers for the DiffAE finetune model."""
        if self.hparams.use_separate_encoders:
            # only optimizing the paired encoder to match the semantic encoder
            params = list(self.fixed_semantic_encoder.parameters())
        else:
            params = list(self.autoencoder.parameters())
            if self.hparams.train_encoder:
                params += list(self.semantic_encoder.parameters())

        opt = self.optimizer(params)
        sched = self.lr_scheduler(optimizer=opt)
        return [opt], [sched]

    def matching_forward(
        self, x: torch.Tensor, y: torch.Tensor, batch_idx: int, stage: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from appropriate encoders
        for matching and plot correlation between them.
        """
        x_feats = self.semantic_encoder(x)
        # Add noise to target features during training (regularization)
        if self.training and stage == "train":
            noise = torch.randn_like(x_feats) * 0.05  # 5% noise
            x_feats = x_feats + noise
        # else:
        #     x_feats = x_feats
        y_feats = self.fixed_semantic_encoder(y)

        # Log distributional metrics every epoch for train and val
        if stage in ["val", "train"]:
            # Mean and std per batch
            # self.log(f"{stage}/x_mean", x_feats.mean())
            # self.log(f"{stage}/y_mean", y_feats.mean())
            # self.log(f"{stage}/x_std", x_feats.std())
            # self.log(f"{stage}/y_std", y_feats.std())

            # Cosine similarity (direction alignment)
            cos_sim = torch.nn.functional.cosine_similarity(x_feats, y_feats, dim=1).mean()
            self.log(f"{stage}/cosine_similarity", cos_sim)

            # Correlation coefficient every epoch
            x_flat = x_feats.reshape(-1).detach().cpu().float().numpy()
            y_flat = y_feats.reshape(-1).detach().cpu().float().numpy()
            r = np.corrcoef(x_flat, y_flat)[0, 1]
            self.log(f"{stage}/coef_corr", r)

        # Only plot correlation scatter every N epochs
        if (
            ((self.trainer.current_epoch + 1) % self.hparams.save_images_every_n_epochs) == 0
            and batch_idx == 0
            and stage == "val"
        ):
            # Flatten to get all crop-feature pairs
            x_flat = x_feats.reshape(-1).detach().cpu().float().numpy()  # [4096]
            y_flat = y_feats.reshape(-1).detach().cpu().float().numpy()  # [4096]

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.scatter(x_flat, y_flat, s=1, alpha=0.3)

            # Correlation across ALL values
            r = np.corrcoef(x_flat, y_flat)[0, 1]
            ax.set_title(f"Overall Feature Alignment (r = {r:.4f})")
            ax.set_xlabel("Fixed Encoder Features")
            ax.set_ylabel("Live Encoder Features")

            # y=x line
            min_val, max_val = min(x_flat.min(), y_flat.min()), max(x_flat.max(), y_flat.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)
            ax.set_aspect("equal")

            plt.tight_layout()
            fig.savefig(f"{self.hparams.save_dir}/epoch_{self.current_epoch}_alignment_{stage}.png")
            plt.close(fig)

        return x_feats, y_feats

    def model_step(
        self, stage: str, batch: torch.Tensor, batch_idx: int
    ) -> tuple[dict, torch.Tensor, None]:
        """Run a model step for the DiffAE finetune model."""
        batch = convert_to_tensor(batch)
        # initialize loss dictionary
        loss = {"diffusion": 0}
        if not self.hparams.use_separate_encoders:
            loss, _, _ = super().model_step(stage, batch, batch_idx)
            # regular DiffAE only has diffusion loss
            loss["diffusion"] = loss["loss"]

        x_feats, y_feats = self.matching_forward(
            batch[self.hparams.condition_key],
            batch[self.hparams.paired_condition_key],
            batch_idx=batch_idx,
            stage=stage,
        )

        # compute matching loss
        loss["mse"] = torch.nn.functional.mse_loss(x_feats, y_feats)  # type: ignore[assignment]
        loss["loss"] = loss["diffusion"] + loss["mse"]

        return loss, y_feats, None

    def encode_image(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Encode an image using the appropriate encoder.
        If `infer_with_fixed` is True, use the fixed semantic encoder;
        otherwise, use the semantic encoder.
        """
        encoder = (
            self.fixed_semantic_encoder if self.hparams.infer_with_fixed else self.semantic_encoder
        )
        with torch.no_grad():
            if self.spatial_inferer is not None:
                z, loc = self.spatial_inferer(x, encoder)
            else:
                z = encoder(x)
                loc = {}
        return z, loc
