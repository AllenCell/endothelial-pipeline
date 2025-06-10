from copy import deepcopy

import torch
from cyto_dl.models.im2im.diffusion_autoencoder import DiffusionAutoEncoder
from matplotlib import pyplot as plt
from monai.utils import convert_to_tensor
from torchmetrics import MeanMetric


class DiffAEFinetune(DiffusionAutoEncoder):
    def __init__(
        self,
        paired_condition_key: str,
        use_separate_encoders: bool = False,
        infer_with_fixed: bool = True,
        **base_kwargs,
    ):
        """
        Class for finetuning a DiffAE model using paired data (e.g. fixed vs. live). A checkpoint should be provided when using this class, as it will initialize the semantic encoder and autoencoder from the checkpoint.

        Parameters
        ----------
        paired_condition_key: str
            The key in the batch that contains the images paired to the reference (i.e. `condition_key`) images
        use_separate_encoders : bool
            If True, use a separate encoder for the fixed semantic encoder. This encoder will be initialized using the weights of the semantic encoder. If False, the semantic encoder and diffusion UNet will both be trained to minimize the diffusion and matching losses.
        infer_with_fixed : bool
            If True, the inference will be done using the fixed semantic encoder. If False, the inference will be done using the semantic encoder.
        base_kwargs : dict
            Additional keyword arguments to pass to the parent class.
        """
        _DEFAULT_METRICS = {
            "train/loss": MeanMetric(),
            "train/loss/diffusion": MeanMetric(),
            "train/loss/mse": MeanMetric(),
            "val/loss": MeanMetric(),
            "val/loss/diffusion": MeanMetric(),
            "val/loss/mse": MeanMetric(),
            "test/loss": MeanMetric(),
        }
        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        if use_separate_encoders:
            self.fixed_semantic_encoder = deepcopy(self.semantic_encoder)
            for p in self.semantic_encoder.parameters():
                p.requires_grad = False
            for p in self.autoencoder.parameters():
                p.requires_grad = False
        else:
            self.fixed_semantic_encoder = self.semantic_encoder

    def configure_optimizers(self):
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

    def matching_forward(self, x, y, batch_idx, stage=None):
        """
        Extract features from appropriate encoders for matching and plot correlation between them.
        """
        x_feats = self.semantic_encoder(x)
        y_feats = self.fixed_semantic_encoder(y)
        if (
            (self.trainer.current_epoch + 1) % self.hparams.save_images_every_n_epochs
        ) == batch_idx == 0 and stage == "val":
            n_latents = x_feats.shape[1]
            fig, ax = plt.subplots(n_latents, 1, figsize=(10, 10))
            for i in range(n_latents):
                r = torch.corrcoef(torch.stack([x_feats[:, i], y_feats[:, i]]))[0][1]
                ax[i].scatter(
                    x_feats[:, i].detach().cpu().numpy(),
                    y_feats[:, i].detach().cpu().numpy(),
                    s=1,
                )
                ax[i].set_title(f"r = {r:.2f}")
                # plot y=x
                min_, max_ = x_feats[:, i].min().item(), x_feats[:, i].max().item()
                ax[i].plot([min_, max_], [min_, max_], color="red", linestyle="--")
            fig.savefig(
                f"{self.hparams.save_dir}/{self.current_epoch}_correlation_{stage}.png"
            )
            plt.close(fig)
        return x_feats, y_feats

    def model_step(self, stage, batch, batch_idx):
        batch = convert_to_tensor(batch)
        loss = {"diffusion": 0}
        if not self.hparams.use_separate_encoders:
            loss, _, _ = super().model_step(stage, batch, batch_idx)
            # regular DiffAE only has diffusion loss
            loss["diffusion"] = loss["loss"]

        x_feats, y_feats = self.matching_forward(
            batch[self.hparams.condition_key],
            batch[self.hparams.paired_condition_key],
            stage=stage,
        )

        # compute matching loss
        loss["mse"] = torch.nn.functional.mse_loss(x_feats, y_feats)
        loss["loss"] = loss["diffusion"] + loss["mse"]

        return loss, y_feats, None

    def encode_image(self, x):
        """
        Encode an image using the appropriate encoder.
        If `infer_with_fixed` is True, use the fixed semantic encoder; otherwise, use the semantic encoder.
        """
        encoder = (
            self.fixed_semantic_encoder
            if self.hparams.infer_with_fixed
            else self.semantic_encoder
        )
        with torch.no_grad():
            if self.spatial_inferer is not None:
                z, loc = self.spatial_inferer(x, encoder)
            else:
                z = encoder(x)
                loc = {}
        return z, loc
