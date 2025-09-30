from cyto_dl.callbacks.latent_walk_diffae import DiffAELatentWalk


class DiffAELatentWalkRank0(DiffAELatentWalk):
    """
    Subclass of DiffAELatentWalk from Cyto-DL that only performs latent walk computation on rank 0
    during distributed training to avoid redundant computation and file conflicts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        # Only perform latent walk on rank 0
        if self._is_rank_zero(trainer):
            super().on_validation_epoch_end(trainer, pl_module)

    def on_predict_epoch_end(self, trainer, pl_module):
        # Only perform latent walk on rank 0
        if self._is_rank_zero(trainer):
            super().on_predict_epoch_end(trainer, pl_module)
