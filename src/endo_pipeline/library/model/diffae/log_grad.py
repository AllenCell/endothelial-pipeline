"""Callback for logging gradient norms of the semantic encoder during training."""

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


class GradientLoggingCallback(Callback):
    """Callback for logging the gradient norms of the semantic encoder.

    This callback computes and logs the L2 norms of the gradients for parameters
    in the "semantic_encoder" module after every backward pass. Parameters are
    optionally grouped by dot-separated name levels, and logging occurs every
    `log_every_n_steps` steps on the rank zero process only.

    Parameters
    ----------
    grouping_level
        Number of dot-separated levels to use for grouping parameter names.
    log_every_n_steps
        Number of steps between each logging event.

    """

    def __init__(self, grouping_level: int = 2, log_every_n_steps: int = 50):
        """Initialize the callback with grouping and logging frequency parameters."""
        super().__init__()
        self.grouping_level = grouping_level
        self.log_every_n_steps = log_every_n_steps

    def _get_group(self, name):
        if self.grouping_level == -1:
            return name
        return ".".join(name.split(".")[: self.grouping_level])

    def on_after_backward(self, trainer, pl_module):
        """Compute and log the average L2 norm of gradients for semantic encoder parameters."""
        if not trainer.is_global_zero:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        group_norms: dict[str, list[float]] = {}

        for name, parameter in pl_module.named_parameters():
            if name.startswith("semantic_encoder") and parameter.grad is not None:
                group_name = self._get_group(name)
                group_norms.setdefault(group_name, []).append(
                    torch.norm(parameter.grad, p=2).item()
                )

        # Compute averaged norm per group
        avg_group_norms = {
            f"grad_norm/{name}": np.mean(group_norms[name])
            for name in group_norms
            if group_norms[name]
        }

        if avg_group_norms:
            trainer.logger.log_metrics(avg_group_norms, step=trainer.global_step)
