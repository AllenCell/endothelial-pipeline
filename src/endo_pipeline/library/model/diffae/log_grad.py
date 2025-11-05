import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

class GradientLoggingCallback(Callback):
    def __init__(self, grouping_level: int = 2, log_every_n_steps: int = 50):
        super().__init__()
        self.grouping_level = grouping_level
        self.log_every_n_steps = log_every_n_steps

    def _get_group(self, name):
        if self.grouping_level == -1:
            return name
        return ".".join(name.split(".")[: self.grouping_level])

    def on_after_backward(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        norms = dict()
        group_norms = dict()

        for name, parameter in pl_module.named_parameters():
            if name.startswith("semantic_encoder") and parameter.grad is not None:
                group_name = self._get_group(name)
                group_norms.setdefault(group_name, []).append(torch.norm(parameter.grad, p=2).item())

        # Compute averaged norm per group
        avg_group_norms = {f"grad_norm/{name}": np.mean(group_norms[name]) for name in group_norms if group_norms[name]}

        if avg_group_norms:
            trainer.logger.log_metrics(avg_group_norms, step=trainer.global_step)

