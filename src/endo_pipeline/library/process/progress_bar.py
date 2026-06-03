"""Wrapper class for progress bars."""

from termcolor import colored
from tqdm import tqdm


class ProgressBar(tqdm):
    """Wrapper for simple progress bars with colored iterations and step notes."""

    def __init__(self, iterable: list[str], description: str, group: str | None = None):
        self.color = "cyan"
        self.description = description

        if group is not None:
            colored_group = colored(group, color=self.color, attrs=["bold"])
            self.description = f"{self.description} {colored_group}"

        bar_format = "{desc}{postfix}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        super().__init__(iterable=iterable, desc=self.description, bar_format=bar_format)

    def set_iteration_name(self, iteration: str):
        """Set progress bar iteration name."""
        colored_iteration = colored(iteration, color=self.color, attrs=["bold", "underline"])
        self.set_description_str(f"{self.description} {colored_iteration}")

    def clear_iteration_name(self):
        """Clear the progress bar iteration name."""
        self.set_description_str(self.description)

    def set_step_description(self, step: str):
        """Set progress bar step description."""
        colored_step = colored(step, color=self.color)
        self.set_postfix_str(colored_step)
