from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from monai.transforms import CenterSpatialCropd, Rotated, Transform
from omegaconf import ListConfig


# making a comment that this class is currently unused
class MinStdCropd(Transform):
    """
    Select a slice of an image based on the
    minimum standard deviation across a specified axis.
    """

    def __init__(
        self,
        keys: list | str,
        offset: int = 5,
        axes: tuple[int, int] = (-1, -2),
        channel: int = 0,
    ):
        super().__init__()
        self.keys = keys if isinstance(keys, list | ListConfig) else [keys]
        self.offset = offset
        self.axes = axes
        self.channel = channel

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Call the transform on the input data."""
        for key in self.keys:
            z_profile = data[key].std(axis=self.axes)[self.channel]
            min_std = z_profile.argmin().item()
            start, stop = min_std - self.offset, min_std + self.offset
            start = max(start, 0)
            stop = min(stop, len(z_profile))

            projection = data[key][:, start:stop]
            projection.meta.update({"min_std_slice": min_std})
            data[key] = projection
        return data


class RotateRanged(Transform):
    """
    Apply a range of rotations to a set of images.
    Useful for e.g. testing rotation dependence of a principal component
    or latent dimension. In use, this transform should be applied **after**
    a cropping transform (like the RandSpatialCropSamplesd used for training)
    where the roi size is at least sqrt(roi_size[0]**2 + roi_size[y]**2)).

    The spatial inferer in the model config should also be set to none to
    avoid overwriting the `start_y` and `start_x` metadata.
    """

    def __init__(
        self,
        keys: list | str,
        roi_size: Sequence[int],
        rotation_range: list[float] | None = None,
        n_steps: int = 10,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        keys: Union[list, str]
            keys to apply range of rotations to
        roi_size: Sequence[int]
            size of the region of interest to crop after rotation
        rotation_range: Sequence[float]
            range of angles to rotate the images, in radians
        n_steps: int
            number of steps to take in the rotation range
        allow_missing_keys: bool
            if True, will not raise an error if keys are missing from the input dictionary
        """
        if rotation_range is None:
            # default to full rotation range
            rotation_range = [0.0, 2 * np.pi]
        super().__init__()
        self.keys = keys if isinstance(keys, list | ListConfig) else [keys]
        self.allow_missing_keys = allow_missing_keys
        self.rotation_range = rotation_range
        self.n_steps = n_steps
        self.cropper = CenterSpatialCropd(keys=self.keys, roi_size=roi_size)

    def _split_dict(self, dict: dict) -> list[dict]:
        """Split channels of keys into a list of dictionaries."""
        meta_keys = {k: v for k, v in dict.items() if k not in self.keys}
        n_channels = dict[self.keys[0]].shape[0]
        new_data = [deepcopy(meta_keys) for _ in range(n_channels)]
        for i in range(n_channels):
            for k in self.keys:
                loc = dict[k].meta["location"]
                start_y, start_x = loc[0][i], loc[1][i]
                elem = dict[k][i].unsqueeze(0)
                elem.meta.update({"start_x": start_x, "start_y": start_y})
                elem.meta.pop("location")
                elem.meta.pop("affine")
                elem.meta.pop("offset")
                elem.meta.pop("count")
                new_data[i][k] = deepcopy(elem)
        return new_data

    def __call__(self, input_dict: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        input_dict: Dict[str, torch.Tensor]
            dict of CZYX tensors/metadata
        """
        new_data = []
        for theta in np.linspace(*self.rotation_range, self.n_steps):
            rotations = Rotated(keys=self.keys, angle=theta, keep_size=False, padding_mode="zeros")(
                input_dict
            )
            rotations = self.cropper(rotations)
            for key in self.keys:
                rotations[key].meta.update({"theta": theta})  # type: ignore
            new_data += self._split_dict(rotations)
        return new_data
