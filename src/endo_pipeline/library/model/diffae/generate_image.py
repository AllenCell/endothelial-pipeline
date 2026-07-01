"""Methods for generating images from a DiffAE model (latent vectors to images)."""

import logging
import typing

import numpy as np
import pandas as pd
import torch

if typing.TYPE_CHECKING:
    import numpy as np
    from cyto_dl.models.im2im.diffusion_autoencoder import (
        DiffusionAutoEncoder as BaseDiffusionAutoEncoder,
    )

from endo_pipeline.library.analyze.pca import fit_pca
from endo_pipeline.library.model.diffae.diffusion_autoencoder import DiffusionAutoEncoder
from endo_pipeline.library.model.latent_walk_utils import (
    add_pc_coordinates_to_dataframe,
    get_num_pcs_from_column_names,
)
from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES

logger = logging.getLogger(__name__)


def add_noise_to_image(
    input_image: np.ndarray,
    noise_image: np.ndarray,
    noise_level: float,
) -> np.ndarray:
    """Add Gaussian noise to an input image at a specified noise level.

    **Noise level weighting**

    The output "noised" image is created using the formula:

    .. code-block:: python

        output_image = sqrt(1 - noise_level) * input_image + sqrt(noise_level) * noise_img

    Using this formula, `noise_level` represents the fraction of the corrupted
    image that is contributed by the noise image, with the remainder contributed
    by the original input image. An input `noise_level` of `0.0` results in no
    noise being added (the output image is identical to the input image), while a
    `noise_level` of `1.0` results in an image composed entirely of noise.

    Parameters
    ----------
    input_image
        The input image to which noise will be added.
    noise_image
        A standard Gaussian noise image of the same shape as the input image.
    noise_level
        The level of noise to add, between 0.0 (no noise) and 1.0 (all noise).

    Returns
    -------
    :
        The resulting noised image.

    """
    if not (0.0 <= noise_level <= 1.0):
        logger.error("Parameter `noise_level` must be between 0.0 and 1.0.")
        raise ValueError("Parameter `noise_level` must be between 0.0 and 1.0.")

    # Check edge cases for numerical efficiency
    if noise_level == 0.0:
        output_image = input_image.copy()
    elif noise_level == 1.0:
        output_image = noise_image.copy()
    else:  # general case
        output_image = np.sqrt(1 - noise_level) * input_image + np.sqrt(noise_level) * noise_image
    return output_image


def generate_from_coords_and_noised_image(
    model: "BaseDiffusionAutoEncoder | DiffusionAutoEncoder",
    coords: np.ndarray,
    noised_image: np.ndarray,
    num_gpus: int | None = None,
) -> np.ndarray:
    """Generate an image from an initial noisy image conditioned on a latent vector.

    **Input array shapes**

    The input conditioning vector array should have shape ``(num_vecs,
    num_dims)``, where ``num_vecs`` is the number of conditioning vectors and
    ``num_dims`` is the dimensionality of the latent space. This allows for
    generating multiple images corresponding to ``num_vecs`` different
    conditioning vectors.

    The noised image tensor should have shape ``(num_channels, num_pixels_y,
    num_pixels_x)``, where ``num_channels`` is the number of channels,
    ``num_pixels_y`` is the height of the image (number of pixels in Y), and
    ``num_pixels_x`` is the width of the image (number of pixels in X). Note
    that this shape should be the same as ``model.image_shape`` in the model's
    configuration.

    **Example usage**

    .. code-block:: python

        from endo_pipeline.io import load_model
        from endo_pipeline.manifests import load_model_manifest
        from endo_pipeline.library.model.diffae import (
            generate_from_coords_and_noised_image
        )

        model_manifest = load_model_manifest("my_model_manifest")
        model_location = model_manifest.locations["my_run_name"]
        model = load_model(model_location)

        gen_image = generate_from_coords_and_noised_image(
            model, coords=my_coords, # shape (num_vecs, num_dims)
            noised_image=my_noised_image, # shape (1, n_y, n_x)
        )

    Parameters
    ----------
    model
        The model to use for image generation (conditioned denoising).
    coords
        A coordinate in the latent space of the model; used to condition the
        denoising.
    noised_image
        An image used as the starting point for denoising by the model.
    num_gpus
        Optional, number of available GPUs.

    Returns
    -------
    :
        The generated image.

    """
    coords_torch = torch.from_numpy(coords).float()
    noised_image_torch = torch.from_numpy(noised_image).float()

    # move model and inputs to gpu if available, else
    # perform reconstruction on cpu
    device = "cpu" if num_gpus is None else "cuda"
    coords_ = coords_torch.to(device)
    noised_image_ = noised_image_torch.to(device)
    model_ = model.to(device)

    if not hasattr(model_, "generate_from_latent_and_noised_image"):
        logger.error(
            "Model class [ %s ] does not support generation from coordinates and noised image.",
            model_.__class__.__name__,
        )
        raise NotImplementedError(
            f"Model class [ {model_.__class__.__name__} ] does not support generation "
            "from coordinates and noised image."
        )

    gen_img = model_.generate_from_latent_and_noised_image(
        conditioning_vector=coords_,
        noised_image=noised_image_,
    )
    return gen_img


def generate_from_dataframe(
    dataframe: pd.DataFrame,
    column_names: list[str],
    model: "DiffusionAutoEncoder",
    num_gpus: int | None = None,
    random_seed: int | None = None,
    n_noise_samples: int = 1,
    average: bool = False,
) -> np.ndarray:
    """
    Reconstruct crops from feature coordinates stored in a given dataframe.

    Parameters
    ----------
    dataframe
        DataFrame containing the feature coordinates for image reconstruction.
    column_names
        List of column names corresponding to the feature coordinates in the
        dataframe.
    model
        DiffusionAutoEncoder model to use for image reconstruction.
    num_gpus
        Optional, number of available GPUs.
    random_seed
        Random seed for reproducibility of image generation. If None, does not
        set a random seed.
    n_noise_samples
        Number of noise samples to use for generating images. Each noise sample
        will result in a separate reconstructed image for each set of feature
        coordinates in the dataframe.
    average
        If True, average the generated images across noise samples for each set of
        feature coordinates, resulting in a single reconstructed image per set of
        feature coordinates. If False, return all generated images without averaging.

    Returns
    -------
    :
        Array of reconstructed images corresponding to the feature coordinates
        in the dataframe. The shape of the array will be (num_samples,
        img_width, img_height), where num_samples is equal to the number of rows
        in the dataframe multiplied by n_noise_samples.

    """

    # get minimum number of pcs needed for the fit pca object based on the
    # column names provided; for example, if "pc_11" is in the column names,
    # then the fit pca object needs to be fit with at least 11 pcs
    num_pcs = get_num_pcs_from_column_names(column_names)
    if num_pcs == 0:
        raise ValueError(f"No PC-related column names found in {column_names}.")

    # Fit PCA object for given number of PCs
    pca = fit_pca(num_pcs=num_pcs)

    # re-transform coordinates if they are in polar format (angle and radius) or
    # if they include flipped pc3
    dataframe = add_pc_coordinates_to_dataframe(dataframe, column_names)

    # get latent coordinates by performing inverse PCA transformation on the PC
    # coordinates from the dataframe; only use the PC columns needed for the
    # inverse transformation based on the number of PCs determined earlier
    pc_column_names = DIFFAE_PC_COLUMN_NAMES[:num_pcs]
    latent_coords = pca.inverse_transform(dataframe[pc_column_names].to_numpy())

    reconstructed_image = generate_from_coords(
        model,
        latent_coords,
        num_gpus=num_gpus,
        random_seed=random_seed,
        n_noise_samples=n_noise_samples,
        average=average,
    )

    return reconstructed_image


def generate_from_coords(
    model: "BaseDiffusionAutoEncoder | DiffusionAutoEncoder",
    coords: np.ndarray,
    n_noise_samples: int = 1,
    average: bool = False,
    num_gpus: int | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    """Generate a synthetic image from coordinates in the latent space of a model.

    Parameters
    ----------
    model
        The model to use for generation.
    coords
        An array of shape (num_vecs, num_dims) containing latent space coordinates.
    n_noise_samples
        The number of noise samples to use for generation.
    average
        Whether to average the generated images.
    num_gpus
        Optional, number of available GPUs.
    random_seed
        Random seed for generating noise. Only available for endo-specific
        DiffusionAutoEncoder model instances.

    Returns
    -------
    :
        The generated image(s).

    """
    if not isinstance(coords, np.ndarray):
        if isinstance(coords, list):
            coords_np = np.array(coords)
        else:
            logger.error("Parameter `coords` must be a numpy array or a list of lists.")
            raise ValueError("coords must be a numpy array or a list of lists")
    else:
        coords_np = coords

    coords_torch = torch.from_numpy(coords_np).float()

    # move model and inputs to gpu if available, else
    # perform reconstruction on cpu
    device = "cpu" if num_gpus is None else "cuda"
    coords_ = coords_torch.to(device)
    model_ = model.to(device)

    if isinstance(model_, DiffusionAutoEncoder):
        reconstructed_image = model_.generate_from_latent(
            coords_,
            n_noise_samples=n_noise_samples,
            average=average,
            save=False,
            random_seed=random_seed,
        )
    else:
        reconstructed_image = model_.generate_from_latent(
            coords_, n_noise_samples=n_noise_samples, average=average, save=False
        )

    if isinstance(reconstructed_image, torch.Tensor):
        reconstructed_image_array = reconstructed_image.detach().cpu().numpy()
    elif isinstance(reconstructed_image, np.ndarray):
        reconstructed_image_array = reconstructed_image

    # remove any singleton dimensions
    reconstructed_image_array = reconstructed_image_array.squeeze()

    # reshape if n_noise_samples > 1; the image shape is returned as
    # (width, height*num_noise_samples) if n_noise_samples > 1,
    # so reshape to (num_noise_samples, width, height)
    if n_noise_samples > 1:
        image_sample_list = []
        image_height = reconstructed_image_array.shape[-1] // n_noise_samples
        for sample in range(n_noise_samples):
            image_sample_list.append(
                reconstructed_image_array[:, sample * image_height : (sample + 1) * image_height]
            )
        reconstructed_image_array = np.stack(image_sample_list, axis=0)

    return reconstructed_image_array


def generate_latent_walk_images(
    model: "DiffusionAutoEncoder",
    walk: np.ndarray,
    ranges: np.ndarray,
    n_noise_samples: int = 1,
    num_gpus: int | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    """Generate images from a latent walk using the provided model.

    Parameters
    ----------
    model
        Model to use for image generation.
    walk
        Array of shape (num_steps, num_dims) containing the latent walk
        coordinates.
    ranges
        Array of shape (num_dims, num_steps) containing the coordinate values
        for each dimension and step. Used to reshape the array of generated
        images.
    n_noise_samples
        Number of noise samples to use for generating images.
    num_gpus
        Number of GPUs to use for image generation. If None, uses CPU.
    random_seed
        Random seed for reproducibility of image generation. If None, does not
        set a random seed.

    Returns
    -------
    :
        Array of stacked generated images from the latent walk, reshaped to
        (n_dim, n_steps, img_width, img_height).

    """
    walk_img = generate_from_coords(
        model, walk, n_noise_samples=n_noise_samples, num_gpus=num_gpus, random_seed=random_seed
    )

    # Reshape to (n_dim, n_steps, img_w, img_h)
    n_dim = ranges.shape[0]
    n_steps_actual = ranges.shape[1]
    image_width = walk_img.shape[-2]
    image_height = walk_img.shape[-1]

    return walk_img.reshape(n_dim, n_steps_actual, image_width, image_height)
