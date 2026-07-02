"""Generate supplemental timelapse movies for Figure 1, 3, and 5 examples.

Movies are created for each channel type (EGFP, BF, BF_std_dev),
cropped to the same region shown in the figures, and filtered to steady-state timepoints.

#supp-movies #figures

## Example usage

```bash
uv run endopipe supp-movies
```

To run in demo mode (5 timepoints per movie):

```bash
uv run endopipe supp-movies -d
```
"""

from multiprocessing import Pool
from pathlib import Path

import imageio
import numpy as np
from tqdm import tqdm

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_unannotated_timepoints_for_position,
    load_dataset_config,
)
from endo_pipeline.io import get_output_path
from endo_pipeline.library.process.image_processing import (
    convert_to_uint8,
    crop_image,
    load_processed_bf_image,
    load_processed_egfp_image,
)
from endo_pipeline.library.visualize.timelapse_movies import (
    add_scalebar_to_frame,
    add_timestamp_to_frame,
    create_multichannel_timelapse_movie,
    load_stitched_image,
)
from endo_pipeline.settings.examples import (
    FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES,
    FIGURE_3_EXAMPLE_IMAGES,
    FIGURE_5_EXAMPLE_IMAGES,
    SUPP_FIG_RETRACTION_FIBER_BLOB,
)
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x

FRAMES_PER_SECOND = 30


def _get_steady_state_timepoints(example, demo_mode: bool = False) -> list[int]:
    """Get filtered timepoints for an example."""
    dataset_config = load_dataset_config(example.dataset_name)
    timepoints = get_unannotated_timepoints_for_position(
        dataset_config,
        example.position,
        annotations=[
            TimepointAnnotation.NOT_STEADY_STATE,
            TimepointAnnotation.CELL_PILING,
        ],
    )
    if demo_mode:
        timepoints = timepoints[:5]
    return timepoints


def _render_movie(kwargs: dict) -> None:
    """Wrapper for multiprocessing — calls create_multichannel_timelapse_movie."""
    create_multichannel_timelapse_movie(**kwargs)


def _render_retraction_fiber_blob_movie(kwargs: dict) -> None:
    """Render RGB movie with pseudo-colored GFP (green), BF (grayscale), and merge panels.

    Matches the coloring used in the retraction fiber blob figure panel:
    - GFP: pseudo-colored green
    - BF: grayscale (displayed as RGB)
    - Merge: BF grayscale with GFP overlaid via max in green channel
    """
    from functools import partial

    dataset_config = load_dataset_config(kwargs["dataset_name"])
    positions = kwargs["positions"]
    timepoints = kwargs["timepoints"]
    output_path = kwargs["output_path"]
    file_name = kwargs["file_name"]
    crop_region = kwargs["crop_region"]
    scale_bar_um = kwargs["scale_bar_um"]
    frames_per_second = kwargs["frames_per_second"]
    annotate_shear_stress = kwargs.get("annotate_shear_stress", True)
    gap_px = kwargs.get("gap_px", 4)
    layout = kwargs.get("layout", "horizontal")

    load_gfp = partial(
        load_stitched_image,
        loader=load_processed_egfp_image,
        config=dataset_config,
        positions=positions,
    )
    load_bf = partial(
        load_stitched_image,
        loader=load_processed_bf_image,
        config=dataset_config,
        positions=positions,
    )

    pixel_size = PIXEL_SIZE_3i_20x

    with imageio.get_writer(
        output_path / file_name,
        fps=frames_per_second,
        codec="libx264",
        format="FFMPEG",  # type: ignore[arg-type]
        ffmpeg_params=[
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-crf",
            "18",
        ],
    ) as writer:
        for tp in tqdm(timepoints, desc=f"{kwargs['dataset_name']}: Creating RGB movie"):
            gfp = load_gfp(timepoints=tp).squeeze().compute()
            bf = load_bf(timepoints=tp).squeeze().compute()

            if crop_region is not None:
                gfp = crop_image(gfp, crop_region[0], crop_region[1], crop_region[2])
                bf = crop_image(bf, crop_region[0], crop_region[1], crop_region[2])

            gfp = convert_to_uint8(gfp)
            bf = convert_to_uint8(bf)

            # Pseudo-color GFP as green
            zeros = np.zeros_like(gfp)
            gfp_rgb = np.stack([zeros, gfp, zeros], axis=-1)

            # BF as RGB grayscale
            bf_rgb = np.stack([bf, bf, bf], axis=-1)

            # Merge: BF grayscale with GFP overlaid via max in green channel
            g_channel = np.maximum(bf, gfp)
            merge_rgb = np.stack([bf, g_channel, bf], axis=-1)

            panels = [gfp_rgb, bf_rgb, merge_rgb]

            # Combine panels with gap
            if gap_px > 0:
                if layout == "horizontal":
                    gap = np.zeros((panels[0].shape[0], gap_px, 3), dtype=np.uint8)
                    interleaved = [panels[0]]
                    for p in panels[1:]:
                        interleaved.extend([gap, p])
                    composite = np.concatenate(interleaved, axis=1)
                else:
                    gap = np.zeros((gap_px, panels[0].shape[1], 3), dtype=np.uint8)
                    interleaved = [panels[0]]
                    for p in panels[1:]:
                        interleaved.extend([gap, p])
                    composite = np.concatenate(interleaved, axis=0)
            elif layout == "horizontal":
                composite = np.concatenate(panels, axis=1)
            else:
                composite = np.concatenate(panels, axis=0)

            # Pad to even dimensions
            h, w = composite.shape[:2]
            pad_h = h % 2
            pad_w = w % 2
            if pad_h or pad_w:
                composite = np.pad(composite, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

            # Add annotations
            add_scalebar_to_frame(composite, scale_bar_um, pixel_size)
            add_timestamp_to_frame(composite, dataset_config, tp, annotate_shear_stress)

            writer.append_data(composite)  # type: ignore[attr-defined]


def _make_job(
    example,
    timepoints: list[int],
    output_path,
    video_num: int,
    ss_bin: str,
    date: str,
    suffix: str,
    *,
    scale_bar_um: int = 100,
    layout: str = "horizontal",
    positions: list[int] | None = None,
    crop_region: tuple[int, int, int] | None = None,
) -> dict:
    """Build a single movie job dict."""
    job: dict = {
        "dataset_name": example.dataset_name,
        "channel_types": ["EGFP", "BF", "BF_std_dev"],
        "output_path": output_path,
        "layout": layout,
        "gap_px": 4,
        "timepoints": timepoints,
        "frames_per_second": FRAMES_PER_SECOND,
        "annotate_shear_stress": True,
        "scale_bar_um": scale_bar_um,
        "file_name": f"Video_S{video_num}_{ss_bin}dymcm2_{date}_scale_bar{scale_bar_um}um_fps{FRAMES_PER_SECOND}_{suffix}.mp4",
    }
    if positions is not None:
        job["positions"] = positions
    if crop_region is not None:
        job["crop_region"] = crop_region
    return job


def main(figures: list[int] | None = None, output_dir: Path | None = None) -> None:
    """
    Generate supplemental timelapse movies for Figure 1, 3, and 5 examples.

    #supp-movies #figures

    ## Example usage

    ```bash
    uv run endopipe supp-movies -d
    ```

    To run only Figure 5 movies:

    ```bash
    uv run endopipe supp-movies -d --figures 5
    ```

    To run Figure 1 and 5:

    ```bash
    uv run endopipe supp-movies -d --figures 1 5
    ```
    """
    from endo_pipeline.cli import DEMO_MODE

    if figures is None:
        figures = [1, 3, 5, 8]

    output_path = output_dir if output_dir is not None else get_output_path("supp_movies")
    jobs: list[dict] = []
    video_num = 1

    # Figure 1: S1-S6 (2 examples x 3 videos: stitched + FOV + inset)
    if 1 in figures:
        for example in FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES:
            dataset_config = load_dataset_config(example.dataset_name)
            date = dataset_config.date
            ss_bin = dataset_config.flow_conditions[0].shear_stress_bin
            timepoints = _get_steady_state_timepoints(example, demo_mode=DEMO_MODE)

            jobs.append(
                _make_job(
                    example,
                    timepoints,
                    output_path,
                    video_num,
                    ss_bin,
                    date,
                    "stitched",
                    layout="vertical",
                )
            )
            jobs.append(
                _make_job(
                    example,
                    timepoints,
                    output_path,
                    video_num + 1,
                    ss_bin,
                    date,
                    "FOV",
                    positions=[example.position],
                    crop_region=(example.crop_x_start, example.crop_y_start, 1000),
                )
            )
            jobs.append(
                _make_job(
                    example,
                    timepoints,
                    output_path,
                    video_num + 2,
                    ss_bin,
                    date,
                    "inset",
                    positions=[example.position],
                    scale_bar_um=20,
                    crop_region=(example.crop_x_start + 5, example.crop_y_start + 372, 256),
                )
            )
            video_num += 3
    else:
        video_num += len(FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES) * 3

    # Figure 3: S7-S16 (4 unique stitched + 6 FOV; skip stitched for datasets in Figure 1)
    fig1_datasets = {e.dataset_name for e in FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES}
    if 3 in figures:
        for example in FIGURE_3_EXAMPLE_IMAGES:
            dataset_config = load_dataset_config(example.dataset_name)
            date = dataset_config.date
            ss_bin = dataset_config.flow_conditions[0].shear_stress_bin
            timepoints = _get_steady_state_timepoints(example, demo_mode=DEMO_MODE)

            if example.dataset_name not in fig1_datasets:
                jobs.append(
                    _make_job(
                        example,
                        timepoints,
                        output_path,
                        video_num,
                        ss_bin,
                        date,
                        "stitched",
                        layout="vertical",
                    )
                )
                video_num += 1
            jobs.append(
                _make_job(
                    example,
                    timepoints,
                    output_path,
                    video_num,
                    ss_bin,
                    date,
                    "FOV",
                    positions=[example.position],
                    crop_region=(example.crop_x_start, example.crop_y_start, 1000),
                )
            )
            video_num += 1
    else:
        # 6 examples: 4 unique stitched + 6 FOV = 10
        n_unique_fig3 = len(FIGURE_3_EXAMPLE_IMAGES) - len(
            [e for e in FIGURE_3_EXAMPLE_IMAGES if e.dataset_name in fig1_datasets]
        )
        video_num += n_unique_fig3 + len(FIGURE_3_EXAMPLE_IMAGES)

    # Figure 5: S17-S23 (3 examples x 2 videos: stitched + FOV, + 1 inset for knock_down)
    if 5 in figures:
        for example in FIGURE_5_EXAMPLE_IMAGES:
            dataset_config = load_dataset_config(example.dataset_name)
            date = dataset_config.date
            ss_bin = dataset_config.flow_conditions[0].shear_stress_bin
            timepoints = _get_steady_state_timepoints(example, demo_mode=DEMO_MODE)

            jobs.append(
                _make_job(
                    example,
                    timepoints,
                    output_path,
                    video_num,
                    ss_bin,
                    date,
                    "stitched",
                    layout="vertical",
                )
            )
            jobs.append(
                _make_job(
                    example,
                    timepoints,
                    output_path,
                    video_num + 1,
                    ss_bin,
                    date,
                    "FOV",
                    positions=[example.position],
                    crop_region=(example.crop_x_start, example.crop_y_start, 1000),
                )
            )
            video_num += 2

            if example.description == "knock_down":
                jobs.append(
                    _make_job(
                        example,
                        timepoints,
                        output_path,
                        video_num,
                        ss_bin,
                        date,
                        "inset",
                        positions=[example.position],
                        scale_bar_um=20,
                        crop_region=(50, 500, 256),
                    )
                )
                video_num += 1
    else:
        video_num += len(FIGURE_5_EXAMPLE_IMAGES) * 2 + 1  # +1 for knock_down inset

    # Supp Fig 8: S24 (retraction fiber blob crop — RGB with GFP green + BF + merge)
    if 8 in figures:
        example = SUPP_FIG_RETRACTION_FIBER_BLOB
        dataset_config = load_dataset_config(example.dataset_name)
        date = dataset_config.date
        ss_bin = dataset_config.flow_conditions[0].shear_stress_bin
        timepoints = list(range(example.timepoint, example.timepoint + 15))

        rfb_job = {
            "dataset_name": example.dataset_name,
            "positions": [example.position],
            "timepoints": timepoints,
            "output_path": output_path,
            "crop_region": (example.crop_x_start, example.crop_y_start, 400),
            "scale_bar_um": 20,
            "frames_per_second": 15,
            "annotate_shear_stress": True,
            "gap_px": 4,
            "layout": "horizontal",
            "file_name": f"Video_S{video_num}_{ss_bin}dymcm2_{date}_scale_bar{20}um_fps{15}_retraction_fiber_blob.mp4",
        }

    # Render all movies in parallel (cap workers to avoid OOM on stitched videos)
    with Pool(processes=min(8, len(jobs) + (1 if 8 in figures else 0))) as pool:
        pool.map(_render_movie, jobs)

    # Render the RGB retraction fiber blob movie separately
    if 8 in figures:
        _render_retraction_fiber_blob_movie(rfb_job)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
