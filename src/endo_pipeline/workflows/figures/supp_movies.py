from endo_pipeline.cli import UniqueIntList


def main(include_movies: UniqueIntList | None = None) -> None:
    """
    **Supplemental Movies**. Timelapse movies of VE-cadherin maximum intensity
    Z-projection, brightfield, and brightfield standard deviation Z-projections

    #supp-movies

    ## Example usage

    To run the movie workflow:

    ```bash
    uv run endopipe supp-movies
    ```

    To run the movie workflow for a specific movie:

    ```bash
    uv run endopipe supp-movies MOVIE
    ```

    ## Movie types

    Movies are one of four types:

    - `stitched` = all positions stitched together for EGFP and BF std dev
      channels, filtered to steady-state timepoints
    - `fov` = FOV of position for EGFP, BF, and BF std dev channels, cropped to
      match figure, filtered to steady-state timepoints
    - `inset` = inset of position for EGFP, BF, and BF std dev channels, cropped
      to match figure
    - `merge` = EGFP and BF channels along with merged EGFP+BF

    Parameters
    ----------
    include_movies
        List of movies to generate. Leave empty to generate all movies.
    """

    from functools import partial

    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.timelapse_movies import (
        create_fov_timelapse_movie_for_example,
        create_inset_timelapse_movie_for_example,
        create_merge_timelapse_movie_for_example,
        create_stitched_timelapse_movie_for_example,
    )
    from endo_pipeline.settings.examples import (
        FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES,
        FIGURE_3_EXAMPLE_IMAGES,
        FIGURE_5_EXAMPLE_IMAGES,
        SUPP_FIG_RETRACTION_FIBER_BLOB,
    )

    output_path = get_output_path(__file__)

    # Build list of all supplemental movies
    movie_examples_and_types = []

    # Figure 1: S1-S6 (2 examples x 3 videos: stitched + FOV + inset)
    for example in FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES:
        movie_examples_and_types.append((example, "stitched"))
        movie_examples_and_types.append((example, "fov"))
        movie_examples_and_types.append((example, "inset_figure_1"))

    # Figure 3: S7-S16 (4 unique stitched + 6 FOV; skip stitched for datasets in Figure 1)
    fig1_datasets = {example.dataset_name for example in FIGURE_1_BIO_SYSTEM_EXAMPLE_IMAGES}
    for example in FIGURE_3_EXAMPLE_IMAGES:
        if example.dataset_name not in fig1_datasets:
            movie_examples_and_types.append((example, "stitched"))
        movie_examples_and_types.append((example, "fov"))

    # Figure 5: S17-S21 (3 examples x 2 videos: stitched + FOV, + 1 inset for
    # knock_down; skip stitched and fov for datasets in Figure 1)
    for example in FIGURE_5_EXAMPLE_IMAGES:
        if example.dataset_name not in fig1_datasets:
            movie_examples_and_types.append((example, "stitched"))
            movie_examples_and_types.append((example, "fov"))
        if example.description == "knock_down":
            movie_examples_and_types.append((example, "inset_knock_down"))

    # Supp Figure 8: S22 (retraction fiber blob crop)
    movie_examples_and_types.append((SUPP_FIG_RETRACTION_FIBER_BLOB, "merge"))

    # Generate all movies if specific movies are not selected
    include_movies = include_movies or list(range(1, len(movie_examples_and_types) + 1))

    # Define movie builders
    movie_builder_map = {
        "stitched": partial(
            create_stitched_timelapse_movie_for_example,
            output_path=output_path,
        ),
        "fov": partial(
            create_fov_timelapse_movie_for_example,
            output_path=output_path,
            crop_size=1000,
        ),
        "inset_figure_1": partial(
            create_inset_timelapse_movie_for_example,
            output_path=output_path,
            crop_size=256,
            crop_x_offset=5,
            crop_y_offset=372,
        ),
        "inset_knock_down": partial(
            create_inset_timelapse_movie_for_example,
            output_path=output_path,
            crop_size=256,
            crop_x_offset=50,
            crop_y_offset=500,
        ),
        "merge": partial(
            create_merge_timelapse_movie_for_example,
            output_path=output_path,
            crop_size=400,
            timepoint_offset=15,
        ),
    }

    # Run all selected movies
    for movie in include_movies:
        example, movie_type = movie_examples_and_types[movie - 1]
        movie_builder = movie_builder_map[movie_type]
        movie_builder(example=example, file_prefix=f"supp_movie_{movie}_")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
