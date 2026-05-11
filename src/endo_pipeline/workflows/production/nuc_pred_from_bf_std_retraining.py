from typing import Any


def main(
    n_proc: int = 1,
    create_training_data: bool = False,
) -> None:
    """
    Run the workflow to retrain a Cellpose model to predict nuclei from brightfield standard
    deviation projections.

    #test-ready #gpu

    To enter a list of datasets to analyze, use the following format:

    .. code-block:: bash

        --datasets 20250818_20X 20250618_20X

    **Workflow demo**

    The ``--demo-mode`` (``-d``) flag can be used to run the workflow on a subset of the training
    data for workflow testing purposes. The resulting model will be named with a ``_DEMO`` suffix
    and does not produce a model that is suitable for label-free nuclei prediction. To produce
    the model for label-free nuclei prediction, run the workflow without the demo mode flag.
    """

    import logging

    import matplotlib.pyplot as plt
    from cellpose import core, models, train
    from cellpose.io import logger_setup

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_image, make_name_unique
    from endo_pipeline.library.process.general_image_preprocessing import build_analysis_queue
    from endo_pipeline.library.process.lib_nuc_pred_from_bf_std_retraining import (
        get_scenes_to_use,
        load_train_and_test_images,
        save_labelfree_nuclei_example_image,
        save_training_test_loss_plot,
    )
    from endo_pipeline.manifests import get_zarr_location_for_position
    from endo_pipeline.settings import DIMENSION_ORDER

    logger = logging.getLogger(__name__)

    model_name = make_name_unique("labelfree_nuc_pred").stem

    if DEMO_MODE:
        create_training_data = True
        model_name += "_DEMO"

    # Create output directory.
    out_dir = get_output_path("models", model_name, include_timestamp=False)

    datasets_to_use = list(get_scenes_to_use().keys())

    analysis_queue = build_analysis_queue(
        datasets_to_use,
        save_output=True,
        image_validation_frequency=1,
        overwrite=True,
        out_dir=out_dir,
        is_test=DEMO_MODE,
    )

    # return whether or not to use a gpu with CellPose
    gpu = core.use_gpu()

    # Load training and testing images
    images_training, labels_training, images_testing, labels_testing = load_train_and_test_images(
        analysis_queue,
        n_proc=n_proc,
        create_training_data=create_training_data,
        gpu=gpu,
    )

    logger.info("Beginning training...")
    sgd = True
    learning_rate = 0.1
    weight_decay = 1e-4
    n_epochs = 300

    # initiate the cellpose logger so that we
    # can extract the training and test losses
    logger_setup(cp_path=out_dir, logfile_name=f"{model_name}.log")

    # will populate this dictionary as we go
    run_record: dict[str, Any] = {}

    model_nuclei_original = models.CellposeModel(gpu=gpu, model_type="nuclei")

    model_path, train_losses, test_losses = train.train_seg(
        model_nuclei_original.net,
        train_data=images_training,
        train_labels=labels_training,
        test_data=images_testing,
        test_labels=labels_testing,
        channels=[0, 0],
        normalize=True,
        weight_decay=weight_decay,
        SGD=sgd,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        save_path=out_dir,
        model_name=model_name,
    )

    run_record[model_name] = {
        "model_path": model_path,
        "train_losses": train_losses,
        "test_losses": test_losses,
    }

    # save the training and test losses to a file
    if any(run_record):
        fig = save_training_test_loss_plot(
            train_losses=run_record[model_name]["train_losses"],
            test_losses=run_record[model_name]["test_losses"],
            model_name=model_name,
            out_dir=out_dir,
        )
        plt.close(fig)

    # generate a test image to see how the model performs
    # on a live example that it has never seen
    model_nuclei_original_finetuned = models.CellposeModel(
        gpu=False, pretrained_model=str(model_path)
    )

    # load the brightfield channel of a test image
    test_dataset_name = get_datasets_in_collection("live_cdh5_seg_based_feat_datasets")[0]
    test_dataset_config = load_dataset_config(test_dataset_name)
    test_zarr_loc = get_zarr_location_for_position(test_dataset_config, position=0)
    test_img_dask_arr = load_image(test_zarr_loc, channels=["BF"], timepoints=0, level=0)
    test_img_dask_arr = test_img_dask_arr.std(axis=DIMENSION_ORDER.index("Z"), keepdims=True)
    test_img_arr = test_img_dask_arr.compute().squeeze()

    # run the model on the test image, we're going to be pretty
    # generous with the flow and cellprob threshold settings
    # just to see what is picked up
    test_prediction, flows, probs = model_nuclei_original_finetuned.eval(
        test_img_arr,
        channels=[0, 0],
        min_size=500,
        flow_threshold=0,
        cellprob_threshold=-6.0,
    )

    # plot and save the resulting nuclei prediction
    fig = save_labelfree_nuclei_example_image(
        original_bf_img_array=test_img_arr,
        nuclei_prediction_img_arr=test_prediction,
        model_name=model_name,
        out_dir=out_dir,
    )
    plt.close(fig)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
