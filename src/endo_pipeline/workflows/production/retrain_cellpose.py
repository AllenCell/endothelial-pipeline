def main(num_processes: int = 1) -> None:
    """
    Retrain Cellpose model to predict nuclei from BF std dev projections.

    #model-training #cellpose #test-ready #gpu

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe retrain-nuclei-prediction-model -vd
    ```

    ## Dataset collection

    This workflow uses datasets in the `cellpose_model_training` dataset
    collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will train on a
    subset of the training data. The resulting model with have `_demo` prefix
    and is not suitable for label-free nuclei prediction.

    Parameters
    ----------
    num_processes
        Number of processes to use.
    """

    import logging

    import matplotlib.pyplot as plt
    from cellpose import models, train
    from cellpose.io import logger_setup

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_image, load_model, make_name_unique
    from endo_pipeline.library.process.general_image_preprocessing import build_analysis_queue
    from endo_pipeline.library.process.lib_nuc_pred_from_bf_std_retraining import (
        load_train_and_test_images,
        save_labelfree_nuclei_example_image,
        save_training_test_loss_plot,
    )
    from endo_pipeline.manifests import (
        ModelLocation,
        get_zarr_location_for_position,
        load_model_manifest,
        save_model_manifest,
    )
    from endo_pipeline.settings import DIMENSION_ORDER

    logger = logging.getLogger(__name__)

    model_name = make_name_unique("labelfree_nuc_pred").stem
    out_dir = get_output_path("models", "labelfree_nuc_pred", include_timestamp=False)

    datasets_to_use = get_datasets_in_collection("cellpose_model_training")

    if DEMO_MODE:
        datasets_to_use = datasets_to_use[:1]
        model_name += "_demo"

    analysis_queue = build_analysis_queue(
        datasets_to_use,
        out_dir=out_dir,
        image_validation_frequency=1,
        t_start=0,
        t_final=1,
        max_positions=5 if DEMO_MODE else None,
        save_output=True,
        overwrite=True,
    )

    # Load training and testing images
    images_training, labels_training, images_testing, labels_testing = load_train_and_test_images(
        out_dir=out_dir,
        analysis_queue=analysis_queue,
        num_processes=num_processes,
    )

    logger.info("Beginning training...")
    sgd = True
    learning_rate = 0.1
    weight_decay = 1e-4
    n_epochs = 10 if DEMO_MODE else 300

    # initiate the cellpose logger so that we
    # can extract the training and test losses
    logger_setup(cp_path=(out_dir / "logs").as_posix(), logfile_name=f"{model_name}.log")

    # will populate this dictionary as we go
    run_record = {}

    model_nuclei_original = models.CellposeModel(gpu=True, model_type="nuclei")

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

    # save the model to the model manifest
    model_manifest = load_model_manifest("nuc_pred_labelfree")
    model_location = ModelLocation(path=model_path)
    model_manifest.locations[model_name] = model_location
    save_model_manifest(model_manifest)

    # generate a test image to see how the model performs
    # on a live example that it has never seen
    model_nuclei_original_finetuned = load_model(model_location)

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
