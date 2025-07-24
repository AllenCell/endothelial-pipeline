TAGS = ["production", "stochastic_dynamics", "diffae_features"]


def main(dynamics_config_name: str = "default", model_name: str = "diffae_04_10") -> None:
    """
    Fit SINDy (polynomial regression) models for drift and diffusion terms using
    the training data generated in the previous step of the workflow
    (`build_sde_model_train_and_test.py`).

    Parameters
    ----------
    dynamics_config_name
        Name of the configuration to load from dynamics_config.yaml.
        Default is "default".
    model_name
        Name of the model from which manifest data is loaded and analyzed.
        Default is "diffae_04_10".

    Returns
    -------
    None
        Saves the trained models for drift and diffusion terms in a specified directory.
        Saved out as a dictionary with keys "drift_model" and "diff_model", where the values
        are the trained regression models for the drift and diffusion terms, respectively.
    """
    import logging

    import pysindy as ps

    from src.endo_pipeline.configs import dynamics_io
    from src.endo_pipeline.io import get_output_path
    from src.endo_pipeline.library.analyze.diffae_features import (
        build_diff_lib,
        build_drift_lib,
        load_train_test,
        save_sde_model,
    )

    logger = logging.getLogger(__name__)

    ################### Load configs from dynamics_config ###################
    logger.info("*** Running workflow using workflow input config: [ %s ]", dynamics_config_name)
    dynamics_config = dynamics_io.load_dynamics_config(dynamics_config_name)

    # get output subdirectory for intermediate workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    savedir = get_output_path(
        "stochastic_dynamics", dynamics_config_name, model_name, "outputs", include_timestamp=False
    )

    # get inputs for regression from config
    pcs = dynamics_config["pcs_to_analyze"]
    dt = dynamics_config["dt"]
    drift_deg = dynamics_config["polynomial_lib"]["drift_feats"]
    diff_deg = dynamics_config["polynomial_lib"]["diffusion_feats"]
    param_deg_drift = dynamics_config["polynomial_lib"]["drift_param"]
    param_deg_diff = dynamics_config["polynomial_lib"]["diffusion_param"]

    ################### Load train test data from file ###################
    logger.debug(
        "Loading train and test data for regression from precomputed file [ %s ]",
        savedir / "train_test_data.npz",
    )
    train_test_dict = load_train_test(savedir / "train_test_data.npz")

    ################### Build SINDy libraries ###################
    # for fitting model of drift and diffusion terms
    drift_lib = build_drift_lib(ndim=len(pcs), drift_deg=drift_deg, param_deg=param_deg_drift)

    diff_lib = build_diff_lib(ndim=len(pcs), diff_deg=diff_deg, param_deg=param_deg_diff)
    ################### Fit SINDy models ###################

    # fit model for drift term - SINDy based regression
    drift_model = ps.SINDy(feature_library=drift_lib, optimizer=ps.SSR())
    drift_model.fit(
        train_test_dict["x_train"],
        t=dt,
        x_dot=train_test_dict["y_train"],
        u=train_test_dict["u_train"],
    )

    # score on test set
    drift_r2 = drift_model.score(
        train_test_dict["x_test"],
        x_dot=train_test_dict["y_test"],
        u=train_test_dict["u_test"],
    )
    # how to redirect these print statements to logger?
    drift_model.print()

    logger.info("Coefficient of determination (R^2) for model of drift term: [ %.4f ]", drift_r2)

    # fit model for diffusion term - SINDy based regression
    diff_model = ps.SINDy(feature_library=diff_lib, optimizer=ps.SSR())
    diff_model.fit(
        train_test_dict["x_train"],
        t=dt,
        x_dot=train_test_dict["v_train"],
        u=train_test_dict["u_train"],
    )

    # score on test set
    diff_r2 = diff_model.score(
        train_test_dict["x_test"],
        x_dot=train_test_dict["v_test"],
        u=train_test_dict["u_test"],
    )
    diff_model.print()

    logger.info("Coefficient of determination (R^2) for model of diffusion term: [ %.4f ]", diff_r2)

    ################### Save trained models ###################
    model_dict = {"drift_model": drift_model, "diff_model": diff_model}
    save_sde_model(model_dict, savedir)


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
