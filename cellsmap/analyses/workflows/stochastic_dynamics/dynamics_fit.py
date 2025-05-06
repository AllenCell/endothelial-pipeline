import fire
import pysindy as ps  # import pysindy package for SINDy based regression

from cellsmap.analyses.utils import model_fitting
from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.util.set_output import get_output_path


def main(config_name: str = "default") -> None:
    """
    Fit SINDy (polynomial regression) models for drift
    and diffusion terms using the training data generated
    in the previous step of the workflow
    (cellsmap/analyses/workflows/stochastic_dynamics/dynamics_preproc.py).

    Input:
    - config_name (str): Name of the configuration to load from dynamics_config.yaml.
        Default is "default".

    Output:
    - Saves the trained models for drift and diffusion terms in a specified directory.
        Saved out as a dictionary with keys "driftModel" and "diffModel",
        where the values are the trained models for the drift and
        diffusion terms, respectively.
    """
    ################### Load configs from dynamics_config ###################
    config = dynamics_io.load_dynamics_config(config_name)

    # get output subdirectory for intermediate workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_output_folder = "stochastic_dynamics/" + config["name"] + "/outputs"
    savedir = get_output_path(workflow_output_folder, verbose=False)

    # get inputs for regression from config
    pcs = config["pcs_to_analyze"]
    dt = config["dt"]
    drift_deg = config["polynomial_lib"]["drift_feats"]
    diff_deg = config["polynomial_lib"]["diffusion_feats"]
    param_deg_drift = config["polynomial_lib"]["drift_param"]
    param_deg_diff = config["polynomial_lib"]["diffusion_param"]

    ################### Load train test data from file ###################
    train_test_dict = dynamics_io.load_train_test(savedir + "train_test_data.npz")

    ################### Build SINDy libraries ###################
    # for fitting model of drift and diffusion terms
    drift_lib = model_fitting.build_drift_lib(
        ndim=len(pcs), drift_deg=drift_deg, param_deg=param_deg_drift
    )

    diff_lib = model_fitting.build_diff_lib(
        ndim=len(pcs), diff_deg=diff_deg, param_deg=param_deg_diff
    )
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
    drift_model.print()

    print(f"Coefficient of determination (R^2) for model of drift term: {drift_r2:.6f}")

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

    print(
        f"Coefficient of determination (R^2) for model of diffusion term: {diff_r2:.6f}"
    )

    ################### Save trained models ###################
    model_dict = {"drift_model": drift_model, "diff_model": diff_model}
    dynamics_io.save_model(model_dict, savedir)


if __name__ == "__main__":
    fire.Fire(main)
