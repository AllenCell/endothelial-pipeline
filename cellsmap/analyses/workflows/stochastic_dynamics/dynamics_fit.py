import fire
import pysindy as ps # import pysindy package for SINDy based regression

from cellsmap.util.set_output import get_output_path
from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.analyses.utils import model_fitting

def main(config_name:str='default') -> None:
    ################### Load configs from dynamics_config ###################
    config = dynamics_io.load_dynamics_config(config_name)

    # get output subdirectory for intermediate workflow outputs (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_output_folder = "stochastic_dynamics/"+config["name"]+"/outputs"
    savedir = get_output_path(workflow_output_folder,verbose=False)

    # get inputs for regression from config
    PCs = config['PCs_to_analyze']
    dt = config['dt']
    drift_deg = config['polynomial_lib']['drift_feats']
    diff_deg = config['polynomial_lib']['diffusion_feats']
    param_deg_drift = config['polynomial_lib']['drift_param']
    param_deg_diff = config['polynomial_lib']['diffusion_param']


    ################### Load train test data from file ###################
    train_test_dict = dynamics_io.load_train_test(savedir+'train_test_data.npz')

    ################### Build SINDy libraries ###################
    # for fitting model of drift and diffusion terms
    drift_lib = model_fitting.build_drift_lib(ndim=len(PCs),drift_deg=drift_deg,param_deg=param_deg_drift)

    diff_lib = model_fitting.build_diff_lib(ndim=len(PCs),diff_deg=diff_deg,param_deg=param_deg_diff)
    ################### Fit SINDy models ###################

    # fit model for drift term - SINDy based regression
    driftModel = ps.SINDy(feature_library = drift_lib, optimizer = ps.SSR())
    driftModel.fit(train_test_dict['X_train'],t=dt,x_dot=train_test_dict['Y_train'],u=train_test_dict['u_train'])

    # score on test set
    drift_R2 = driftModel.score(train_test_dict['X_test'],x_dot=train_test_dict['Y_test'],u=train_test_dict['u_test'])
    driftModel.print()

    print('Coefficient of determination (R^2) for model of drift term: %f' %drift_R2)

    # fit model for diffusion term - SINDy based regression
    diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
    diffModel.fit(train_test_dict['X_train'],t=dt,x_dot=train_test_dict['V_train'],u=train_test_dict['u_train'])

    # score on test set
    diff_R2 = diffModel.score(train_test_dict['X_test'],x_dot=train_test_dict['V_test'],u=train_test_dict['u_test'])
    diffModel.print()

    print('Coefficient of determination (R^2) for model of diffusion term: %f' %diff_R2)

    ################### Save trained models ###################
    model_dict = {'driftModel':driftModel,'diffModel':diffModel}
    dynamics_io.save_model(model_dict, savedir)

if __name__ == "__main__":
    fire.Fire(main)
