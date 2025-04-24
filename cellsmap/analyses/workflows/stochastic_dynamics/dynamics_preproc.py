import fire

from cellsmap.util import manifest_io, manifest_pca
from cellsmap.util.set_output import get_output_path
from cellsmap.analyses.utils import regression_main
from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.analyses.utils.viz import manifest_viz, viz_base as vb

def main(config_name:str='default') -> None:
    ################### Load manifest data and fit PCA ###################
    # make save directory for workflow outputs (set in config file dynamics_config.yaml)
    print("\n","*** Running workflow using config: ", config_name,"\n")
    config = dynamics_io.load_dynamics_config(config_name)

    # get output subdirectory for intermediate workflow outputs (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_output_folder = "stochastic_dynamics/"+config["name"]+"/outputs"
    savedir = get_output_path(workflow_output_folder)

    # get output subdirectory for figures that workflow outputs (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_fig_folder = "stochastic_dynamics/"+config["name"]+"/figs"
    fig_savedir = get_output_path(workflow_fig_folder)

    # load manifest to DataFrame with metadata
    df = manifest_io.load_manifest_to_df()

    # fit PCA to data
    pca = manifest_pca.fit_pca(df, num_pcs=8)

    # save out PCA object (need later for analysis and summary of fit dynamical systems model)
    manifest_io.save_pca_model(pca, savedir)

    ################### Visualize PCA results ###################
    # plot explained variance ratio of PCA components
    fig, _ = manifest_viz.plot_explained_variance(pca['pca'].explained_variance_ratio_)
    vb.save_plot(fig,filename=fig_savedir+'explained_variance_ratio',format='.png',dpi=500)

    # plot top 3 principal components of feature data vs. frame number
    fig, _ = manifest_viz.plot_top_3_PCs_alldata(df,pca)
    vb.save_plot(fig,filename=fig_savedir+'top_3_PCs',format='.png',dpi=500)

    ################### Build train-test data for regression ###################
    # load inputs from dynamics_config.yaml
    PCs = config['PCs_to_analyze']
    dt = config['dt']
    ds_to_skip = config['datasets_to_skip']
    kramers_moyal_config = config['kramers_moyal']
    Nbins = kramers_moyal_config['Nbins']
    km_method = kramers_moyal_config['method']
    kernel_params=None
    if 'kernel_params' in kramers_moyal_config:
        kernel_params = kramers_moyal_config['kernel_params']

    # build train-test data for regression
    train_test_dict = regression_main.build_kramers_moyal_train_test(df, pca, PCs, Nbins, dt, ds_to_skip, fig_savedir,
                                                                     method=km_method, kernel_params=kernel_params)

    ################### Save train-test data ###################
    dynamics_io.save_train_test(train_test_dict, savedir)

if __name__ == "__main__":
    fire.Fire(main)