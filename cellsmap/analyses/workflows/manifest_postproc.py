# %%
from pathlib import Path

from cellsmap.analyses.utils import manifest_pca, regression_main
from cellsmap.analyses.utils.io import dynamics_io, manifest_io
from cellsmap.analyses.utils.viz import manifest_viz, viz_base as vb

# %%
################### Load manifest data and fit PCA ###################
# make save directory for workflow outputs (set in config file dynamics_config.yaml)
config = dynamics_io.load_dynamics_config()
assert "output_subdir" in config, "output_subdir not found in dynamics_config.yaml"

# get head of analyses folder in cellsmap repo
analyses_folder = Path(__file__).resolve().parent.parent
savedir = str(analyses_folder / 'results' / config["output_subdir"])+'/' # directory to save results

manifest_io.make_savedir(savedir)

# figures saved into folder at head of repo
parent_folder = analyses_folder.parent
fig_savedir = str(parent_folder / 'figs')+'/'
# if figs directory does not exist, create it (make_savedir function will not overwrite existing directory)
manifest_io.make_savedir(fig_savedir)

# load manifest to DataFrame with metadata
df = manifest_io.load_manifest_to_df()

# fit PCA to data
df, pca = manifest_pca.fit_pca(df, num_pcs=8)

# save out PCA object (need later for analysis and summary of fit dynamical systems model)
manifest_io.save_pca_model(pca, savedir)
# %%
################### Visualize PCA results ###################
# plot explained variance ratio of PCA components
fig, ax = manifest_viz.plot_explained_variance(pca['pca'].explained_variance_ratio_)
vb.save_plot(fig,filename=fig_savedir+'explained_variance_ratio',format='.png',dpi=500)

# plot top 3 principal components of feature data vs. frame number
fig, axs = manifest_viz.plot_top_3_PCs_alldata(df,pca)
vb.save_plot(fig,filename=fig_savedir+'top_3_PCs',format='.png',dpi=500)

# %%
################### Build train-test data for regression ###################
# load inputs from dynamics_config.yaml
PCs = config['PCs_to_analyze']
Nbins = config['Nbins_kramers_moyal']
dt = config['dt']
ds_to_skip = config['datasets_to_skip']

# build train-test data for regression
train_test_dict = regression_main.build_kramers_moyal_train_test(df, pca, PCs, Nbins, dt, ds_to_skip)

# %%
################### Save train-test data ###################
dynamics_io.save_train_test(train_test_dict, savedir)

# %%
