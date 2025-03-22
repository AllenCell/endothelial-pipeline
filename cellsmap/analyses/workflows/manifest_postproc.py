# %%
from cellsmap.analyses.utils import manifest_pca, regression_main
# import config parameters
from cellsmap.analyses.configs.manifest_postproc_config import savedir, PCs, ds_to_skip, Nbins, dt
from cellsmap.analyses.utils.io import dynamics_io, manifest_io
from cellsmap.analyses.utils.viz import manifest_viz, viz_base as vb

# %%
# make save directory for workflow outputs (set in config file manifest_postproc_config.py)
manifest_io.make_savedir(savedir)
# %%
# load manifest to DataFrame with metadata
df = manifest_io.load_manifest_to_df()
# fit PCA to data
df, pca = manifest_pca.fit_pca(df, num_pcs=8)
# save out PCA object (need later for analysis and summary of fit dynamical systems model)
manifest_io.save_pca_model(pca, savedir+'outputs/')
#%%
# plot explained variance ratio of PCA components
fig, ax = manifest_viz.plot_explained_variance(pca['pca'].explained_variance_ratio_)
vb.save_plot(fig,filename=savedir+'figs/explained_variance_ratio',format='.png',dpi=500)
# %%
# plot top 3 principal components of feature data vs. frame number
fig, axs = manifest_viz.plot_top_3_PCs_alldata(df,pca)
vb.save_plot(fig,filename=savedir+'figs/top_3_PCs',format='.png',dpi=500)

# %%
train_test_dict = regression_main.build_kramers_moyal_train_test(df, pca, PCs, Nbins, dt, ds_to_skip)

# %%
dynamics_io.save_train_test(train_test_dict, savedir+'outputs/')
