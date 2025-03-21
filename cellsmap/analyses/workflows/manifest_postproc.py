# %%
from cellsmap.analyses.utils import manifest_viz, manifest_io, manifest_pca, regression_main, dynamics_io
# import config parameters
from cellsmap.analyses.configs.manifest_postproc_config import savedir, PCs, ds_to_skip, Nbins, dt

# %%
# make save directory for workflow outputs
manifest_io.make_savedir(savedir)
# %%
# load manifest to DataFrame with metadata
df = manifest_io.load_manifest_to_df()
# fit PCA to data
df, pca = manifest_pca.fit_pca(df, num_pcs=8)
# get list of datasets by 'group' identifier

# %%
# plot explained variance ratio of PCA components
fig, ax = manifest_viz.plot_explained_variance(pca['pca'].explained_variance_ratio_)
manifest_viz.save_plot(fig,filename=savedir+'figs/explained_variance_ratio',format='.png',dpi=500)
# %%
# plot top 3 principal components of feature data vs. frame number
fig, axs = manifest_viz.plot_top_3_PCs_alldata(df,pca)
manifest_viz.save_plot(fig,filename=savedir+'figs/top_3_PCs',format='.png',dpi=500)

# %%
train_test_dict = regression_main.build_kramers_moyal_train_test(df, pca, PCs, Nbins, dt, ds_to_skip)

# %%
dynamics_io.save_train_test(train_test_dict, savedir+'outputs/')


# %%
