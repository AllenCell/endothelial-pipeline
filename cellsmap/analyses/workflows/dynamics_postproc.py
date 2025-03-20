# %%
import numpy as np
import pandas as pd
import pysindy as ps
import numdifftools as nd

from cellsmap.analyses.utils import manifest_viz, manifest_io, manifest_pca, regression_main

# import config parameters
from cellsmap.analyses.configs.dynamics_postproc_config import savedir, PCs, ds_to_skip, ndim, Nbins

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
