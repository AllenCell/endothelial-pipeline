# %%
import numpy as np
import pandas as pd
import pysindy as ps
import numdifftools as nd

import cellsmap.analyses.utils.gen_potential as gp
import cellsmap.analyses.utils.regression as eareg
from cellsmap.analyses.utils import manifest_viz, manifest_io, manifest_pca

# %%
# load data
df = manifest_io.load_manifest_to_df()
# fit PCA to data
df, pca = manifest_pca.get_pca(df, num_pcs=8)
# add outliers to dataframe
df = manifest_pca.get_outliers(df)
# filepath for this dataset in manifest includes barcode, so we need to change the group name to match data config
# something that should be fixed in the manifest in the future
df.loc[df.group.str.contains('20250224'),'group'] = '20250224_20X'
# get list of datasets by 'group' identifier
list_of_datasets = manifest_io.get_list_of_datasets(df,'group',verbose=True)

# plot explained variance ratio of PCA components
fig, ax = manifest_viz.plot_explained_variance(pca['pca'].explained_variance_ratio_)

# %%
# plot top 3 principal components of feature data vs. frame number
# should write a function to do this for all datasets in the manifest via data_config
title_dict = {'20241016_20X':'24H High, 24H Low',
              '20241105_20X':'24H Low, 24H High (11/5/24)',
              '20241120_20X':'48H High',
              '20241203_20X':'48H Low',
              '20241210_20X':'48H No Flow 1',
              '20241217_20X':'48H No Flow 2',
              '20250224_20X':'24H Low, 24H High (2/24/25)',}

fig, axs = manifest_viz.plot_top_3_PCs_alldata(df,pca['pca'],list_of_datasets, title_dict)