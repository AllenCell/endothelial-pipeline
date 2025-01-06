import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def get_feats_proj(df, my_mv, pca):
    '''Given a dataframe with features, a dataset (movie) name, and a PCA object, 
    return the PCA-transformed features for the movie as an array of shape (num_crops,num_T,num_PCs).
    '''
    df_one_mv = df[df['filename_or_obj'] == my_mv].copy()
    start_x = df_one_mv[df_one_mv['T']==0]['start_x'].values.tolist()
    start_y = df_one_mv[df_one_mv['T']==0]['start_y'].values.tolist()
    tup_list = list(zip(start_x,start_y))

    def pos_to_index(x,y):
        return tup_list.index((x,y))

    df_one_mv['crop_index'] = df_one_mv.apply(lambda x: pos_to_index(x['start_x'],x['start_y']),axis=1)

    df_one_mv.sort_values(['crop_index','T'])
    num_T = df_one_mv['T'].nunique()
    num_crop = df_one_mv['crop_index'].nunique()

    feats_proj = df_one_mv.drop(columns = ['start_x','start_y','filename_or_obj','T','crop_index']).astype(float)
    feats_proj = pca.transform(feats_proj).reshape(num_T,num_crop,-1)
    feats_proj = np.swapaxes(feats_proj,0,1)
    return feats_proj

# path to the data with the features for various datasets
path_to_data = '//allen/aics/assay-dev/users/Benji/CurrentProjects/im2im_dev/cyto-dl/logs/eval/runs/diffae/large_latent_large_encoder/2024-11-25_09-43-32/patched.parquet'
df = pd.read_parquet(path_to_data) # read in the data

# drop columns that are not features
df_ = df.drop(columns = ['T','start_x','start_y','filename_or_obj']).astype(float)
pca = PCA(n_components=df_.shape[1]) # initialize PCA object - full PCA (no dimensionality reduction, slice to get individual PCs)
pca.fit((df_ - df_.mean()) / df_.std()) # fit to normalized data

del df_ # free up memory (only necessary if working on slurm-vscode with limited memory)

list_of_datasets = np.unique(df['filename_or_obj'].values).tolist() # list of all datasets in the dataframe
mv = list_of_datasets[0] # choose a dataset to extract the PCs for
feats_proj = get_feats_proj(df,mv)




