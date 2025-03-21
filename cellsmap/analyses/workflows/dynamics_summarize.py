# %%
from cellsmap.analyses.utils import dynamics_io, manifest_io, model_analysis
import cellsmap.analyses.utils.regression_helper as rh
from cellsmap.analyses.configs.manifest_postproc_config import savedir, ds_to_skip, PCs
from cellsmap.analyses.configs.dynamics_viz_config import Nbins_plot, bin_limits, pplane_xlim, pplane_ylim

# %%
model_dict = dynamics_io.load_model(savedir+'outputs/drift_diffusion_model.pkl')

driftModel = model_dict['driftModel']
diffModel = model_dict['diffModel']
myModel = [driftModel,diffModel]
# %%
# for plotting phase plane and histogram plots, fix grid and bin limits across all datasets
bins, centers = rh.get_bins(Nbins_plot,bin_limits=bin_limits)

plt_args = {'pplane_xlim': pplane_xlim, 'pplane_ylim': pplane_ylim, 'pplane_N': 50,
            'plt_xlabel': 'PC'+str(PCs[0]+1), 'plt_ylabel': 'PC'+str(PCs[1]+1),
            'truncate_p':[True,[0,Nbins_plot[0]-0],[0,Nbins_plot[1]-0]]}
# %%
df = manifest_io.load_manifest_to_df()
list_of_datasets = manifest_io.get_list_of_datasets(df)
pca = manifest_io.load_pca_model(savedir+'outputs/')

for ds_name in list_of_datasets: 
    # if we don't want to fit model using this dataset, skip it
    if ds_name in ds_to_skip:
        print('**** Skipping dataset',ds_name,'**** \n')
        continue

    print('**** Running model analysis for dataset',ds_name,'**** \n')

    # project data from this one dataset onto PCs as defined by fit PCA object pca
    df_proj = manifest_io.project_PCA_one_dataset(df,pca,ds_name)

    # split out data by flow condition
    df_by_flow, shear_list = rh.get_X_by_flow(df_proj,ds_name)
    del df_proj # free up memory
    num_flow = len(shear_list)

    # for extracting just the PCs we want from the dataframe when passing to model analysis
    feat_cols = [str(i) for i in PCs]
    
    for j in range(num_flow): # get bins and centers for data at high and low flow    
        print('**** Shear stress =',shear_list[j],'dyn/cm^2 **** \n')

        plot_tuple = model_analysis.run_model_analysis_2D(myModel,df_by_flow[j],feat_cols,bins,centers,shear_list[j],args=plt_args)


# %%
