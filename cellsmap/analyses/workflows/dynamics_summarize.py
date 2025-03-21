# %%
from cellsmap.analyses.utils import dynamics_io, manifest_io
from cellsmap.analyses.configs.manifest_postproc_config import savedir, ds_to_skip

# %%
model_dict = dynamics_io.load_model(savedir+'outputs/drift_diffusion_model.pkl')

driftModel = model_dict['driftModel']
diffModel = model_dict['diffModel']
# %%
list_of_datasets = manifest_io.load_list_of_datasets(savedir+'outputs/')

for ds_name in list_of_datasets: 
    # if we don't want to fit model using this dataset, skip it
    if ds_name in ds_to_skip:
        print('**** Skipping dataset',ds_name,'**** \n')
        continue

    print('**** Running model analysis for dataset',ds_name,'**** \n')

    # project data from this one dataset onto PCs as defined by fit PCA object pca
    df_proj = manifest_io.project_PCA_one_dataset(df,pca,'group',my_mv)

    # for extracting just the PCs we want from the dataframe
    feat_cols = [str(i) for i in PCs]

    # get 2-pt trajectories and for each flow condition present in the dataset as well as the flow conditions themselves
    traj_list, flow_list = eareg.get_2pt_traj_and_flow(df_proj,mv_name,feat_cols=feat_cols,verbose=True)
    del df_proj # free up memory
    num_flow = len(flow_list)

    
    for j in range(num_flow): # get bins and centers for data at high and low flow    
        print('**** Shear stress u =',flow_list[j],'dyn/cm^2 **** \n')
        plot_tuple = model_analysis.run_model_analysis_2D(myModel,traj_list[j],bins,centers,flow_list[j],args=plt_args)

