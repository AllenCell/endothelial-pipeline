# configuration parameters/inputs for post-processing of manifest data for dynamics analysis

from pathlib import Path

# get head of analyses folder in cellsmap repo
parent_folder = Path(__file__).resolve().parent.parent
dir_name = 'workflow_test' # subfolder in cellsmap/analyses/results directory to save results
savedir = str(parent_folder / 'results' / dir_name)+'/' # directory to save results

# to be passed to the manifest_postproc script
dt = 5 # time between consecutive frames in minutes
PCs=[0,1] # index of the principal components to be used for fitting
ds_to_skip = ['20241016_20X','20241210_20X','20241217_20X'] # names of datasets to skip when fitting SDE model
Nbins = [35,25] # number of bins for each principal component, used for binning data to get Kramers-Moyal coefficients (input to SDE model fitting pipeline)