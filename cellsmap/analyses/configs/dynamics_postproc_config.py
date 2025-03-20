# configuration parameters/inputs for post-processing of manifest data for dynamics analysis

from pathlib import Path

# get head of analyses folder in cellsmap repo
parent_folder = Path(__file__).resolve().parent.parent
savedir = str(parent_folder / 'dynamics_output')+'/' # directory to save results

# to be passed to the dynamics_fit.py script
PCs=[0,1] # index of the principal components to be used for fitting
ds_to_skip = ['20241016_20X','20241210_20X','20241217_20X'] # names of datasets to skip when fitting SDE model
Nbins = [35,25] # number of bins for each principal component, used for binning data to get Kramers-Moyal coefficients (input to SDE model fitting pipeline)