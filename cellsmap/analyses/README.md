# Workflows for fitting stochastic dynamical model to time series data of endothelial cells

## Data formatting for analysis

To start, ensure that your time series data are saved in a `.csv` or `.parquet` file where the columns represent the extracted features (i.e., observed variables), the rows represent inividual data points. There may be additional metedata columns as needed.

For example, suppose we have data that are 8 features for each crop of the image at each frame in the image series. In the manifest file that we load into `dynamics_preproc.py`, the rows are each instance of the data (i.e., one single image crop at a given frame), the first 8 columns represent each of the extracted features, and the remaining columns are metadata for, e.g., what dataset the original image is from, the FOV the crop was taken from, the time point in the movie, etc.

## Environment variables and management

Install `pdm` and configure project at the root of the `cellsmap` repository.

## Running SDE model fitting and analysis

Set working directory to be the head of the `cellsmap` repository.

`pdm run cellsmap/analyses/workflows/dynamics_preproc.py [config_name]`
* Load manifest (Diffusion AE output: crop-based features for mutliple datasets), remove outliers, fit PCA to get shared low dimensional state space. 
* Get one time step displacements of crops over time, train/test split for fitting drift
 $$\mathbf{f}(\mathbf{x})$$ 
 and diffusion 
 $$D(\mathbf{x})$$ 
 coefficients from these displacements.

`pdm run cellsmap/analyses/workflows/dynamics_fit.py [config_name]`
* Load train/test sets from `manifest_postproc.py`, regression (SINDy - regression against set of basis functions) to fit callable drift and diffusion functions.

`pdm run cellsmap/analyses/workflows/dynamics_summarize.py [config_name]`
* Using fit SINDy objects (callable functions learned via regression), generate summary plots of various analyses of the SDE model 
$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}) + \sqrt{2 D(\mathbf{x})} \xi(t)$$

* Analyses include:
    * Phase portrait for shear stress values (control parameter of fit model) present in the data
    * Comparison of "stationary" histogram of the data and predicted stationary distribution for the SDE model
    * Fixed points and stability as a function of shear stress (quasi bifurcation diagram)
    * Entropy production rate as a function of shear stress
    * Generalized potential energy ("landscape") $U = -ln P$ for various values of shear stress

For each workflow, `[config_name]` is an optional command line input to specify the config in `dynamics_config.yaml` to use when running the workflow. If this is not specified via command line, the workflows are run using the `default` config in `dynamics_config.yaml`.

### Config file documentation
- `name` (type: `str`): Name of this set of config variables. This is what gets passed in as `[config_name]` via command line.
- `output_subdir` (type: `str`): Name of subdirectory to save workflow outputs (e.g., train/test set for vector field regression) and figures. If it does not already exist, the directories `cellsmap/analyses/results/[output_subdir]` and `cellsmap/figs/[output_subdir]` are made for saving the outputs and figures, respectively. 
- `PCs_to_analyze` (type `list[int]`): Which two of the principal component axes to project data onto and analyze trajectories from there.
- `datasets_to_skip` (type `list[str]`):
- `N_bins_kramers_moyal` (type `list`):
- `dt` (type `int` or `float`):
- `polynomial_lib` type `dict`:
    - `drift_feats` (type `int`):
    - `drift_param` (type `int`):
    - `diffusion_feats` (type `int`):
    - `diffusion_param` (type `int`):
- `plt_xlim` (type `list[int]`):
- `plt_ylim` (type `list[int]`):
- `N_pts_pplane` (type `list[int]`):
- `N_bins_hist` (type `list[int]`):
- `N_bins_landscape` (type `list[int]`):
- `shear_range` (type `list[int]`):
- `N_shear_fpt` (type `int`):
- `N_shear_landscape` (type `int`):
- `downsample_quiver` (type `int`):
- `norm_vectors` (type `bool`):

### Acessing and interpreting outputs
Set output directory via `dynamics_config.yaml`, subdirectory of `cellsmap/analyses/results`
* Contains fit PCA model (`sklearn.pipeline.Pipeline` object), train/test sets for displacement vectors, and fit drift and diffusion functions (`pysindy.SINDy` object)

Figures are saved to the `figs` directory at the head of the `cellsmap` repo, subdirectory set with same name as results output directory via `dynamics_config.yaml`
* summary figures for data, PCA, and SDE model

