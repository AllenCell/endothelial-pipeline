# Workflows for fitting stochastic dynamical model to time series data of endothelial cells

## Data formatting for analysis

To start, ensure that your time series data are saved in a `.csv` or `.parquet` file where the columns represent the extracted features (i.e., observed variables), the rows represent inividual data points. There may be additional metadata columns as needed.

For example, suppose we have data that are 8 features for each crop of the image at each frame in the image series. In the manifest file that we load into `build_train_and_test.py`, the rows are each instance of the data (i.e., one single image crop at a given frame), the first 8 columns represent each of the extracted features, and the remaining columns are metadata for, e.g., what dataset the original image is from, the FOV the crop was taken from, the time point in the movie, etc.

## Environment variables and management

Install `uv` and configure project at the root of the `cellsmap` repository.

## Running SDE model fitting and analysis

Set working directory to be the head of the `cellsmap` repository.

`uv run src/endo_pipeline/workflows/2d_diffae_dynamics/build_train_and_test.py [config_name]`
* Load manifest (Diffusion AE output: crop-based features for mutliple datasets), remove outliers, fit PCA to get shared low dimensional state space.
* Get one time step displacements of crops over time, train/test split for fitting drift
 $\mathbf{f}(\mathbf{x})$
 and diffusion
 $\mathbf{D}(\mathbf{x})$
 coefficients from these displacements.

`uv run src/endo_pipeline/workflows/2d_diffae_dynamics/fit_sde_model.py [config_name]`
* Load train/test sets from `manifest_postproc.py`, regression (SINDy - regression against set of basis functions) to fit callable drift and diffusion functions.

`uv run src/endo_pipeline/workflows/2d_diffae_dynamics/summarize_sde_model.py [config_name]`
* Using fit SINDy objects (callable functions learned via regression), generate summary plots of various analyses of the SDE model
$$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}) + \sqrt{2 \mathbf{D}(\mathbf{x})} \xi(t)$$

* Analyses include:
    * Phase portrait for shear stress values (control parameter of fit model) present in the data
    * Comparison of "stationary" histogram of the data and predicted stationary distribution for the SDE model
    * Fixed points and stability as a function of shear stress (quasi bifurcation diagram)
    * Entropy production rate as a function of shear stress
    * Generalized potential energy ("landscape")
    $U = -\ln P$
    for various values of shear stress

For each workflow, `[config_name]` is an optional command line input to specify the config in `src/configs/dynamics_config.yaml` to use when running the workflow. If this is not specified via command line, the workflows are run using the `default` config in `dynamics_config.yaml`.

### Config file documentation
- `name` (type: `str`): Name of this set of config variables. This is what gets passed in as `[config_name]` via command line.
- `pcs_to_analyze` (type `list[int]`): Which two of the principal component axes to project data onto and analyze trajectories from there.
- `datasets_to_skip` (type `list[str]`): Datasets to skip when fitting drift and diffusion models.
    - *NOTE:* This will probably be depreciated in the future, as one of the datasets being skipped (`20241016_20X`) will be recollected and we will replace that dataset with the replicate in the config. The other datasets to skip are the no flow datasets (not being used to fit the SDE model, are being used to generate principal component axes), and that can be checked without needing to specify datasets here.
- `dt` (type `int` or `float`): Time interval between frames in minutes. I am hardcoding this here because it is the same for all timelapse datasets we are collecting to be passed into the Diffusion AE model.
- `kramers_moyal` (type `dict`): Dictionary of arguments to pass into kernel-based regression pipeline for estimating the Kramers-Moyal coefficients (up to 2nd order - drift and diffusion coefficients)
    - `num_bins` (type `list[int]`): Number of bins to use in each dimension when computing the Kramers-Moyal estimates. These kernel density estimates are what get passed into the SINDy regression script to learn the drift and diffusion as functions of the state variables (i.e., the coordinates as specified by `pcs_to_analyze`) and the shear stress level.
    - `kernel_params` (type `dict`): Bandwidth (`bandwidth`) and kernel function (`kernel`) to use for kernel density estimation.
- `polynomial_lib` type `dict`: Highest order polynomial term to include in SINDy basis library for regression against Kramers-Moyal estimates to get the drift and diffusion functions.
    - `drift_feats` (type `int`): Polynomial order in powers of the state variables (i.e., the "features") for estimate of drift.
    - `drift_param` (type `int`): Polynomial order in powers of shear stress (dynamical systems model parameter) for estimate of drift.
    - `diffusion_feats` (type `int`): Polynomial order in powers of the state variables (i.e., the "features") for estimate of diffusion.
        - Default is 0, i.e., additive noise / D is assumed constant over state space. If
        $>0$
        , then we are assuming multiplicative noise.
    - `diffusion_param` (type `int`): Polynomial order in powers of shear stress (dynamical systems model parameter) for estimate of diffusion.
- `plt_xlim` (type `dict`): Plotting limits along x-axis for model summary plots. This axis corresponds to the principal component axis specified in the first element of `PCs`.
    - `pplane` (type `list[float]`): Bounds along x-axis for grid of points used to generate `pplane` summary plots (deterministic dynamics, i.e., drift flow field).
    - `hist` (type `list[float]`): Bounds along x-axis for bins used to generate plots comparing histogram of data at last ~100 frames of a given flow condition to stationary histogram predicted by fit SDE model.
- `plt_ylim` (type `dict`): Same as `plt_xlim` but for y-axis. This axis corresponds to the principal component axis specified in the second element of `pcs_to_analyze`.
- `num_pts_pplane` (type `list[int]`): Number of grid points to use when generating `pplane` summary figures.
- `num_bins_hist` (type `list[int]`): Number of bins to use when generating histogram/stationary distribution comparison plots.
- `num_bins_landscape` (type `list[int]`): Number of bins to use when generating plots of the generalize potential energy landscape, defined as the negative log of the predicted stationary probability distribution. The `hist` entry of `plt_xlim` and `plt_ylim` is re-used for generating the bins for these plots.
- `shear_range` (type `list[float]`): Range (low, high) of shear stress values to consider when summarizing model predictions (e.g., fixed points and stability) for values of shear stress beyond what is present in the data.
- `num_shear_fpt` (type `int`): Number of values of shear stress within `shear_range` to consider when plotting fixed points and their stability as a function of shear stress (quasi-bifurcation diagram).
- `num_shear_landscape` (type `int`): Number of values of shear stress within `shear_range` to consider when plotting generalized potential landscape as a function of shear stress.
- `downsample_quiver` (type `int`): Downsampling of vector field on grid for plotting gradient-flux decomposition of drift term on top of the landscape plot.
- `norm_vectors` (type `bool`): Whether or not to normalize the gradient and flux vector fields when plotting (sometimes looks better when vectors are unit vectors).

### Acessing and interpreting outputs
Intermediate workflow outputs (e.g., train/test sets) are saved to `results/stochastic_dynamics/[config_name]/outputs`. Figures are saved to `results/stochastic_dynamics/[config_name]/figs`. If they do not already exist, these directories are created automatically.

Intermediate outputs:
* train/test sets for displacement vectors
* fit drift and diffusion functions (`pysindy.SINDy` object)

Figures:
* summary figures for data, PCA, and SDE model
