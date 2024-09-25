# Workflows for fitting stochastic dynamical model to time series data of endothelial cells

## Data formatting for analysis

To start, ensure that your time series data are saved in the following format: `.csv` file where the columns represent the extracted features (i.e., observed variables), the rows represent inividual data points, and there are two additional metadata columns to denote 1. the trajectory index and 2. time point corresponding to that row.

For example, suppose we have data that are 256 features for each crop of the image at each frame in the image series. In the `.csv` file that we pass into `fit_SDE_model.py`, the rows are each instance of the data (i.e., one single image crop at a given frame), the first 256 columns represent each of the extracted features, and there are two additional metadata columns that denote  ["crop_index","T"], # list of names of metadata columns in the data (first is trajectory index, second is frame #).

## Environment variables and management

If you are running this workflow on a server with GPUs, an optional step is to select the GPU device to use via running `export CUDA_VISIBLE_DEVICES=[device ID]` in the command line, where `[device ID]` is the approprate ID for the desired GPU device (see device usage via running `nvidia-smi`, and get UIDs via `nvidia-smi -L`). If the device is not specified, a device is selected automatically in `fit_SDE_model.py` based on usage.

Install `pdm` and configure project at the root of the `cellsmap` repository.

## Model inputs via `dynamics_config.yaml`

In the main folder of the `cellsmap` repo, enter configuration parameters for the Langevin Regression method for learning stochastic dynamics ([Callaham <i> et al.</i> 2021](https://royalsocietypublishing.org/doi/10.1098/rspa.2021.0092)) into the file `dynamics_config.yaml`. The input variables are as follows:
  - `name`: Name of configuration, tells `fit_SDE_model.py` which set of configuration variables to use.
  - `dt`: Time interval between data points (units depends on set value of `dt` and rate of data collection).
  - `PCA`: If `"yes"`, perform PCA on the data and analyze the trajectories along the top `ndim` principal components (see next variable). If `"no"`, analysis is done on original data.
  - `ndim`: Number of dimensions to keep in the analysis. If PCA is `"yes"`, data are projected onto the top `ndim` principal components. If PCA is `"no"`, then user must specify which of the features (i.e., components of the vector-valued data) to perform analysis on. <b>NOTE:</b> The Langevin Regression method is currently only implemented in one and two dimensions, so `ndim` must be $\leq 2$.
  - `feats_to_analyze`: If PCA is `"yes"`, this variable can be set to `None`. Else, the user must specifcy which of the features to analyze via a list of length `ndim` of the corresponding column indices.
  - `center_traj`: If `"yes"`, the initial conditions of all trajectories are centered at 0 before passing into the Langevin Regression algorithm (but after projecting onto principal components, if `PCA == "yes"`). If `"no"`, then the data are left as-is. (<b>NOTE:</b> All data are automatically z-scored before PCA and/or this step regardless).
  - `split_flow`: If `"yes"`, the trajectories are split into separate datasets corresponding to the distinct flow regimes (i.e., distinct experimental conditions) before fitting dynamical model. That is, stochastic dynamics are learned for each distinct flow rate conditions. If `"no"`, a model fit is attempted using the full trajectories across all flow conditions.
  - `split_frame`: If `split_flow` is `"yes"`, this variable is a list of tuples indicating the start and end frames of the distinct flow conditions in the data. For example, if the flow changes from high to low at frame 283, then `split_flow` is set to `[(0,283),(283,-1)]`. If `split_flow` is `"no"`, then split flow can be set to `None`.
  - `split_order`: A list of strings indicating the order of the flow conditions in the data. For example, if the data were collected under high flow followed by low flow, then `split_order` is set to `["high","low"]`. This variable is used for appropriate file naming when saving the outputs of the Langevin Regression model for each flow condition. As with `split_frame`, if `split_flow == "no"`, then this variable can be set to `None`.
  - `metadata_cols`: List of strings of the metadata column names in the original `.csv` data file that indicate the trajectory index and the time point corresponding to each data point. For example, if the name of the trajectory index column is `crop_index` and the name of the frame number column is `T`, you would set `metadata_cols` to `["crop_index","T"]`. This variable is used when loading the data into `numpy` array format.
  - `N_bins`: Number of bins creating for mesh over each dimension of the data (required for Langevin Regression). For example, if `ndim == 1` and you want to bin the state space in 64 bins, you would set `N_bins` to 64. If `ndim == 2` and you want 48 bins along the first dimension and 64 bins along the second, you would set `N_bins` to `[48,64]`.
  - `auto_bin`: If `"yes"`, the upper and lower bounds along each dimension used for binning the data are determined automatically based on the values present in the data. Else, it is `"no"`, and `bin_limits` must be specified (see below). 
  - `bin_limits`: If `auto_bin == "yes"`, then `bin_limits` may be set to `None`. Else, if `auto_bin == "no"`, `bin_limits` must be a tuple (`ndim == 1`) or list of tuples (`ndim == 2`) specifying the upper and lower bounds of the bins in each dimension. If `auto_bin == "no"` and `bin_limits` is not specified, `fit_SDE_model.py` will raise an error.
  - `poly_degree_drift`: Polynomial order of the function library used for fitting the drift function of the dynamics (SINDy expression obtained via Langevin Regression). For example, if `poly_degree_drift == 2` and `ndim == 2`, each component of the 2D vector drift function will be expressed as a linear combination of the terms $1, x_1, x_2, x_1^2, x_1 x_2,$ and $x_2^2$. 
  - `poly_degree_diffusion`: Same as `poly_degree_diffusion`, except for fitting the diffusion coefficients. If you want to fit a purely additive noise model (i.e., diffusion coefficient does not depend on the state variable $x$) set `poly_degree_diffusion` to 0.
  - `savedir`: Path to directory where you want model outputs to be saved.
  - `logging`: If `"yes"`, a `.txt` file is created in the specified save directory for logging print statements. If `"no"`, print statements will appear in-line.

## Running Langevin Regression model fitting script

From any directory in the `cellsmap` git repo, run `pdm run path/to/fit_SDE_model.py`, where `path/to/fit_SDE_model.py` depends on the current working directory. If working from the root of the repository, the path is `cellsmap/analyses/workflows/fit_SDE_model.py`. This script loads the feature data from the appropriate `.csv`, performs any transformations (e.g., PCA) specified in `dynamics_config.yaml`, and runs the Langevin Regression algorithm on the resulting trajectories. Upon running the script, the user will be prompted to enter the following:
1. The name of the set of configurations from `dynamics_config.yaml` to use (the `name` variable in the desired dictionary). For example, if you have a set of configurations with `name == "mae_cdh5"`, you would enter `mae_cdh5` when prompted.
2. The name of the dataset you want to analyze (available datasets from `data_config.yaml` are printed in-line to aid this selection).
3. The name of the set of features extracted from this dataset that you want to analyze (available features from `data_config.yaml` dictionary corresponding to the selected dataset are printed in-line as well).

Once these inputs have been specified, the code will run until prompting the user again, this time to specify the time lag to use for the Langevin Regression algorithm.
    - use the plots in `[savedir]/figs/select_lag_[flow]` -- where `[savedir]` is the save directory pulled from `dynamics_config.yaml` and `[flow]` is the flow condition (relevant if fitting flow conditions separately, else defaults to `[flow] == "all"`) -- to select the appropriate time step lag to feed into the Langevin Regression algorithm (done for each flow condition if fitting separately)

### Acessing and interpreting outputs

Code then runs uninterrupted until terminating. Saves coefficients for SINDy expressions for drift and diffusion at varying levels of sparsity, also saves plot showing cost function and "active" "function library terms at each of these levels of sparsity.
- look at cost function plot to determine optimal sparsity level (`[savedir]/figs/cost_function_plot_[flow]`)

## Selecting stochastic model and running model analysis script

- once `fit_SDE_model.py` has finished running, in the `cellsmap` git repo run `pdm run cellsmap/analyses/workflows/analyze_SDE_model.py`
    - user inputs via command line: name of config used to fit model (used to determine where to pull saved model outputs from), number of terms to retain in the SINDy regression expressions for the drift and diffusion (see above step)
    - outputs phase portrait plot (phase line if 1D, phase plane if 2D), comparison of histogram and model's stationary density, plot of generalized stationary potential with and without the gradient/flux vector decomposition (saved in `[savedir]/figs`)
