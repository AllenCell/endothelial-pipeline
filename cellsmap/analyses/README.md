# Workflows for fitting stochastic dynamical model to time series data of endothelial cells

## Data formatting for analysis

To start, ensure that your time series data are saved in a `.csv` or `.parquet` file where the columns represent the extracted features (i.e., observed variables), the rows represent inividual data points. There may be additional metedata columns as needed.

For example, suppose we have data that are 8 features for each crop of the image at each frame in the image series. In the manifest file that we load into `manifest_postproc.py`, the rows are each instance of the data (i.e., one single image crop at a given frame), the first 8 columns represent each of the extracted features, and the remaining columns are metadata for, e.g., what dataset the original image is from, the FOV the crop was taken from, the time point in the movie, etc.

## Environment variables and management

Install `pdm` and configure project at the root of the `cellsmap` repository.

## Running SDE model fitting and analysis

Working directory: head of `cellsmap` repository


`pdm run cellsmap/analyses/workflows/manifest_postproc.py`
* Load manifest (Diffusion AE output: crop-based features for mutliple datasets), remove outliers, fit PCA to get shared low dimensional state space. 
* Get one time step displacements of crops over time, train/test split for fitting drift ($\mathbf{f}(\mathbf{x})$) and diffusion ($D(\mathbf{x})$) coefficients from these displacements.

`pdm run cellsmap/analyses/workflows/dynamics_fit.py`
* Load train/test sets from `manifest_postproc.py`, regression (SINDy - regression against set of basis functions) to fit callable drift and diffusion functions.

`pdm run cellsmap/analyses/workflows/dynamics_summarize.py`
* Using fit SINDy objects (callable functions learned via regression), generate summary plots of various analyses of the SDE model $$\frac{d\mathbf{x}}{dt} = \mathbf{f}(\mathbf{x}) + \sqrt{2 D(\mathbf{x})} \xi(t)$$
    * Phase portrait for shear stress values (control parameter of fit model) present in the data
    * Comparison of "stationary" histogram of the data and predicted stationary distribution for the SDE model
    * Fixed points and stability as a function of shear stress (quasi bifurcation diagram)
    * Entropy production rate as a function of shear stress
    * Generalized potential energy ("landscape") $U = -ln P$ for various values of shear stress


### Acessing and interpreting outputs
Set output directory via `cellsmap/analyses/configs/manifest_postproc.py`
* `outputs` subdirectory: fit PCA model (`sklearn.pipeline.Pipeline` object), train/test sets for displacement vectors, and fit drift and diffusion functions (`pysindy.SINDy` object)
* `figs` subdirectory: summary figures for data, PCA, and SDE model

