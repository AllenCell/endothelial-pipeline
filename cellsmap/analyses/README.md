# Workflows for fitting stochastic dynamical model to time series data of endothelial cells

## Data formatting for analysis

To start, ensure that your time series data are saved in a `.csv` or `.parquet` file where the columns represent the extracted features (i.e., observed variables), the rows represent inividual data points. There may be additional metedata columns as needed.

For example, suppose we have data that are 8 features for each crop of the image at each frame in the image series. In the `.csv` file that we load in `fit_and_analyze_SDE.py`, the rows are each instance of the data (i.e., one single image crop at a given frame), the first 8 columns represent each of the extracted features, and te remaining columns are metadata for, e.g., what dataset the original image is from, the FOV the crop was taken from, the time point in the movie, etc.

## Environment variables and management

Install `pdm` and configure project at the root of the `cellsmap` repository.

## Running SDE Regression model fitting script


### Acessing and interpreting outputs


## Running model analysis script

