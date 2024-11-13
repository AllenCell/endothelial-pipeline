# FEATURED FUNCTIONS AND WHERE TO FIND THEM
This document contains functions that were requested by others on the team (and thus more likely to be used by people other than their author) as well as a brief description and the location of the function.

## Functions

### `get_classic_segmentation`
- DESCRIPTION - 
Takes an image with a membrane-labeled structure and returns an instance segmentation as an array with the same shape as 'image'.
- USE CASE - You've loaded the Cdh5 channel from one of the timepoints in one of the datasets and want to recreate the instance segmentations.
- LOCATION - `cellsmap/util/cdh5_preprocessing.py`


### `get_density_map_from_thresholds`
- DESCRIPTION - Given a dataset name and timepoint will return an image estimating the local cell density.
- USE CASE - You want to see the approximate cell density in different regions of the image at a given timepoint to see if there are any patterns or biases over time.

- LOCATION - `cellsmap/features/cdh5_seg_density_map.py`
