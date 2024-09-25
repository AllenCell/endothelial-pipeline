# cellsmap
cellular state mapping for endos



## Installation
```bash
pdm sync
```

## Datasets
A catalog of the current datasets we have is [here](https://alleninstitute-my.sharepoint.com/:x:/g/personal/chantelle_leveille_alleninstitute_org/Ea2enebMkAROgiQIGnNC5ggBqMt19hA2esbT0_TzZvUz7A?e=wlty6T)


## Analysis: features
**NOTE:** Looking at the movies I have noticed that these cells tend to have long cadherin-rich protrusions or retraction fibers, and that these can go over the surface of other cells. This will make some cells appear smaller than they actually are.
Also the nuclei in the brightfield images are very noticeable, which might make them suitable for cell tracking purposes. 
**/END NOTE.**

### cdh5_classic_seg.py
I used features/cdh5_classic_seg.py to create segmentations of the cdh5-labeled cells. 
Right now the output directory for the results is hard-coded path to a folder on Vast -- this should be changed if ever released.

The cells are a little over-segmented and there are some imperfections in regions where the initial hysteresis threshold that is used as a segmentation aid breaks (e.g. small cell close to the bottom border slightly to the left of the vertical center line). I added hierarchical merging of adjacent regions to resolve some of the over-segmentation, but some issues persist.

### cdh5_nodes_and_edges.py
Cell edge alignment measurements are created using features/cdh5_nodes_and_edges.py from the raw cdh5 channel. It does so by 1. thresholding the cdh5-containing GFP channel, 2. skeletonizing this threshold, 3. sorting the skeleton into node and edge pixels, 4. connecting neighboring nodes with straight lines, 5. measuring the lengths of these straight lines and the angle that they make relative to a horizontal line (resulting in an angle between 0-90 where 0 = parallel to the horizontal flow of fluid and 90 = perpendicular to it).
This script outputs overlays of the raw images with the nodes, edges, and (rasterized versions of) the lines. It also outputs plots and tables for each timepoint that is analyzed as well as a master table with the tables of all the timepoints concatenated together.