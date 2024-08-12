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

I used features/initial_test_cadherin.py to create segmentations of the E-cadherin-labeled cells. 
Right now the output directory for the results is hard-coded path to a folder on Vast -- this should be changed if ever released. 

The cells are a little over-segmented and there are some imperfections in regions where the initial hysteresis threshold that is used as a segmentation aid breaks (e.g. small cell close to the bottom border slightly to the left of the vertical center line). 

I tried different methods for filtering or removing regions such that each cell would only have 1 seed point for a watershed, but that has not been very effective and would be a fragile approach. 
I also tried converting the initial segmentation attempt into a Region Adjacency Graph (with success) and I think that region merging regions according to some rules might produce a more robust and effective solution, but have not had time to investigate further.
