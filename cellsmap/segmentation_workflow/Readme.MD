## Segmentation and tracking workflow


The nuclei were segmented using cellpose


```
# Nuclei segmentation workflow - This applies two cellpose models to the raw images
nuclei_segmentation_workflow.py

# Tracking workflow
# Note: This requires Ultrack to be installed. However, I could not get the installation requirements compatible with this repo's toml file. Therefore, we can either install it as a standalone tool in a new environment or use goutham_ultrack on the slurm cluster

tracking_ultrack_endo_multiseg_cell.py

tracking_ultrack_endo_multiseg.py

```
