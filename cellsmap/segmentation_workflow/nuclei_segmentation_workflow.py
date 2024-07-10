from cellpose import models
import os
import numpy as np
import argparse
from cellsmap.util import load_dataset
import skimage
import tifffile
from aicsimageio.writers import OmeTiffWriter

'''
Segmentation workflow for nuclei segmentation using cellpose for long stitched timelapse dataset

This performs segmentation using two different approaches: (1) using the cellpose model trained on mips of the brightfield and (2) using cellpose model trained on mips of the brightfield standard deviation. Output segmentations are saved as zarr files. This is then fed into the tracking workflow using ultrack
'''

parser= argparse.ArgumentParser()
parser.add_argument("--original_input_dir", type= str, default="/allen/aics/assay-dev/MicroscopyData/John Paul/2024/20240305/20240305_T01_001_3D_MTG_FULL.dir/20240305_T01_001 - Position 1 [6] 3DMontage Complete-1710177899-789.imgdir")
parser.add_argument("--output_dir", type= str, default="/allen/aics/assay-dev/computational/data/endothileal_cell_data/Timelapse_20x_dataset/reproducibility_test")
parser.add_argument("--model_bf_max_project", type=str, default="/allen/aics/assay-dev/computational/data/endothileal_cell_data/Timelapse_20x_dataset/montage_data_trainsets/models_weights/BF_MIP_patch_model/models/bf_mip_model_adapthist")
parser.add_argument("--model_bf_std_project", type=str, default="/allen/aics/assay-dev/computational/data/endothileal_cell_data/Timelapse_20x_dataset/montage_data_trainsets/models_weights/BF_STD_patch_model/models/bf_std_model_no_preprocess")



def get_filenames(original_input_dir):
    filenames_all = []
    for subdir, dirs, files in os.walk(original_input_dir):
        for file in files:
            if file.endswith(".npy") and file.startswith("ImageData") and "Ch2" in file:
                filenames_all.append(os.path.join(subdir, file))
    return filenames_all

def get_std_proj(img):
    '''returns max project of image'''
    max_proj = np.std(img, axis=0)[np.newaxis, ...][0,:,:].astype(np.uint16)
    return max_proj

def get_max_proj(img):
    '''returns max project of image'''
    max_proj = np.max(img, axis=0)[np.newaxis, ...][0,:,:]
    return max_proj



if __name__ == "__main__":
    # TODO: model dirs moved to a seperate config file would be a good idea
    #TODO: SAVE AS TWO CHANNEL TIFF?
    args = parser.parse_args()
    #dat = load_dataset('cdh5_path') # currently only has green channel 
    bf_maxproject_output_dir = os.path.join(args.output_dir, "bf_maxproject")
    bf_stdproject_output_dir = os.path.join(args.output_dir, "bf_stdproject")

    if not os.path.exists(bf_maxproject_output_dir):
        os.makedirs(bf_maxproject_output_dir)
    if not os.path.exists(bf_stdproject_output_dir):
        os.makedirs(bf_stdproject_output_dir)
    
    model_bf_maxproject = models.CellposeModel(gpu=True, pretrained_model=args.model_bf_max_project)
    model_bf_stdproject = models.CellposeModel(gpu=True, pretrained_model=args.model_bf_std_project)

    filepaths_brightfield = get_filenames(args.original_input_dir)
    filepaths_brightfield.sort()
    for file in filepaths_brightfield:
        img_std = get_std_proj(np.load(file))
        img_bf = get_max_proj(np.load(file))
        # get masks, flows, styles, diams
        img_bf = skimage.exposure.equalize_adapthist(img_bf, nbins=2**16, kernel_size=100)

        masks_std, _, _ = model_bf_stdproject.eval([img_std], channels=[0,0], min_size=50, flow_threshold=0.6, cellprob_threshold=-3.0)
        masks_bf, _, _ = model_bf_maxproject.eval([img_bf], channels=[0,0], min_size=50 , flow_threshold=0.6, cellprob_threshold=-3.0)

        OmeTiffWriter.save(masks_std[0], os.path.join(bf_stdproject_output_dir, f"SEG_{os.path.basename(file).split('.npy', 1)[0]}.tiff"), dim_order="YX", channel_names=["nuclei_segmentation"])
        OmeTiffWriter.save(masks_bf[0], os.path.join(bf_maxproject_output_dir, f"SEG_{os.path.basename(file).split('.npy', 1)[0]}.tiff"), dim_order="YX", channel_names=["nuclei_segmentation"])#

        
        


