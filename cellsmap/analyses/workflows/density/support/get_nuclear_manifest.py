#%%
import os
import pandas as pd
from bioio import BioImage
from cellsmap.util import dataset_io, set_ouput
from skimage.measure import label, regionprops
from tqdm import tqdm


# %%
def create_nuclear_manifest(dataset):
    positions = []    
    frames = []
    nuclear_labels = []
    x_coords = []
    y_coords = []
    areas = []
    
    n_positions = dataset_io.get_total_number_of_positions(dataset)
    for position in range(n_positions):
        dataset_position_path = dataset_io.get_nuclear_prediction_path(dataset, position)
        imgs = os.listdir(dataset_position_path)

        for frame, img_name in tqdm(enumerate(imgs), total=len(imgs), desc=f"Processing frames in {dataset} P{position}"):
            fov_path = os.path.join(dataset_position_path, img_name)
            nuc_seg_image = BioImage(fov_path)
            image_data = nuc_seg_image.get_image_data("XY", C=2)
            labeled_image = label(image_data)
            props = regionprops(labeled_image)
            
            for prop in props:
                positions.append(position)
                frames.append(frame)
                nuclear_labels.append(prop.label)
                x_coords.append(round(prop.centroid[1], 2))
                y_coords.append(round(prop.centroid[0], 2))
                areas.append(prop.area)

        # Create a DataFrame from the lists
        df = pd.DataFrame({
            'dataset': dataset,
            'position': positions,
            'frame': frames,
            'fov_path': fov_path,
            'nuclear_label': nuclear_labels,
            'x': x_coords,
            'y': y_coords,
            'area': areas
        })
    
    out_put_path = set_ouput.get_output_path('nuclear_seg_manifests')
    df.to_parquet(f'{out_put_path}{dataset}_nuclear_manifest.parquet')
        
    return df

# %%
if __name__ == '__main__':
    
    dataset_list = ['20241016_20X']
    
    for dataset in dataset_list:
        create_nuclear_manifest(dataset)
# %%
