#%%
from pathlib import Path
from cellsmap.util import dataset_io
from bioio import BioImage
from tqdm import tqdm
#%%
print('Available datasets:')
dataset_name_list = dataset_io.get_available_datasets()
print('\n')

#%% Test opening original data file
for dataset_name in tqdm(dataset_name_list, total=len(dataset_name_list), desc='Opening original files', unit='dataset'):
    config_data = dataset_io.get_dataset_info(dataset_name)
    original_path = Path(config_data['original_path'])
    try:
        img = BioImage(original_path)
    except Exception as e:
        print(f"Failed to open original for dataset {dataset_name}: {e}")
        
# %% Test opening zarr file
def get_position_zarr_path(zarr_path, position):
    name_fmsid = Path(zarr_path).name
    position_path = f'{zarr_path}/{name_fmsid}_P{position}.ome.zarr'
    return position_path

for dataset_name in tqdm(dataset_name_list, total=len(dataset_name_list), desc='Opening zarr files', unit='dataset'):
    config_data = dataset_io.get_dataset_info(dataset_name)
    zarr_path = Path(config_data['zarr_path'])

    if 'scene_list' in config_data and config_data['scene_list']:
        number_of_positions = len(config_data['scene_list'])
    else:
        number_of_positions = dataset_io.get_total_number_of_positions(dataset_name)

    try:
        # Check if zarr_path is 'none' and skip processing if true
        if str(zarr_path).lower() == 'none':
            print(f"Zarr path does not exist for {dataset_name}")
            continue  # Skip to the next dataset
        
        # Process positions if zarr_path is valid
        for position in range(number_of_positions):
            position_path = Path(get_position_zarr_path(zarr_path, position))
            img = BioImage(position_path)
    except Exception as e:
        print(f"Failed to open zarr for dataset {dataset_name}: {e}")
#%%
