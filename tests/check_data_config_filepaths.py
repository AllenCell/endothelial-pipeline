from pathlib import Path
from cellsmap.util import io

print('Available datasets:')
dataset_name_list = io.get_available_datasets()
print('\n')
filepaths_found = {}
for dataset_name in dataset_name_list:
    config_data = io.get_dataset_info(dataset_name)
    original_path = Path(config_data['original_path'])
    zarr_path = Path(config_data['original_path'])
    print(f"{dataset_name}: original_path {original_path.exists()}, zarr_path {zarr_path.exists()}")
    filepaths_found[dataset_name] = {'original': original_path.exists(), 'zarr': zarr_path.exists()}
print('\n')
[print(f'{dataset_name} has one or more invalid paths...') for dataset_name in dataset_name_list if not all(filepaths_found[dataset_name].values())]
assert all([all(fp_exists.values()) for dataset_name, fp_exists in filepaths_found.items()]), "Invalid filepaths found."
print(f'\N{party popper} All filepaths are valid.')
