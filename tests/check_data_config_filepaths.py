from pathlib import Path
from cellsmap.util import dataset_io
from bioio import BioImage
from typing import Literal, List, Dict

def check_filepaths():
    filepaths_found = {}
    for dataset_name in dataset_name_list:
        config_data = dataset_io.get_dataset_info(dataset_name)
        original_path = Path(config_data['original_path'])
        zarr_path = Path(config_data['zarr_path'])
        print(f"{dataset_name}: original_path {original_path.exists()}, zarr_path {zarr_path.exists()}")
        print('Trying to get BioImage objects...')
        try:
            original_img = BioImage(original_path)
        except:
            original_img = False
        try:
            zarr_img = BioImage(zarr_path)
        except:
            zarr_img = False
        filepaths_found[dataset_name] = {'original': original_path.exists(),
                                        'zarr': zarr_path.exists(),
                                        'bioio_original': original_img,
                                        'bioio_zarr': zarr_img}
    return filepaths_found

def get_broken_paths(filepaths_found: Dict, path_kind_list: None|List[Literal['original', 'zarr', 'bioio_original', 'bioio_zarr']]=None):
    for dataset_name in dataset_name_list:
        path_kind_list = filepaths_found[dataset_name] if not path_kind_list else path_kind_list
        if not all(filepaths_found[dataset_name].values()):
            for path_kind in path_kind_list:
                if not filepaths_found[dataset_name][path_kind]:
                    print(f'{dataset_name} has an invalid path: {path_kind}')

print('Available datasets:')
dataset_name_list = dataset_io.get_available_datasets()
print('\n')

filepaths_found = check_filepaths()

print('\n')
get_broken_paths(filepaths_found, path_kind_list=['bioio_original'])

assert all([all(fp_exists.values()) for dataset_name, fp_exists in filepaths_found.items()]), "Invalid filepaths found."
print(f'\N{party popper} All filepaths are valid.')
