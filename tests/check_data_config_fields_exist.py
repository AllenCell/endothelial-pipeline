from pathlib import Path
from cellsmap.util import dataset_io

print('Available datasets:')
dataset_name_list = dataset_io.get_available_datasets()
print('\n')

config_data_fields = [
  'name',
  'original_path',
  'zarr_path',
  'barcode',
  'cell_lines',
  'live_or_fixed_sample',
  'microscope',
  'shear_stress_regime',
  'use_cases',
  'pixel_size_xy_in_um',
  'duration',
  'time_interval_in_minutes',
  'flow',
  'egfp_channel_index',
  'brightfield_channel_index',
  'n_positions',
  'notes',
  ]

print('\nChecking for missing fields in dataset config files...')
missing_fields_found = False
for dataset_name in dataset_name_list:
    config_data = dataset_io.get_dataset_info(dataset_name)
    missing_fields = [field_name for field_name in config_data_fields if field_name not in config_data]
    missing_fields += [field_name for field_name in config_data if config_data[field_name] == None]
    if missing_fields:
        print (f"Dataset {dataset_name} is missing or has an empty field for: {missing_fields}")
        missing_fields_found = True
    else:
        pass
if not missing_fields_found:
    print('\N{party popper} No missing fields found!') 
print('Done.')
