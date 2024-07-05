from cellsmap.util import io

def test_load_config():
    # check if the config file is loaded correctly
    config = io.load_config()
    assert config[0]['name'] == '20240305_T01_001'

def test_get_available_datasets(capsys):
    # check if the available datasets are printed correctly
    io.get_available_datasets()
    captured = capsys.readouterr()
    assert captured.out == '20240305_T01_001\n'

def test_get_dataset_info():
    # check if the dataset info is returned correctly
    dataset_info = io.get_dataset_info('20240305_T01_001')
    assert dataset_info['zarr_path'] == '/allen/aics/assay-dev/computational/data/holistic/endos/feasibility/cdh5.ome.zarr'

def test_load_dataset():
    # check end point specification
    movie = io.load_dataset('20240305_T01_001', time_end = 2)
    assert movie.shape[0] == 3
    # check start point specification
    movie = io.load_dataset('20240305_T01_001', time_start=1, time_end = 2)
    assert movie.shape[0] == 2
    # check resolution specification
    movie = io.load_dataset('20240305_T01_001', time_start=1, time_end = 2, resolution=1)
    assert movie.shape[1:] == (856, 4796)