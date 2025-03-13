from cellsmap.util import dataset_io

def test_load_config():
    # check if the config file is loaded correctly
    config = dataset_io.load_config()
    assert config[0]['name'] == '20240305_T01_001'

def test_load_all_datasets():
    for ds in dataset_io.get_available_datasets():
        channels = dataset_io.get_available_channels(ds)
        data = dataset_io.load_dataset(ds, channels)
        assert data is not None, f"Dataset {ds} returned None"
        assert data.shape[0] > 0, f"Dataset {ds} has an unexpected shape: {data.shape}"
        assert all(dim > 0 for dim in data.shape), f"Dataset {ds} has invalid dimensions: {data.shape}"

def test_get_dataset_info():
    # check if the dataset info is returned correctly
    dataset_info = dataset_io.get_dataset_info('20240305_T01_001')
    assert dataset_info['zarr_path'] == '//allen/aics/assay-dev/computational/data/holistic/endos/feasibility/20240305_T01_001.ome.zarr'

def test_get_zarr_path():
    path = dataset_io.get_zarr_path('20240305_T01_001')
    assert path == '//allen/aics/assay-dev/computational/data/holistic/endos/feasibility/20240305_T01_001.ome.zarr'

def test_load_dataset():
    # check end point specification
    movie = dataset_io.load_dataset('20240305_T01_001', channels=["CDH5_Tubulin"], time_end = 2)
    assert movie.shape == (3, 1, 1712, 9592)
    # check start point specification
    movie = dataset_io.load_dataset('20240305_T01_001', channels=["CDH5_Tubulin"], time_start=1, time_end = 2)
    assert movie.shape == (2, 1, 1712, 9592)
    # check resolution specification
    movie = dataset_io.load_dataset('20240305_T01_001', channels=["CDH5_Tubulin"], time_start=1, time_end=2, level=1)
    assert movie.shape == (2, 1, 856, 4796)
    movie = dataset_io.load_dataset('20240305_T01_001', channels=["CDH5_Tubulin", "Nuc_Seg"], time_start=1, time_end=2, level=1)
    assert movie.shape == (2, 2, 856, 4796)

def test_get_available_models(capsys):
    # check if the available models are printed correctly
    dataset_io.get_available_models()
    captured = capsys.readouterr()
    assert captured.out == 'mae_cdh5\nvicreg_cdh5\nvicreg_no_rot_cdh5\njepa_cdh5\nmae_std_bf\n'

def test_get_model_info():
    # check if the model info is returned correctly
    model_info = dataset_io.get_model_info('mae_cdh5')
    assert model_info['name'] == 'mae_cdh5'
    assert model_info['eval_config_path'] == "//allen/aics/assay-dev/users/Benji/cellsmap/cellsmap/model_features/configs/mae_cdh5/eval_config.yaml"
