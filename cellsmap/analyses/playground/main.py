from cyto_dl.api import CytoDLModel
from cellsmap.util import get_dataset_info


if __name__ == '__main__':

    model = CytoDLModel()
    model.load_config_from_file("/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cellsmap/cellsmap//model_features/configs/vicreg/eval_config.yaml")
    movie_path = get_dataset_info("20240305_T01_001")['zarr_path']

    print(movie_path)
