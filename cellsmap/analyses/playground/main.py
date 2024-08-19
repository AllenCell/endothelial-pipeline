import numpy as np
from cyto_dl.api import CytoDLModel
from cyto_dl.utils import extract_array_predictions
from cellsmap.util import get_dataset_info


if __name__ == '__main__':

    model = CytoDLModel()
    model.load_config_from_file("/allen/aics/assay-dev/MicroscopyOtherData/Viana/projects/cellsmap/cellsmap//model_features/configs/vicreg/eval_config.yaml")
    movie_path = get_dataset_info("20240305_T01_001")['zarr_path']

    # model.load_default_experiment(
    #     "segmentation_array", output_dir="./output", overrides=["data=im2im/numpy_dataloader_predict"]
    # )

    data = [np.random.rand(1, 1, 512, 512), np.random.rand(1, 1, 512, 512)]

    _, _, output = model.predict(data=data)
    preds = extract_array_predictions(output)

    print(output)
