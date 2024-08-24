import numpy as np
from cyto_dl.api import CytoDLModel
from cyto_dl.utils import extract_array_predictions
from cellsmap.util import io


if __name__ == '__main__':

    df = io.load_precomputed_features(dataset_name="20240305_T01_001", model_name="mae_cdh5")
    feature_crop_0_T_0 = df.query('crop_index == 0 and T == 0').iloc[0]

    model = CytoDLModel()
    model.load_config_from_file("//allen/aics/assay-dev/users/Benji/cellsmap/cellsmap/model_features/configs/mae_cdh5/array_eval_config.yaml")
    img = io.load_dataset("20240305_T01_001", channels=["CDH5_Tubulin", "Nuc_Seg"])

    data = [img[0, :1, :512, :512]]

    _, _, output = model.predict(data=data)

    # Need to compare "output" with "feature_crop_0_T_0" to make sure they match

    print(output)
