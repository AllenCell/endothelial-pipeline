import numpy as np
from cyto_dl.api import CytoDLModel
from cyto_dl.utils import extract_array_predictions
from cellsmap.util import dataset_io
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df = dataset_io.load_precomputed_features(dataset_name="20240305_T01_001", model_name="mae_cdh5")
    feature_crop_0_T_0 = df.query('crop_index == 0 and T == 0').drop(columns=["crop_index","T"]).iloc[0]

    model = CytoDLModel()
    model.load_config_from_file("//allen/aics/assay-dev/users/Benji/cellsmap/cellsmap/model_features/configs/mae_cdh5/array_eval_config.yaml")
    reader = dataset_io.load_dataset("20240305_T01_001", channels=["CDH5_Tubulin", "Nuc_Seg"], time_start=0, time_end=1)
    img = reader.compute()

    data = [img[0, slice(0,1), :512, :512]]

    _, _, output = model.predict(data=data)
    feature_crop_0_T_0_new = output[0][0][1:].mean(axis=(0,1))

    fig, ax = plt.subplots(1,1)
    ax.scatter(feature_crop_0_T_0, feature_crop_0_T_0_new)
    ax.set_xlabel("Pre computed features for patch 0 at time T=0")
    ax.set_ylabel("Array predicted features for patch 0 at time T=0")
    plt.savefig("test.png")

