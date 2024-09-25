import numpy as np
from cyto_dl.api import CytoDLModel
from cyto_dl.utils import extract_array_predictions
from cellsmap.util import io

model = CytoDLModel()
model.load_config_from_file("//allen/aics/assay-dev/users/Benji/cellsmap/cellsmap/model_features/configs/vicreg_no_rot_cdh5/array_eval_config.yaml")

# NOTE: data must be CYX and shape (1, 512, 512)
data = [np.random.rand(1, 512, 512), np.random.rand(1, 512, 512)]

# output is a list with the form [(features_image1, metadata_image1), (features_image2, metadata_image2), ...]
_, _, output = model.predict(data=data)

print(output)
# do something with your output
