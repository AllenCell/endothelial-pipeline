# %%
from cellsmap.analyses.utils import dynamics_io
from cellsmap.analyses.configs.manifest_postproc_config import savedir

# %%
model_dict = dynamics_io.load_model(savedir+'outputs/drift_diffusion_model.npz')

driftModel = model_dict['driftModel']
diffModel = model_dict['diffModel']
# %%
