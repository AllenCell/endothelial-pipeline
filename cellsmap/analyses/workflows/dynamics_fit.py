# %%
import pysindy as ps # import pysindy package for SINDy based regression

from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.analyses.utils import model_fitting
# import config parameters
from cellsmap.analyses.configs.manifest_postproc_config import savedir, PCs, dt
from cellsmap.analyses.configs.dynamics_fit_config import drift_deg, diff_deg, param_deg_drift, param_deg_diff, sigmoid_funcs, func_names, include_sigmoid

# %%
# load train test data from file
train_test_dict = dynamics_io.load_train_test(savedir+'outputs/train_test_data.npz')
# %%
# build SINDy libraries for fitting model of drift and diffusion terms
drift_lib = model_fitting.build_drift_lib(ndim=len(PCs),drift_deg=drift_deg,param_deg=param_deg_drift,
                                          include_sigmoid=include_sigmoid,sigmoid_funcs=sigmoid_funcs,func_names=func_names)

diff_lib = model_fitting.build_diff_lib(ndim=len(PCs),diff_deg=diff_deg,param_deg=param_deg_diff)
# %%


# fit model for drift term - SINDy based regression
driftModel = ps.SINDy(feature_library = drift_lib, optimizer = ps.SSR())
driftModel.fit(train_test_dict['X_train'], t= dt,x_dot=train_test_dict['Y_train'],u=train_test_dict['u_train'])

# score on test set
drift_R2 = driftModel.score(train_test_dict['X_test'],x_dot=train_test_dict['Y_test'],u=train_test_dict['u_test'])
driftModel.print()

print('Coefficient of determination (R^2) for model of drift term: %f' %drift_R2)

# fit model for diffusion term - SINDy based regression
diffModel = ps.SINDy(feature_library = diff_lib, optimizer = ps.SSR())
diffModel.fit(train_test_dict['X_train'], t= dt,x_dot=train_test_dict['V_train'],u=train_test_dict['u_train'])

# score on test set
diff_R2 = diffModel.score(train_test_dict['X_test'],x_dot=train_test_dict['V_test'],u=train_test_dict['u_test'])
diffModel.print()

print('Coefficient of determination (R^2) for model of diffusion term: %f' %diff_R2)
# %%
# save models
model_dict = {'driftModel':driftModel,'diffModel':diffModel}
dynamics_io.save_model(model_dict, savedir+'outputs/')
# %%
# %%
