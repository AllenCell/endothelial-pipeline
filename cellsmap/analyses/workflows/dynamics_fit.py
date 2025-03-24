# %%
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
driftModel = model_fitting.fit_sindy_model(drift_lib,train_test_dict['X_train'],train_test_dict['Y_train'],
                                           train_test_dict['u_train'],t=dt,optimizer=ps.SSR())

# score on test set
drift_R2 = driftModel.score(train_test_dict['X_test'],x_dot=train_test_dict['Y_test'],u=train_test_dict['u_test'])
driftModel.print()

print('Score (R^2) for model of drift term: %f' %drift_R2)
# %%
# fit model for diffusion term - SINDy based regression
diffModel = model_fitting.fit_sindy_model(diff_lib,train_test_dict['X_train'],train_test_dict['V_train'],
                                          train_test_dict['u_train'],t=dt,optimizer=ps.SSR())

# score on test set
diff_R2 = diffModel.score(train_test_dict['X_test'],x_dot=train_test_dict['V_test'],u=train_test_dict['u_test'])
diffModel.print()

print('Score (R^2) for model of diffusion term: %f' %diff_R2)
# %%
# save models
model_dict = {'driftModel':driftModel,'diffModel':diffModel}
dynamics_io.save_model(model_dict, savedir+'outputs/')
# %%




# %%
import numpy as np
import pysindy as ps

# full library for drift term (functions of state variables)
drift_feature_lib = ps.PolynomialLibrary(degree=3, include_bias=True)


ndim = len(PCs)

# library for model dependence on control parameters (shear stress)
drift_parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=True) # library for model dependence on control parameters (shear stress)

# build full library for drift term: pySINDy parameterized library
drift_lib_=ps.ParameterizedLibrary(feature_library=drift_feature_lib,
    parameter_library=drift_parameter_lib,num_features=ndim,num_parameters=1) 

# build library for diffusion term (polynomial library only)
diff_feature_lib=ps.PolynomialLibrary(degree=0, include_bias=True)
diff_parameter_lib=ps.PolynomialLibrary(degree=3, include_bias=False)
diff_lib_=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
    parameter_library=diff_parameter_lib,num_features=ndim,num_parameters=1)


# fit model for drift term - SINDy based regression
driftModel_ = ps.SINDy(feature_library = drift_lib_, optimizer = ps.SSR())
driftModel_.fit(train_test_dict['X_train'], t= dt,x_dot=train_test_dict['Y_train'],u=train_test_dict['u_train'])

# score on test set
drift_R2 = driftModel_.score(train_test_dict['X_test'],x_dot=train_test_dict['Y_test'],u=train_test_dict['u_test'])
driftModel_.print()

print('Coefficient of determination (R^2) for model of drift term: %f' %drift_R2)

# fit model for diffusion term - SINDy based regression
diffModel_ = ps.SINDy(feature_library = diff_lib_, optimizer = ps.SSR())
diffModel_.fit(train_test_dict['X_train'], t= dt,x_dot=train_test_dict['V_train'],u=train_test_dict['u_train'])

# score on test set
diff_R2 = diffModel_.score(train_test_dict['X_test'],x_dot=train_test_dict['V_test'],u=train_test_dict['u_test'])
diffModel_.print()

print('Coefficient of determination (R^2) for model of diffusion term: %f' %diff_R2)
# %%
model_dict = {'driftModel':driftModel_,'diffModel':diffModel_}
dynamics_io.save_model(model_dict, savedir)
# %%
