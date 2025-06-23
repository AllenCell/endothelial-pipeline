import pysindy as ps


def build_drift_lib(
    ndim: int, drift_deg: int = 3, param_deg: int = 3
) -> ps.ParameterizedLibrary:
    """
    Build library of polynomial basis functions for
    SINDy-based regression on Kramers-Moyal averages
    to get drift term of SDE model as a function of
    the state variables and control parameter (shear stress).

    Inputs:
    - ndim: int, number of dimensions of the SDE model
        (passed into the ParameterizedLibrary object as num_features)
    - drift_deg: int, degree of polynomial basis functions
        for the drift term of the SDE model
    - param_deg: int, degree of polynomial basis functions
        for the dependence of the drift term
        on the control parameter (shear stress)

    Outputs:
    - drift_lib: ps.ParameterizedLibrary object,
        library of polynomial basis functions
        for the drift term of the SDE model
    """
    # build set of basis functions for regression model for drift term of SDE model
    drift_feature_lib = ps.PolynomialLibrary(degree=drift_deg, include_bias=True)

    # library for model dependence on control parameters (shear stress)
    drift_parameter_lib = ps.PolynomialLibrary(
        degree=param_deg, include_bias=True
    )  # library for model dependence on control parameters (shear stress)

    # build full library for drift term: pySINDy parameterized library
    drift_lib = ps.ParameterizedLibrary(
        feature_library=drift_feature_lib,
        parameter_library=drift_parameter_lib,
        num_features=ndim,
        num_parameters=1,
    )

    return drift_lib


def build_diff_lib(
    ndim: int, diff_deg: int = 0, param_deg: int = 3
) -> ps.ParameterizedLibrary:
    """
    Build library of polynomial basis functions for
    SINDy-based regression on Kramers-Moyal averages
    to get diffusion term of SDE model as a function of
    the state variables and control parameter (shear stress).

    Inputs:
    - ndim: int, number of dimensions of the SDE model
        (passed into the ParameterizedLibrary object as num_features)
    - diff_deg: int, degree of polynomial basis functions
        for the diffusion term of the SDE model
    - param_deg: int, degree of polynomial basis functions
        for the dependence of the diffusion term
        on the control parameter (shear stress)

    Outputs:
    - diff_lib: ps.ParameterizedLibrary object,
        library of polynomial basis functions
        for the diffusion term of the SDE model
    """

    # build set of basis functions for regression model for diffusion term of SDE model
    diff_feature_lib = ps.PolynomialLibrary(degree=diff_deg, include_bias=True)

    # library for model dependence on control parameters (shear stress)
    diff_parameter_lib = ps.PolynomialLibrary(degree=param_deg, include_bias=False)

    # build full library for diffusion term: pySINDy parameterized library
    diff_lib = ps.ParameterizedLibrary(
        feature_library=diff_feature_lib,
        parameter_library=diff_parameter_lib,
        num_features=ndim,
        num_parameters=1,
    )

    return diff_lib
