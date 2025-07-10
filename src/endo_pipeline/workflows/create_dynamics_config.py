# %% [markdown]
# # Create dynamics analysis pipeline config

# %% [markdown]
"""
Create and save a new config file for the 2D Diff AE feature dynamics pipeline from a `DynamicsConfig` object.

The config is saved to the `configs/dynamics_pipeline` directory with file name matching
the name of the dataset. If a config with the same name already exists, it will
be overwritten.

### Optional fields

Some fields in the config are optional, and will be set to a default value if
not provided. All optional fields are provided as commented lines of code. If an
optional field should be set, uncomment the corresponding line to set the value.
"""

# %%
if __name__ != "__main__":
    raise ImportError("This module is a notebook and is not meant to be imported")

# %%
from src.endo_pipeline.configs import (
    DynamicsConfig,
    KramersMoyalParameters,
    SINDyRegressionParameters,
    save_dynamics_config,
)

# %%
model = DynamicsConfig(
    # ============================ REQUIRED FIELDS =============================
    name="unique_configuration_name",
    # ============================ OPTIONAL FIELDS =============================
    pcs_to_analyze=[0, 2],  # Principal components to analyze
    dt=5,  # Time step for dynamics analysis (sets unit of time)
    kramers_moyal=KramersMoyalParameters(
        num_bins=[70, 70],  # Number of bins for Kramers-Moyal estimation
        bandwidth=0.075,  # Bandwidth for kernel regression
        kernel="gaussian",  # Kernel type for Kramers-Moyal estimation
    ),
    sindy_parameters=SINDyRegressionParameters(
        drift_feat_degree=4,  # Polynomial degree for feature variable expansion of drift coefficient
        diffusion_feat_degree=4,  # Polynomial degree for feature variable expansion of diffusion coefficient
        drift_param_degree=6,  # Polynomial degree for control parameter expansion of drift coefficient
        diffusion_param_degree=6,  # Polynomial degree for control parameter expansion of diffusion coefficient
    ),
    num_points_pplane=50,  # Number of points for grid in phase plane visualization
    num_bins_histogram=[50, 50],  # Number of bins for histogram visualization
    num_bins_landscape=[60, 60],  # Number of bins for phase plane visualization
    shear_stress_range=[
        4.0,
        30.0,
    ],  # Range for shear stress values in fixed point analysis and plotting landscape
    num_shear_fixed_points=10,  # Number of shear stress values to plot predicted fixed points for
    num_shear_landscape=10,  # Number of shear stress values to plot landscape for
    quiver_downsample_factor=10,  # Downsample factor for quiver plot
    norm_vectors=True,  # Normalize vectors in quiver plot
)

save_dynamics_config(model)

# %%
