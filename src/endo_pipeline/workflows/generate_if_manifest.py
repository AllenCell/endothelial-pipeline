# %%
from src.endo_pipeline.library.process.if_feature_extraction import run_nuclei_feature_extraction

# %%
dataset = "20250509_20X_IF1"
df = run_nuclei_feature_extraction(dataset)
# %%
