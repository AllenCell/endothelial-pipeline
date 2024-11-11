# %%
import numpy as np
import pandas as pd
import sys, os
sys.path.append('/allen/aics/assay-dev/users/Erin/git-repos/nonparametric-changepoint-detection')
import changepoint_detection as cpd
from cellsmap.util import io
from cellsmap.analyses.workflows.fit_SDE_model import get_scaled_traj
import ruptures as rpt
import matplotlib.pyplot as plt
# %%
# Load data

config_name = "mae_cdh5_large_patch_0917"
dataset_name = "20240917"
feature_name = "mae_cdh5_large_patch"

# config_name = "mae_cdh5_small_patch_LR"
# dataset_name = "20240305_T01_001"
# feature_name = "mae_cdh5_small_patch"

# config_name = "seg_graph_LR"
# dataset_name = "20240305_T01_001"
# feature_name = "seg_graph"

data_config = io.get_dataset_info(dataset_name)
path_to_data = data_config['features'][feature_name]


metadata, PCA, ndim, dt, feats_to_analyze, center_traj, split_flow, split_frame, split_order, N, auto_bin, bin_limits, nf, ns, savedir, log_file = io.get_dynamics_inputs(config_name)

savedir = "/allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/playground/test_homogeneity/20240305/"

if not os.path.isdir(savedir):
    print("*** Creating directory to save results... \n")
    os.makedirs(savedir)
    os.makedirs(savedir+'data')
    os.makedirs(savedir+'outputs')
    os.makedirs(savedir+'figs')
    os.makedirs(savedir+'logs')

print("\n","*** Getting trajectories from data... \n",sep="")
X_t = get_scaled_traj(path_to_data,metadata,savedir,PCA,ndim,feats_to_analyze,log_file=log_file)

# %%
results = []
for i in range(X_t.shape[0]):
    result = rpt.Pelt().fit_predict(X_t[i,:,:],pen=10)
    if result is not None:
        results.append(result)
# %%
rpt.display(X_t[i,:,:],result)

# %%
plt.hist([bk for res in results for bk in res],bins=10)
# %%
mean_bk = np.mean([bk for res in results for bk in res])
# %%
rpt.display(np.mean(X_t[:,:,:],axis=0),[mean_bk,577]) 
# %%
class BadPartitions(Exception):
    """Exception raised when the partition is bad."""

    pass


def sanity_check(bkps1, bkps2):
    """Checks if two partitions are indeed partitions of the same signal.

    Args:
        bkps1 (list): list of the last index of each regime.
        bkps2 (list): list of the last index of each regime.

    Raises:
        BadPartitions: whenever a partition does not respect some conditions.

    Returns:
        None:
    """
    # checks if empty.
    for nom, bkps in zip(("first", "second"), (bkps1, bkps2)):
        if len(bkps) == 0:
            raise BadPartitions("The {} partition is empty.".format(nom))
    # checks if both ends with the same index.
    if max(bkps1) != max(bkps2):
        raise BadPartitions(
            "The end of the last regime is not the same for each of the "
            "partitions:\n{}\n{}".format(bkps1, bkps2)
        )
    # checks if there is repetition.
    for bkps in (bkps1, bkps2):
        seen = set()
        if any(i in seen or seen.add(i) for i in bkps):
            raise BadPartitions("Some indexes are repeated: {}".format(bkps))
        
def randindex(bkps1, bkps2):
    """Computes the Rand index (between 0 and 1) between two segmentations.

    The Rand index (RI) measures the similarity between two segmentations and
    is equal to the proportion of aggreement between two partitions.

    RI is between 0 (total disagreement) and 1 (total agreement).
    This function uses the efficient implementation of [1].

    [1] Prates, L. (2021). A more efficient algorithm to compute the Rand Index for
    change-point problems. ArXiv:2112.03738.

    Args:
        bkps1 (list): sorted list of the last index of each regime.
        bkps2 (list): sorted list of the last index of each regime.

    Returns:
        float: Rand index
    """
    sanity_check(bkps1, bkps2)
    n_samples = bkps1[-1]
    bkps1_with_0 = [0] + bkps1
    bkps2_with_0 = [0] + bkps2
    n_bkps1 = len(bkps1)
    n_bkps2 = len(bkps2)

    disagreement = 0
    beginj: int = 0  # avoids unnecessary computations
    for index_bkps1 in range(n_bkps1):
        start1: int = bkps1_with_0[index_bkps1]
        end1: int = bkps1_with_0[index_bkps1 + 1]
        for index_bkps2 in range(beginj, n_bkps2):
            start2: int = bkps2_with_0[index_bkps2]
            end2: int = bkps2_with_0[index_bkps2 + 1]
            nij = max(min(end1, end2) - max(start1, start2), 0)
            disagreement += nij * abs(end1 - end2)

            # we can skip the rest of the iteration, nij will be 0
            if end1 < end2:
                break
            else:
                beginj = index_bkps2 + 1

    disagreement /= n_samples * (n_samples - 1) / 2
    return 1.0 - disagreement

rands = []
for i in range(len(results)):
    for j in range(i+1,len(results)):
        rands.append(randindex(results[i],results[j]))

print(np.mean(rands))
# %%
X_stack = np.swapaxes(X_t,0,1).swapaxes(1,2).reshape(X_t.shape[1],-1)
# %%
result = rpt.Pelt().fit_predict(X_stack,pen=30)
rpt.display(np.mean(X_t[:,:,:],axis=0),result)

# %%
result
# %%
