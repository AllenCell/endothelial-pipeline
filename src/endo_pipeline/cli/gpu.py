def setup_gpu(num_gpus: int) -> int | None:
    """
    Set up the GPU environment for workflow.

    Picks the GPUs with most free memory or sets CUDA_VISIBLE_DEVICES for
    multi-GPU & MIG, using the number of GPUs specified by the user.

    Parameters
    ----------
    num_gpus
        Number of GPUs to use with the workflow.
    """

    import logging
    import os
    import re
    import subprocess

    logger = logging.getLogger("")

    try:
        subprocess.check_output("nvidia-smi")
        logger.info("Setting up environment to run workflow using %d GPU(s).", num_gpus)
    except Exception as exception:
        logger.error("GPUs requested but could not be set up.")
        raise exception

    # Query list of devices to detect MIG
    device_query = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE).stdout.decode()
    is_mig = "MIG" in device_query
    mig_uuids = re.findall(r"UUID: (MIG-[a-f0-9-]+)", device_query)

    if is_mig and num_gpus > 1:
        logger.error("Cannot use DDP with MIG devices. Only one MIG device can be used.")
        raise RuntimeError("Cannot use DDP with MIG devices.")

    if is_mig and not mig_uuids:
        logger.error("MIG partitioning detected, but no UUIDs seen! No MIG UUIDs found.")
        raise RuntimeError("No MIG UUIDs found, but MIG is enabled.")

    if is_mig:
        selected_uuid = mig_uuids[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_uuid
        logger.info("Set CUDA_VISIBLE_DEVICES to [ %s ]", selected_uuid)
        return 1

    # Query for memory information as (memory_free, gpu_index)
    memory_query = ["nvidia-smi", "--query-gpu=memory.free,index", "--format=csv,noheader,nounits"]
    mem_info = subprocess.run(memory_query, stdout=subprocess.PIPE).stdout.decode().strip()
    gpu_available = re.findall(r"(\d+), (\d+)", mem_info)

    if not gpu_available:
        logger.error("Unable to get memory information for GPUS.")
        raise RuntimeError("Unable to automatically set up environment for GPU.")

    # Sort by free memory, descending, to get available device indices
    gpu_available_sorted = sorted(gpu_available, key=lambda x: int(x[0]), reverse=True)
    chosen_gpus = [gpu[1] for gpu in gpu_available_sorted[:num_gpus]]

    if num_gpus > len(gpu_available):
        logger.warning(
            "Requested %d devices, but only %d available. Using all available.",
            num_gpus,
            len(gpu_available),
        )

    chosen_gpus_joined = ",".join(chosen_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen_gpus_joined
    logger.info("Set CUDA_VISIBLE_DEVICES to [ %s ]", chosen_gpus_joined)

    return len(chosen_gpus)
