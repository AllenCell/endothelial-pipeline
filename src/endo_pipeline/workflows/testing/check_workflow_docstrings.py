def main() -> None:
    """Check all workflow docstrings and tags for consistency."""

    import ast
    import re
    from collections import Counter
    from pathlib import Path

    valid_tags = {
        "cdh5-segmentation",
        "cdh5-tracking",
        "cell-centered",
        "cellpose",
        "correlation-analysis",
        "datasets",
        "diffae",
        "dynamical-systems",
        "dynamics",
        "first-passage-time",
        "fixed-points",
        "fms",
        "gpu",
        "grid-based",
        "internal",
        "manifests",
        "migration-coherence",
        "model-comparison",
        "model-evaluation",
        "model-training",
        "model-performance",
        "nuclei-prediction",
        "optical-flow",
        "pca",
        "preprocessing",
        "quality-control",
        "test-ready",
        "tfe",
        "validation",
        "vast",
        "visualization",
        "workers",
        "zarr-conversion",
    }

    all_tags = []

    for workflow_file in Path(__file__).resolve().parents[1].rglob("*/*py"):
        group, workflow = workflow_file.with_suffix("").parts[-2:]
        contents = workflow_file.read_text()

        if group == "figures":
            continue

        # Check if the workflow contains example usage
        if "## Example usage" not in contents and group != "figures":
            print(f"Workflow '{workflow}' does not contain example usage in docstring")

        # Check that workflow includes the #test-ready tag if it has DEMO_MODE
        # (only for production and development groups). Exclude the DiffAE model
        # training and evaluation workflows, which need to run in a specific
        # order and therefore does not work with how run all testable workflows
        # is currently set up
        if (
            "DEMO_MODE" in contents
            and "#test-ready" not in contents
            and group in ("production", "development")
            and workflow
            not in (
                "create_diffae_train_dataframe",
                "build_diffae_train_config",
                "train_diffae",
                "create_diffae_eval_dataframe",
                "build_diffae_eval_config",
                "eval_diffae",
            )
        ):
            print(f"Workflow '{workflow}' has DEMO MODE but is not tagged #test-ready")

        # Check that testing and internal workflows do not have #test-ready tag.
        # Exclude the run all testable workflows and this workflow
        if (
            "#test-ready" in contents
            and group in ("testing", "internal")
            and workflow not in ("run_all_testable_workflows", "check_workflow_docstrings")
        ):
            print(f"Workflow '{workflow}' should not be tagged #test-ready")

        # Check that the workflow includes the #gpu tag if it has NUM_GPUS.
        # Exclude the DiffAE config workflows, which use NUM_GPUs to set the
        # output configs rather than actually using the GPU
        if (
            "NUM_GPUS" in contents
            and "#gpu" not in contents
            and workflow not in ("build_diffae_train_config", "build_diffae_eval_config")
        ):
            print(f"Workflow '{workflow}' uses NUM_GPUS but is not tagged #gpu")

        # Check that the workflow does not include the #gpu tag if it does
        # not use NUM_GPUS. Exclude the Cellpose model training and eval, which
        # are tagged #gpu but do not directly use NUM_GPUS, and the "run all"
        # workflows, which reference the tag but do not use the GPU directly
        if (
            "#gpu" in contents
            and "NUM_GPUS" not in contents
            and workflow not in ("run_labelfree_nuclei_prediction", "retrain_cellpose")
            and "run_all" not in workflow
        ):
            print(f"Workflow '{workflow}' is tagged #gpu but does not use NUM_GPUS")

        # Check that the workflow includes the #workers tag if it has NUM_WORKERS
        if (
            "NUM_WORKERS" in contents
            and "#workers" not in contents
            and workflow not in ("build_diffae_train_config", "build_diffae_eval_config")
        ):
            print(f"Workflow '{workflow}' uses NUM_WORKERS but is not tagged #workers")

        # Check that the workflow does not include the #workers tag if it does
        # not use NUM_WORKERS
        if "#workers" in contents and "NUM_WORKERS" not in contents:
            print(f"Workflow '{workflow}' is tagged #workers but does not use NUM_WORKERS")

        # Extract docstring from file contents
        ast_contents = ast.parse(contents)
        if workflow.endswith("_nb"):
            docstring = ast.get_docstring(ast_contents) or ""
        else:
            docstring = ast.get_docstring(
                [
                    node
                    for node in ast.walk(ast_contents)
                    if isinstance(node, ast.FunctionDef) and node.name == "main"
                ][0]
            )

        # Get all tags and check that they are all valid
        found_tags = re.findall(r"#([a-z0-9\-]+)", docstring or "")
        all_tags.extend(found_tags)
        invalid_tags = set(found_tags) - valid_tags
        if invalid_tags:
            print(f"Workflow '{workflow}' is invalid tags: {invalid_tags}")

    tag_counts = Counter(all_tags)
    print(tag_counts)
