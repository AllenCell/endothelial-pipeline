def main() -> None:
    """
    Validate fixed points identified in 2D in (r, rho) and 1D in theta against
    the fixed points identified in the full 3D (r, rho, theta) space.

    #validation #fixed-points #dynamics
    """
    print("init!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
