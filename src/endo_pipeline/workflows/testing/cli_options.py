def main() -> None:
    """Test CLI options"""

    from endo_pipeline.cli import UPLOAD_TO_FMS
    from endo_pipeline.io.fms import FMS

    print(f"Upload to FMS: {UPLOAD_TO_FMS}")
    print(FMS)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
