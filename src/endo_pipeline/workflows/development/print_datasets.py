from endo_pipeline.cli import Datasets


def main(datasets: Datasets):
    print(datasets)


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
