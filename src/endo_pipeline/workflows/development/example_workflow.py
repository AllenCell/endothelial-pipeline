def main():
    from endo_pipeline import DEMO_MODE
    from endo_pipeline.io.fms import FMS

    print(FMS._metadata_management_client._http_client._baseUri)
    print(FMS._upload_service._file_repository._file_storage_client._http_client._baseUri)
    print(FMS._upload_service._file_repository._file_explorer_client._http_client._baseUri)
    print(FMS._file_repository._upload_tracker_repository._job_status_client._http_client._baseUri)

    if DEMO_MODE:
        print("DEMO MODE ACTIVATED")


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
