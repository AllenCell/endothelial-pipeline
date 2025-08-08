def main():
    from src.endo_pipeline import TESTING_MODE
    from src.endo_pipeline.io.fms import FMS

    print(FMS._metadata_management_client._http_client._baseUri)
    print(FMS._upload_service._file_repository._file_storage_client._http_client._baseUri)
    print(FMS._upload_service._file_repository._file_explorer_client._http_client._baseUri)
    print(FMS._file_repository._upload_tracker_repository._job_status_client._http_client._baseUri)

    if TESTING_MODE:
        print("TESTING MODE ACTIVATED")


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
