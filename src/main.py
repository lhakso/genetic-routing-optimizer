import gspread
import os
import logging
import auth_utils
from clean_sheet import SpreadsheetCleaner
from dotenv import load_dotenv

load_dotenv()

SHEET_URL = os.getenv('GOOGLE_SHEETS_URL', 'https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit')
CIRCUIT_NAME = os.getenv('CIRCUIT_NAME', '2663')

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_data_processing_workflow():
    """
    Main function to orchestrate the data processing workflow:
    1. Authenticates with Google Sheets.
    2. Calls the data cleaning utility.
    3. Prints results or error messages.
    """
    logger.info("Starting the data processing workflow...")

    # 1. Authenticate using the auth_utils module
    # This will attempt to get an authenticated gspread client.
    g_client = auth_utils.get_authenticated_gspread_client()

    if g_client:
        logger.info(
            f"Successfully authenticated. "
            f"Preparing to process worksheet: '{CIRCUIT_NAME}' "
            f"from URL: {SHEET_URL}"
        )

        # 2. Process the spreadsheet using the sheet_utils module
        # It expects an authenticated gspread client and sheet details.
        cleaner = SpreadsheetCleaner(
            client=g_client,
            circuit_name=CIRCUIT_NAME,
            sheet_url=SHEET_URL,
        )
        data_df = cleaner.clean()

        # 3. Handle and display the results
        if data_df is not None:
            logger.info("Data processing completed successfully.")

            print("\n--- Processed 'data_df' (selected columns, incomplete items) ---")
            if not data_df.empty:
                print(data_df.head())
            else:
                print(
                    "Resulting 'data_df' is empty (no incomplete items or other filtering)."
                )
            print(f"Shape of 'data_df': {data_df.shape}")

            print("\n--- Processed 'orig_df' (different columns, incomplete items) ---")

        else:
            # this indicates an error occurred during clean_spreadsheet_data execution.
            logger.error("Data processing failed. Check logs for details.")
    else:
        # This case means get_authenticated_gspread_client returned None.
        logger.error(
            "Authentication failed. Cannot proceed with data processing. Exiting."
        )

    logger.info("Data processing workflow finished.")


if __name__ == "__main__":
    try:
        run_data_processing_workflow()
    except Exception as e:
        logger.critical(
            f"An unexpected critical error occurred in the main application: {e}",
            exc_info=True,
        )
