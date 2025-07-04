import gspread
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Path(__file__) gives the full path to this file.
# .resolve() makes it an absolute path.
# .parent gives the directory containing this file (i.e., 'src/').
# .parent.parent gives the parent of 'src/' (i.e., your_project_directory/).
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent

# Now define the path to your credentials directory relative to the project root
CREDS_DIR = PROJECT_ROOT_DIR / "config"

# Get paths from environment or use defaults
CREDENTIALS_FILE_PATH = Path(
    os.getenv("GOOGLE_CREDENTIALS_PATH", CREDS_DIR / "credentials.json")
)
TOKEN_FILE_PATH = Path(os.getenv("GOOGLE_TOKEN_PATH", CREDS_DIR / "token.json"))

# Configure logging (as before)
logger = logging.getLogger(__name__)


def get_authenticated_gspread_client() -> gspread.Client | None:
    logger.info(f"Attempting OAuth 2.0 authentication using: {CREDENTIALS_FILE_PATH}")

    if not CREDENTIALS_FILE_PATH.exists():
        logger.error(f"Credentials file not found: {CREDENTIALS_FILE_PATH}")
        logger.error(
            f"Please ensure your credentials file is available at the path specified by GOOGLE_CREDENTIALS_PATH environment variable or at the default location: {CREDS_DIR / 'credentials.json'}"
        )
        return None

    # Ensure the directory exists for token.json (get parent directory of token file)
    TOKEN_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        gc = gspread.oauth(
            credentials_filename=str(CREDENTIALS_FILE_PATH),
            authorized_user_filename=str(TOKEN_FILE_PATH),
        )
        logger.info("Successfully authenticated with Google Sheets.")
        return gc
    except Exception as e:
        logger.error(
            f"An error occurred during Google Sheets authentication: {e}", exc_info=True
        )
        return None
