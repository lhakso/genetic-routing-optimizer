import gspread
import pandas as pd
import logging

pd.set_option(
    "future.no_silent_downcasting", True
)  # to test the new downcasting behavior

DEFAULT_HEADER_ROW = 12
DEFAULT_DATA_START_ROW = 14
MAX_COLUMN_LETTER = "U"

logger = logging.getLogger(__name__)

COLNAME_REPLACEMENTS = {" ": "_", "/": "", "#": "no", "__": "_"}

# Define standard value replacements (Resolve ambiguity of 'X' and '')
# Suggestion: Use more descriptive keys if possible
VALUE_REPLACEMENTS = {
    # For 'is_complete' column primarily
    "Not Started": False,
    "Done": True,
    "In Process": False,
    "Hold/ Change in Contract": True,
    # For 'projected_hours'
    "X": 1.25,  # TODO: Confirm meaning of 'X'
    "": 1.25,  # TODO: Confirm meaning of blank ""
}

# Define columns needed for the final outputs
DEFAULT_DATA_COLS = [
    "site_no",
    "address",
    "work_description",
    "owner_phone_comments",
    "no_parks",
    "nbw",
    "projected_hours",
    "flagging",
    "requires_squirt_boom",
    "merge",
    "notes",
    "also_clear_for",
    "is_complete",
]


class SpreadsheetCleaner:
    def __init__(
        self,
        client: gspread.Client,
        sheet_url: str,
        circuit_name: str,
        header_row: int = DEFAULT_HEADER_ROW,
        data_start_row: int = DEFAULT_DATA_START_ROW,
        max_column_letter: str = MAX_COLUMN_LETTER,
        colname_replacements: dict = None,
        value_replacements: dict = None,
        data_cols: list = None,
    ):
        self.client = client
        self.sheet_url = sheet_url
        self.circuit_name = circuit_name
        self.header_row = header_row
        self.max_column_letter = max_column_letter
        self.data_start_row = data_start_row
        self.colname_replacements = (
            colname_replacements
            if colname_replacements is not None
            else COLNAME_REPLACEMENTS.copy()
        )
        self.value_replacements = self.value_replacements = (
            value_replacements
            if value_replacements is not None
            else VALUE_REPLACEMENTS.copy()
        )
        self.data_cols = (
            data_cols if data_cols is not None else DEFAULT_DATA_COLS.copy()
        )
        self.df: pd.DataFrame | None = None
        self.raw_column_names: list | None = None

        logger.info(
            f"SpreadsheetCleaner initialized for worksheet: '{self.circuit_name}'."
        )

    def _fetch_raw_df(self):
        logger.info(
            f"clean_spreadsheet_data called for circuit: '{self.circuit_name}' to return a single DataFrame."
        )
        if self.header_row < 1 or self.data_start_row <= self.header_row:
            logger.error("Invalid header_row_num or data_start_row_num.")
            raise ValueError(
                "Header row must be less than data start row, and both positive."
            )

        try:
            spreadsheet = self.client.open_by_url(self.sheet_url)
            worksheet = spreadsheet.worksheet(self.circuit_name)
            logger.info(f"Successfully opened worksheet: '{self.circuit_name}'")

            all_sheet_values = worksheet.get(
                f"A{self.header_row}:{self.max_column_letter}"
            )
            if not all_sheet_values:
                logger.warning(
                    f"No data found in '{self.circuit_name}' from row {self.header_row}."
                )
                self.df = pd.DataFrame(columns=self.data_cols)
                return True

            raw_column_names = all_sheet_values[0]
            data_rows_start_index_in_fetched_list = (
                self.data_start_row - self.header_row
            )
            data_values = all_sheet_values[data_rows_start_index_in_fetched_list:]

            if not data_values:
                logger.warning(
                    f"No data rows found after header in '{self.circuit_name}'."
                )
                self.df = pd.DataFrame(columns=self.data_cols)
                return True

            # --- Pad data_values if rows are shorter than raw_column_names ---
            num_expected_cols = len(raw_column_names)
            data_values_padded = []
            for i, row in enumerate(data_values):
                row_len = len(row)
                if row_len < num_expected_cols:
                    # Pad the row with empty strings (or None, or any placeholder)
                    padding = [""] * (num_expected_cols - row_len)
                    padded_row = row + padding
                    data_values_padded.append(padded_row)
                    if i == 0:  # Log for the first problematic row for easier debugging
                        logger.debug(
                            f"Padded first data row. Original length: {row_len}, Target length: {num_expected_cols}. Original row: {row[:10]}... Padded row: {padded_row[:10]}..."
                        )
                else:
                    # If row is already long enough (or longer, though pandas will truncate if more cols than names)
                    data_values_padded.append(
                        row[:num_expected_cols]
                    )  # Ensure it's not longer

                data_values = data_values_padded  # Use the padded data

            if len(raw_column_names) < len(self.data_cols):
                logger.error(
                    f"CRITICAL COLUMN COUNT MISMATCH for worksheet '{self.circuit_name}': "
                    f"Cannot create DataFrame. Header columns: {len(raw_column_names)}, Data columns: {len(data_values[0])}. "
                    f"Check the header row and data start row."
                    f"This may indicate a structural change in the sheet."
                    f"If the head row is correct, check DATA_COLS and adjust if needed."
                )
                return False

            self.df = pd.DataFrame(data_values, columns=raw_column_names)
            logger.info(
                f"Initial DataFrame: {self.df.shape[0]} rows, {self.df.shape[1]} columns."
            )
            return True

        except gspread.exceptions.SpreadsheetNotFound:
            logger.error(f"Spreadsheet not found: {self.sheet_url}")
            return False
        except gspread.exceptions.WorksheetNotFound:
            logger.error(
                f"Worksheet '{self.circuit_name}' not found in {self.sheet_url}"
            )
            return False
        except Exception as e:
            logger.error(f"Error processing '{self.circuit_name}': {e}", exc_info=True)
            return False

    def _normalize_column_names(self):

        cleaned_cols = self.df.columns.str.strip().str.lower()
        for char, replacement in self.colname_replacements.items():
            cleaned_cols = cleaned_cols.str.replace(char, replacement, regex=False)

        self.df.columns = cleaned_cols
        rename_map = {"squirt_boom": "requires_squirt_boom", "status": "is_complete"}

        # --- Dynamically add to rename_map for pattern-based 'site_no' ---
        site_id_found_raw_name = (
            None  # Initialize to track if we've found a site_id column
        )

        for (
            col_name
        ) in self.df.columns:  # Iterate through the *now cleaned* column names
            if col_name.startswith("site_no") or col_name.startswith("site_num"):
                if site_id_found_raw_name:  # Already found one
                    logger.warning(
                        f"Multiple columns found starting with 'site_no/site_num': "
                        f"'{site_id_found_raw_name}' and '{col_name}'. "
                        f"Using the first one found: '{site_id_found_raw_name}'."
                    )
                else:  # This is the first one we've found
                    site_id_found_raw_name = col_name  # Store the name we found
                    logger.debug(
                        f"Identified column '{site_id_found_raw_name}' to be renamed to 'site_no'."
                    )
                    rename_map[site_id_found_raw_name] = (
                        "site_no"  # Add to map: current_name -> new_standard_name
                    )

        if not site_id_found_raw_name:
            logger.warning(
                "No column found starting with 'site_no' or 'site_num'. Standard 'site_no' column will not be created through renaming."
            )

        # Perform the rename operation using the accumulated map
        if rename_map:  # Only rename if there's something in the map
            self.df.rename(columns=rename_map, inplace=True)
            logger.debug(f"Columns after renaming: {self.df.columns.tolist()}")

        else:
            logger.debug(
                "No columns were identified for renaming based on the rename_map"
            )

    def _get_data_rows(self):
        """
        Filters self.df to include rows from the first row where 'site_no' is numeric
        to the last row where 'site_no' is numeric. Modifies self.df in place.
        """
        if self.df is None or self.df.empty:
            logger.warning(
                "DataFrame is empty or None. Cannot filter by 'site_no' range."
            )
            return

        key_column = "site_no"  # The column to check for numeric values

        if key_column not in self.df.columns:
            logger.error(
                f"Key column '{key_column}' not found in DataFrame. Cannot filter by range."
            )
            # Optionally, you might want to set self.df to an empty frame or handle this error
            return

        # --- Find the first potential data row ---
        # We're looking for something that can be an integer.
        # `errors='coerce'` will turn non-numbers into NaN.
        numeric_series = pd.to_numeric(self.df[key_column], errors="coerce")

        # A valid site number for starting the block must be a number (not NaN)
        potential_start_indices = self.df[numeric_series.notna()].index

        if potential_start_indices.empty:
            logger.warning(
                f"No numeric-like values found in '{key_column}' to start data block."
            )
            self.df = pd.DataFrame(columns=self.df.columns)  # Make DF empty
            return

        first_potential_data_index_label = potential_start_indices.min()

        # --- Find the end of the contiguous data block ---
        last_contiguous_data_index_label = first_potential_data_index_label
        num_of_contiguous_invalid = 0  # only cut if multiple (3) NaN in a row to account for regular missing data

        # Get the integer position of the first potential data row
        try:
            start_pos = self.df.index.get_loc(first_potential_data_index_label)
        except KeyError:
            logger.error(
                f"Could not find index label {first_potential_data_index_label} after filtering. This should not happen."
            )
            return

        for i in range(start_pos, len(self.df)):
            current_index_label = self.df.index[i]
            current_value_in_key_col = self.df.loc[current_index_label, key_column]
            # Define what a "valid ongoing data row" looks like
            # For site_no, it's not NaN (for now)
            # If it's blank or text like "Total Sites", the block has ended.
            is_valid_ongoing_site_no = False
            if (
                pd.notna(current_value_in_key_col)
                and str(current_value_in_key_col).strip() != ""
            ):
                is_valid_ongoing_site_no = True
                # Add more specific checks if needed, e.g. positive, within a range.

            if is_valid_ongoing_site_no:
                last_contiguous_data_index_label = current_index_label
                num_of_contiguous_invalid = 0

            elif not is_valid_ongoing_site_no and num_of_contiguous_invalid < 3:
                num_of_contiguous_invalid += 1
                logger.debug(
                    f"Invalid row encountered at index {current_index_label} (value: '{self.df.loc[current_index_label, key_column]}'). "
                    f"Contiguous invalid count: {num_of_contiguous_invalid}"
                )

            else:
                last_contiguous_data_index_label = last_contiguous_data_index_label
                # This row is no longer a valid site number, so the block ended at the previous row.
                logger.debug(
                    f"End of contiguous '{key_column}' block detected at index before {current_index_label} "
                    f"(value: '{current_value_in_key_col}')"
                )
                break  # Stop iterating

        logger.info(
            f"Dynamically identified data block for '{key_column}': "
            f"starts at index label {first_potential_data_index_label}, "
            f"ends at index label {last_contiguous_data_index_label}."
        )

        self.df = self.df.loc[
            first_potential_data_index_label:last_contiguous_data_index_label
        ].copy()
        logger.info(
            f"DataFrame sliced to contiguous '{key_column}' block. New shape: {self.df.shape}"
        )

    def _apply_value_transformations(self):
        self.df.replace(self.value_replacements, inplace=True)
        self.df = self.df.infer_objects(copy=False)
        logger.info("Applied general value_replacements and inferred objects.")

        if "requires_squirt_boom" in self.df.columns:
            true_values = ["TRUE", "T", "YES", "Y", "1"]
            self.df["requires_squirt_boom"] = (
                self.df["requires_squirt_boom"]
                .astype(str)
                .str.upper()
                .isin(true_values)
            )
            logger.debug("Converted 'requires_squirt_boom' to boolean.")

        if "is_complete" in self.df.columns:
            # Ensure 'is_complete' is boolean after VALUE_REPLACEMENTS
            # Handles cases where a value wasn't in the map and might be e.g. a number if "X" was in that column
            self.df["is_complete"] = self.df["is_complete"].apply(
                lambda x: x if isinstance(x, bool) else False
            )
            logger.debug("Ensured 'is_complete' is boolean.")
        else:
            logger.warning(
                "'is_complete' column not found after cleaning. Cannot filter by completion status."
            )

    def _filter_and_select_final_columns(self):

        available_final_cols = [col for col in self.data_cols if col in self.df.columns]
        missing_final_cols = [
            col for col in self.data_cols if col not in available_final_cols
        ]
        if missing_final_cols:
            logger.warning(
                f"For the final DataFrame, missing expected columns: {missing_final_cols}. They will be excluded."
            )

        # Filter for incomplete items

        if "is_complete" in self.df.columns and not self.df.empty:
            # Ensure the column exists before trying to filter on it
            self.df = self.df.loc[
                self.df["is_complete"] == False, available_final_cols
            ].copy()
        elif not self.df.empty:
            # 'is_complete' column is missing, return available columns without this filter
            logger.warning(
                "Cannot filter by 'is_complete' as column is missing. Returning selected columns unfiltered by completion."
            )
            self.df = self.df.loc[:, available_final_cols].copy()
        else:  # self.df is empty
            self.df = pd.DataFrame(columns=available_final_cols)
        logger.info(f"Filtering complete. Final DataFrame shape: {self.df.shape}")

    def clean(self):
        logger.info(f"Starting cleaning process for worksheet: {self.circuit_name}...")
        if not self._fetch_raw_df():
            logger.error("Failed to fetch initial data. Aborting cleaning process.")
            # _fetch_data might return an empty df with specific columns if no data was found
            # but still indicates a "success" in terms of not having an exception.
            # If _fetch_data returns False due to an exception or critical issue:
            if (
                self.df is None
            ):  # If df was never even initialized due to critical fetch error
                return None
            # The current _fetch_data returns False on critical error, or if no data (sets self.df to empty)
            # if _fetch_data returns False, we assume a critical issue.
            # If self.df is already an empty DataFrame with final_output_cols, that's fine.
            if (
                self.df is not None
                and self.df.empty
                and list(self.df.columns) == self.final_output_cols
            ):
                logger.warning(
                    "No data to process, returning empty DataFrame with expected columns."
                )
                return self.df
            return None  # Critical fetch error

        self._normalize_column_names()
        self._get_data_rows()
        self._apply_value_transformations()
        self._filter_and_select_final_columns()  # Corrected method name

        logger.info(
            f"Cleaning process finished. Final DataFrame shape: {self.df.shape if self.df is not None else 'None'}"
        )
        return self.df
