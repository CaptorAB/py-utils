"""Module for downloading and processing TPT reports."""

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
SMALL_VALUE_THRESHOLD = 0.000001


class TPTDownloadError(Exception):
    """Exception raised when TPT report download fails."""


class TPTProcessingError(Exception):
    """Exception raised when TPT report processing fails."""


def _download_single_report(
    report_id: str,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Download a single TPT report from the API.

    Args:
        report_id: The ID of the report to download.
        timeout: Optional timeout in seconds for the request.

    Returns:
        The downloaded report data as a dictionary.

    Raises:
        TPTDownloadError: If the report download fails.

    """
    url = f"https://api.captor.se/public/api/tpts/{report_id}"
    response = requests.get(url=url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def download_fund_tpt_report(
    report_id: str,
    directory: Path | None = None,
    timeout: int | None = None,
    max_retries: int = 3,
) -> Path:
    """Download and process a TPT report for a fund.

    Args:
        report_id: The ID of the report to download.
        directory: Optional directory to save the report.
        timeout: Optional timeout in seconds for the request.
        max_retries: Maximum number of retry attempts.

    Returns:
        Path to the saved Excel file.

    Raises:
        TPTDownloadError: If the report download fails after retries.
        TPTProcessingError: If the report processing fails.

    """
    last_error = None
    for attempt in range(max_retries):
        try:
            data = _download_single_report(report_id=report_id, timeout=timeout)
            report_data = pd.DataFrame(data["data"])
            processed_data = report_data.map(replace_small_values)

            output_dir = directory or Path.cwd()
            output_file = output_dir / f"{data['name']}.xlsx"
            processed_data.to_excel(output_file, index=False)
        except requests.RequestException as e:  # noqa: PERF203
            last_error = e
            if attempt == max_retries - 1:
                raise
            logger.warning(
                "Attempt %d failed, retrying...",
                attempt + 1,
            )
        else:
            return output_file

    msg = f"Failed to download report after {max_retries} attempts"
    if last_error:
        raise last_error
    raise TPTDownloadError(msg)


def collate_fund_tpt_reports(
    report_ids: list[str],
    sheetfile: Path,
    timeout: int | None = None,
) -> Path:
    """Download and collate multiple TPT reports into a single Excel file.

    Args:
        report_ids: List of report IDs to download.
        sheetfile: Path to save the collated Excel file.
        timeout: Optional timeout in seconds for each request.

    Returns:
        Path to the saved Excel file.

    Raises:
        TPTDownloadError: If any report download fails.
        TPTProcessingError: If report processing fails.

    """
    report_dataframes = []
    for report_id in report_ids:
        data = _download_single_report(report_id=report_id, timeout=timeout)
        report_data = pd.DataFrame(data["data"])
        processed_data = report_data.map(replace_small_values)
        report_dataframes.append(processed_data)

    try:
        result = pd.concat(report_dataframes, axis=0, ignore_index=True)
        result.to_excel(sheetfile, index=False)
    except Exception as e:
        msg = f"Error collating TPT reports: {e!s}"
        raise TPTProcessingError(msg) from e
    else:
        return sheetfile


def sort_key(col_name: str) -> tuple[float | int, str]:
    """Extract sort key from column name.

    Args:
        col_name: Column name to process.

    Returns:
        Tuple of (numeric part, string part) for sorting.

    """
    match = re.match(r"(\d+)([A-Za-z]*)_", col_name)
    if match:
        number = int(match.group(1))
        letter = match.group(2)
        return number, letter

    return float("inf"), col_name


def replace_small_values(value: float | str | None) -> float | str | None:
    """Replace very small numeric values with 0.

    Args:
        value: Value to process.

    Returns:
        Processed value with small numbers replaced by 0.

    """
    if isinstance(value, (int, float)) and abs(value) < SMALL_VALUE_THRESHOLD:
        return 0.0
    return value


if __name__ == "__main__":
    xlsxpath = Path.home() / "Documents"
    _ = download_fund_tpt_report(
        report_id="67a5ca93079b64d59bb66ccd",
        directory=xlsxpath,
    )
