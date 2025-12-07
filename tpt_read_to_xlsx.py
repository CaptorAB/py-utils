"""Module for downloading and processing TPT reports."""

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Any

import requests
from pandas import DataFrame, concat

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
SMALL_VALUE_THRESHOLD = 0.000001
ABOVE_THRESHOLD_VALUE = 0.0000011
MAX_RETRIES = 3
MAX_RETRIES_TEST = 2
CALL_COUNT_SUCCESS = 3

# Error messages
INVALID_REPORT_FORMAT = "Invalid report data format"
INVALID_REPORT_FORMAT_WITH_ID = "Invalid report data format for report {report_id}"
PROCESSING_ERROR = "Failed to process report data: {error}"
REPORT_PROCESSING_ERROR = "Failed to process report {report_id}: {error}"
COLLATION_ERROR = "Error collating TPT reports: {error}"
NO_REPORTS_ERROR = "No reports to collate"


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
        timeout: Timeout in seconds for each request. Default is 10.

    Returns:
        The downloaded report data as a dictionary.

    """
    url = f"https://api.captor.se/public/api/tpts/{report_id}"
    response = requests.get(url=url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _download_report_list(
    report_date: dt.date,
    timeout: int = 10,
) -> list[dict[str, str]]:
    """Download a single TPT report from the API.

    Args:
        report_date: The date for the required report list.
        timeout: Optional timeout in seconds for the request.

    Returns:
        A list of dictionaries with meta data for each report.

    """
    url = "https://api.captor.se/public/api/tpts"
    response = requests.get(url=url, timeout=timeout)
    response.raise_for_status()
    return [
        report
        for report in response.json()
        if report["date"] == report_date.strftime("%Y-%m-%d")
    ]


def download_fund_tpt_report(
    report_id: str,
    directory: Path | None = None,
    timeout: int = 10,
    max_retries: int = MAX_RETRIES,
) -> Path:
    """Download and process a TPT report for a fund.

    Args:
        report_id: The ID of the report to download.
        directory: Optional directory to save the report.
        timeout: Timeout in seconds for each request. Default is 10.
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
            if not isinstance(data, dict) or "data" not in data or "name" not in data:
                raise TPTProcessingError(INVALID_REPORT_FORMAT)

            try:
                report_data = make_dataframe(data=data["data"])
            except Exception as e:
                error_msg = PROCESSING_ERROR.format(error=str(e))
                raise TPTProcessingError(error_msg) from e

            output_dir = directory or Path.cwd()
            output_file = output_dir / f"{data['name']}.xlsx"
            report_data.to_excel(output_file, index=False)
        except requests.RequestException as e:
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
    sheetfile: Path,
    report_ids: list[str] | None = None,
    report_date: dt.date | None = None,
    timeout: int = 10,
) -> Path:
    """Download and collate multiple TPT reports into a single Excel file.

    Args:
        sheetfile: Path to save the collated Excel file.
        report_ids: Optional list of report IDs to download.
        report_date: The date for the report list if no report_ids are given.
        timeout: Timeout in seconds for each request. Default is 10.

    Returns:
        Path to the saved Excel file.

    Raises:
        TPTDownloadError: If any report download fails.
        TPTProcessingError: If report processing fails.

    """
    if not report_ids and not report_date:
        raise TPTProcessingError(NO_REPORTS_ERROR)

    report_dataframes = []

    if not report_ids:
        if not report_date:
            message = "When report_ids are not given a date must be provided."
            raise TPTProcessingError(message)
        reports = _download_report_list(report_date=report_date, timeout=timeout)
        if len(reports) == 0:
            message = (
                "Report list fetch returned no data for "
                f"date {report_date.strftime('%Y-%m-%d')}"
            )
            raise TPTDownloadError(message)
        report_ids = [rpt["_id"] for rpt in reports]

    for report_id in report_ids:
        data = _download_single_report(report_id=report_id, timeout=timeout)
        if not isinstance(data, dict) or "data" not in data:
            error_msg = INVALID_REPORT_FORMAT_WITH_ID.format(report_id=report_id)
            raise TPTProcessingError(error_msg)

        if not isinstance(data["data"], list):
            error_msg = REPORT_PROCESSING_ERROR.format(
                report_id=report_id,
                error="Data must be a list of dictionaries",
            )
            raise TPTProcessingError(error_msg)

        try:
            report_data = make_dataframe(data=data["data"])
            report_dataframes.append(report_data)
        except Exception as e:
            error_msg = REPORT_PROCESSING_ERROR.format(
                report_id=report_id,
                error=str(e),
            )
            raise TPTProcessingError(error_msg) from e

    try:
        result = concat(report_dataframes, axis=0, ignore_index=True)
        result.to_excel(sheetfile, index=False)
    except Exception as e:
        error_msg = COLLATION_ERROR.format(error=str(e))
        raise TPTProcessingError(error_msg) from e
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


def make_dataframe(data: list[dict]) -> DataFrame:
    """Replace very small numeric values with 0, sort columns and create DataFrame.

    Args:
        data: data to process.

    Returns:
        Pandas DataFrame with data cleaned and sorted.

    """
    dataframe = DataFrame(data=data)
    sorted_columns = sorted(dataframe.columns, key=sort_key)
    return dataframe[sorted_columns].apply(lambda col: col.map(replace_small_values))


if __name__ == "__main__":
    rpt_date = dt.date(2025, 8, 29)
    xlsxpath = (
        Path.home() / "Documents" / f"allreports_{rpt_date.strftime('%Y%m%d')}.xlsx"
    )
    _ = collate_fund_tpt_reports(sheetfile=xlsxpath, report_date=rpt_date)
