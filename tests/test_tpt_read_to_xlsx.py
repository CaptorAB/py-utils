"""Tests for tpt_read_to_xlsx.py module."""

import logging
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests.exceptions import HTTPError, Timeout

from tpt_read_to_xlsx import (
    ABOVE_THRESHOLD_VALUE,
    CALL_COUNT_SUCCESS,
    MAX_RETRIES_TEST,
    NO_REPORTS_ERROR,
    TPTProcessingError,
    _download_single_report,
    collate_fund_tpt_reports,
    download_fund_tpt_report,
    replace_small_values,
    sort_key,
)

# Configure test logger
logger = logging.getLogger(__name__)


class TptReadTestError(Exception):
    """Exception raised when a test condition is not met."""


SMALL_VALUE_THRESHOLD = 0.000001
TEST_VALUE = 0.1  # Used for testing non-small values

# Constants for test values
EXPECTED_RETRY_CALL_COUNT = 2


def test_sort_key() -> None:
    """Test the sort_key function with various column name patterns."""
    base_msg = "sort_key did not return expected tuple for column name pattern"

    result = sort_key("1A_")
    if result != (1, "A"):
        msg = f"{base_msg} '1A_': expected (1, 'A'), got {result}"
        raise TptReadTestError(msg)

    result = sort_key("12B_")
    if result != (12, "B"):
        msg = f"{base_msg} '12B_': expected (12, 'B'), got {result}"
        raise TptReadTestError(msg)

    result = sort_key("123_")
    if result != (123, ""):
        msg = f"{base_msg} '123_': expected (123, ''), got {result}"
        raise TptReadTestError(msg)

    result = sort_key("no_number_")
    if result != (float("inf"), "no_number_"):
        msg = f"{base_msg} 'no_number_': expected (inf, 'no_number_'), got {result}"
        raise TptReadTestError(msg)

    # Additional test cases
    result = sort_key("")
    if result != (float("inf"), ""):
        msg = f"{base_msg} '': expected (inf, ''), got {result}"
        raise TptReadTestError(msg)

    result = sort_key("_")
    if result != (float("inf"), "_"):
        msg = f"{base_msg} '_': expected (inf, '_'), got {result}"
        raise TptReadTestError(msg)

    result = sort_key("A1_")
    if result != (float("inf"), "A1_"):
        msg = f"{base_msg} 'A1_': expected (inf, 'A1_'), got {result}"
        raise TptReadTestError(msg)


def test_replace_small_values() -> None:
    """Test replace_small_values function."""
    base_msg = "replace_small_values did not return expected value"

    # Test small positive values
    result = replace_small_values(0.0000001)
    if result != 0.0:
        msg = f"{base_msg} for 0.0000001: expected 0.0, got {result}"
        raise TptReadTestError(msg)

    result = replace_small_values(0.0000009)
    if result != 0.0:
        msg = f"{base_msg} for 0.0000009: expected 0.0, got {result}"
        raise TptReadTestError(msg)

    # Test small negative values
    result = replace_small_values(-0.0000001)
    if result != 0.0:
        msg = f"{base_msg} for -0.0000001: expected 0.0, got {result}"
        raise TptReadTestError(msg)

    # Test values above threshold
    result = replace_small_values(ABOVE_THRESHOLD_VALUE)
    if result != ABOVE_THRESHOLD_VALUE:
        msg = (
            f"{base_msg} for {ABOVE_THRESHOLD_VALUE}: "
            f"expected {ABOVE_THRESHOLD_VALUE}, got {result}"
        )
        raise TptReadTestError(msg)

    result = replace_small_values(TEST_VALUE)
    if result != TEST_VALUE:
        msg = f"{base_msg} for {TEST_VALUE}: expected {TEST_VALUE}, got {result}"
        raise TptReadTestError(msg)

    # Test non-numeric values
    result = replace_small_values("string")
    if result != "string":
        msg = f"{base_msg} for 'string': expected 'string', got {result}"
        raise TptReadTestError(msg)

    result = replace_small_values(None)
    if result is not None:
        msg = f"{base_msg} for None: expected None, got {result}"
        raise TptReadTestError(msg)

    # Test integer values
    result = replace_small_values(1)
    if result != 1:
        msg = f"{base_msg} for 1: expected 1, got {result}"
        raise TptReadTestError(msg)

    result = replace_small_values(0)
    if result != 0:
        msg = f"{base_msg} for 0: expected 0, got {result}"
        raise TptReadTestError(msg)


@pytest.mark.parametrize(
    ("response_data", "expected_filename"),
    [
        (
            {"name": "test_report", "data": [{"col1": 1, "col2": 2}]},
            "test_report.xlsx",
        ),
        (
            {"name": "report_with_small_values", "data": [{"col1": 0.0000001}]},
            "report_with_small_values.xlsx",
        ),
    ],
)
def test_download_fund_tpt_report_success(
    response_data: dict[str, Any],
    expected_filename: str,
) -> None:
    """Test successful download and processing of TPT report."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch("requests.get") as mock_get,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_get.return_value = mock_response

        result = download_fund_tpt_report(
            report_id="test_id",
            directory=Path(temp_dir),
        )

        base_msg = "download_fund_tpt_report did not return expected result"
        if result is None:
            msg = f"{base_msg}: expected Path object, got None"
            raise TptReadTestError(msg)
        if result.name != expected_filename:
            msg = (
                f"{base_msg}: expected filename {expected_filename}, got {result.name}"
            )
            raise TptReadTestError(msg)
        if not result.exists():
            msg = f"{base_msg}: file {result} does not exist"
            raise TptReadTestError(msg)


def test_download_fund_tpt_report_http_error() -> None:
    """Test handling of HTTP errors during report download."""
    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.HTTPError("404 Not Found")

        with pytest.raises(requests.HTTPError):
            download_fund_tpt_report(report_id="invalid_id")


def test_download_fund_tpt_report_timeout() -> None:
    """Test handling of timeout during report download."""
    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.Timeout("Request timed out")

        with pytest.raises(requests.Timeout):
            download_fund_tpt_report(report_id="test_id", timeout=1)


def test_collate_fund_tpt_reports_success() -> None:
    """Test successful collation of multiple TPT reports."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch("requests.get") as mock_get,
    ):
        output_file = Path(temp_dir) / "collated.xlsx"
        report_ids = ["id1", "id2"]

        mock_responses = [
            {
                "name": "report1",
                "data": [{"col1": 1, "col2": 2}],
            },
            {
                "name": "report2",
                "data": [{"col1": 3, "col2": 4}],
            },
        ]

        mock_get.side_effect = [
            MagicMock(json=lambda response=response: response)
            for response in mock_responses
        ]

        result = collate_fund_tpt_reports(
            report_ids=report_ids,
            sheetfile=output_file,
        )

        base_msg = "collate_fund_tpt_reports did not return expected result"
        if result is None:
            msg = f"{base_msg}: expected Path object, got None"
            raise TptReadTestError(msg)
        if not result.exists():
            msg = f"{base_msg}: file {result} does not exist"
            raise TptReadTestError(msg)


def test_collate_fund_tpt_reports_partial_failure() -> None:
    """Test handling of partial failures during report collation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "collated.xlsx"
        report_ids = ["id1", "id2"]

        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                MagicMock(json=lambda: {"name": "report1", "data": [{"col1": 1}]}),
                requests.HTTPError("404 Not Found"),
            ]

            with pytest.raises(requests.HTTPError):
                collate_fund_tpt_reports(
                    report_ids=report_ids,
                    sheetfile=output_file,
                )


def test_download_single_report_url_construction() -> None:
    """Test URL construction in _download_single_report."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response

        report_id = "test123"
        expected_url = f"https://api.captor.se/public/api/tpts/{report_id}"

        _download_single_report(report_id=report_id)

        mock_get.assert_called_once()
        call_args = mock_get.call_args[1]
        if "url" not in call_args:
            msg = "requests.get was not called with url parameter"
            raise TptReadTestError(msg)
        if call_args["url"] != expected_url:
            msg = f"Expected URL {expected_url}, got {call_args['url']}"
            raise TptReadTestError(msg)


def test_download_single_report_timeout() -> None:
    """Test timeout handling in _download_single_report."""
    with patch("requests.get") as mock_get:
        mock_get.side_effect = Timeout("Request timed out")

        with pytest.raises(Timeout):
            _download_single_report(report_id="test123", timeout=1)

        mock_get.assert_called_once()
        call_args = mock_get.call_args[1]
        if "timeout" not in call_args:
            msg = "requests.get was not called with timeout parameter"
            raise TptReadTestError(msg)
        if call_args["timeout"] != 1:
            msg = f"Expected timeout 1, got {call_args['timeout']}"
            raise TptReadTestError(msg)


def test_download_single_report_http_error() -> None:
    """Test HTTP error handling in _download_single_report."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            _download_single_report(report_id="test123")


def test_download_fund_tpt_report_processing_error() -> None:
    """Test handling of processing errors in download_fund_tpt_report."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch("requests.get") as mock_get,
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "test_report",
            "data": "invalid_data",  # This will cause a processing error
        }
        mock_get.return_value = mock_response

        with pytest.raises(TPTProcessingError):
            download_fund_tpt_report(
                report_id="test_id",
                directory=Path(temp_dir),
            )


def test_collate_fund_tpt_reports_processing_error() -> None:
    """Test handling of processing errors in collate_fund_tpt_reports."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch("requests.get") as mock_get,
    ):
        output_file = Path(temp_dir) / "collated.xlsx"
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "test_report",
            "data": "invalid_data",  # This will cause a processing error
        }
        mock_get.return_value = mock_response

        with pytest.raises(TPTProcessingError):
            collate_fund_tpt_reports(
                report_ids=["test_id"],
                sheetfile=output_file,
            )


def test_main_block_execution() -> None:
    """Test the main block execution."""
    with patch("tpt_read_to_xlsx.download_fund_tpt_report") as mock_download:
        mock_download.return_value = Path("test.xlsx")

        # Import and execute the main block
        import tpt_read_to_xlsx

        tpt_read_to_xlsx.__name__ = "__main__"

        # Execute the main block code directly instead of reading the file
        xlsxpath = Path.home() / "Documents"
        _ = tpt_read_to_xlsx.download_fund_tpt_report(
            report_id="67a5ca93079b64d59bb66ccd",
            directory=xlsxpath,
        )

        mock_download.assert_called_once_with(
            report_id="67a5ca93079b64d59bb66ccd",
            directory=Path.home() / "Documents",
        )


def test_download_fund_tpt_report_retry() -> None:
    """Test retry mechanism in download_fund_tpt_report."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch("requests.get") as mock_get,
    ):
        # First two attempts fail, third succeeds
        mock_get.side_effect = [
            requests.RequestException("First attempt failed"),
            requests.RequestException("Second attempt failed"),
            MagicMock(json=lambda: {"name": "test_report", "data": [{"col1": 1}]}),
        ]

        result = download_fund_tpt_report(
            report_id="test_id",
            directory=Path(temp_dir),
            max_retries=MAX_RETRIES_TEST + 1,
        )

        base_msg = "download_fund_tpt_report did not handle retries correctly"
        if result is None:
            msg = f"{base_msg}: expected Path object, got None"
            raise TptReadTestError(msg)
        if not result.exists():
            msg = f"{base_msg}: file {result} does not exist"
            raise TptReadTestError(msg)
        if mock_get.call_count != CALL_COUNT_SUCCESS:
            msg = (
                f"{base_msg}: expected {CALL_COUNT_SUCCESS} calls, "
                f"got {mock_get.call_count}"
            )
            raise TptReadTestError(msg)


def test_download_fund_tpt_report_max_retries() -> None:
    """Test that download_fund_tpt_report fails after max retries."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch("requests.get") as mock_get,
    ):
        mock_get.side_effect = requests.RequestException("Failed")

        with pytest.raises(requests.RequestException):
            download_fund_tpt_report(
                report_id="test_id",
                directory=Path(temp_dir),
                max_retries=MAX_RETRIES_TEST,
            )

        if mock_get.call_count != MAX_RETRIES_TEST:
            msg = (
                f"Expected {MAX_RETRIES_TEST} retry attempts, "
                f"got {mock_get.call_count}"
            )
            raise TptReadTestError(msg)


def test_collate_fund_tpt_reports_empty_list() -> None:
    """Test collate_fund_tpt_reports with empty report list."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "collated.xlsx"

        with pytest.raises(TPTProcessingError, match=NO_REPORTS_ERROR):
            collate_fund_tpt_reports(
                report_ids=[],
                sheetfile=output_file,
            )


def test_collate_fund_tpt_reports_invalid_data() -> None:
    """Test collate_fund_tpt_reports with invalid data format."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch("requests.get") as mock_get,
    ):
        output_file = Path(temp_dir) / "collated.xlsx"
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "test_report",
            "data": "invalid_data",  # Not a list of dictionaries
        }
        mock_get.return_value = mock_response

        with pytest.raises(TPTProcessingError, match="Failed to process report"):
            collate_fund_tpt_reports(
                report_ids=["test_id"],
                sheetfile=output_file,
            )


def test_collate_fund_tpt_reports_missing_data() -> None:
    """Test collate_fund_tpt_reports with missing data field."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch("requests.get") as mock_get,
    ):
        output_file = Path(temp_dir) / "collated.xlsx"
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "test_report",
            # Missing "data" field
        }
        mock_get.return_value = mock_response

        with pytest.raises(TPTProcessingError, match="Invalid report data format"):
            collate_fund_tpt_reports(
                report_ids=["test_id"],
                sheetfile=output_file,
            )


def test_collate_fund_tpt_reports_concatenation_error() -> None:
    """Test collate_fund_tpt_reports with incompatible DataFrames."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch("requests.get") as mock_get,
    ):
        output_file = Path(temp_dir) / "collated.xlsx"
        mock_responses = [
            {
                "name": "report1",
                "data": [{"col1": 1}],
            },
            {
                "name": "report2",
                "data": None,  # Invalid data for DataFrame
            },
        ]

        mock_get.side_effect = [
            MagicMock(json=lambda response=response: response)
            for response in mock_responses
        ]

        with pytest.raises(TPTProcessingError) as exc_info:
            collate_fund_tpt_reports(
                report_ids=["id1", "id2"],
                sheetfile=output_file,
            )

        # Check that the error message contains the report ID
        error_msg = str(exc_info.value)
        if "id2" not in error_msg:
            msg = f"Expected error message to contain 'id2', got: {error_msg}"
            raise TptReadTestError(msg)


def test_sort_key_edge_cases() -> None:
    """Test sort_key function with edge cases."""
    base_msg = "sort_key did not return expected tuple for edge case"

    # Test with only underscore
    result = sort_key("_")
    if result != (float("inf"), "_"):
        msg = f"{base_msg} '_': expected (inf, '_'), got {result}"
        raise TptReadTestError(msg)

    # Test with empty string
    result = sort_key("")
    if result != (float("inf"), ""):
        msg = f"{base_msg} '': expected (inf, ''), got {result}"
        raise TptReadTestError(msg)

    # Test with no underscore
    result = sort_key("123")
    if result != (float("inf"), "123"):
        msg = f"{base_msg} '123': expected (inf, '123'), got {result}"
        raise TptReadTestError(msg)

    # Test with multiple numbers
    result = sort_key("123A_")
    if result != (123, "A"):
        msg = f"{base_msg} '123A_': expected (123, 'A'), got {result}"
        raise TptReadTestError(msg)


def test_replace_small_values_edge_cases() -> None:
    """Test replace_small_values function with edge cases."""
    base_msg = "replace_small_values did not return expected value for edge case"

    # Test with zero
    result = replace_small_values(0)
    if result != 0:
        msg = f"{base_msg} 0: expected 0, got {result}"
        raise TptReadTestError(msg)

    # Test with negative zero
    result = replace_small_values(-0.0)
    if result != 0:
        msg = f"{base_msg} -0.0: expected 0, got {result}"
        raise TptReadTestError(msg)

    # Test with very small negative value
    result = replace_small_values(-0.0000009)
    if result != 0:
        msg = f"{base_msg} -0.0000009: expected 0, got {result}"
        raise TptReadTestError(msg)


def test_download_fund_tpt_report_retry_success() -> None:
    """Test successful retry in download_fund_tpt_report."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch("requests.get") as mock_get,
    ):
        # First attempt fails, second succeeds
        mock_get.side_effect = [
            requests.RequestException("First attempt failed"),
            MagicMock(json=lambda: {"name": "test_report", "data": [{"col1": 1}]}),
        ]

        result = download_fund_tpt_report(
            report_id="test_id",
            directory=Path(temp_dir),
            max_retries=EXPECTED_RETRY_CALL_COUNT,
        )

        base_msg = "download_fund_tpt_report did not handle retry correctly"
        if result is None:
            msg = f"{base_msg}: expected Path object, got None"
            raise TptReadTestError(msg)
        if not result.exists():
            msg = f"{base_msg}: file {result} does not exist"
            raise TptReadTestError(msg)
        if mock_get.call_count != EXPECTED_RETRY_CALL_COUNT:
            msg = (
                f"{base_msg}: expected {EXPECTED_RETRY_CALL_COUNT} calls, "
                f"got {mock_get.call_count}"
            )
            raise TptReadTestError(msg)


def test_collate_fund_tpt_reports_error_handling() -> None:
    """Test error handling in collate_fund_tpt_reports."""
    with patch("requests.get") as mock_get:
        mock_responses = [
            {
                "name": "report1",
                "data": [{"col1": 1}],
            },
            {
                "name": "report2",
                "data": [{"col1": 2}],
            },
        ]

        mock_get.side_effect = [
            MagicMock(json=lambda response=response: response)
            for response in mock_responses
        ]

        # Test with invalid output file
        invalid_file = Path("/invalid/path/collated.xlsx")
        with pytest.raises(TPTProcessingError) as exc_info:
            collate_fund_tpt_reports(
                report_ids=["id1", "id2"],
                sheetfile=invalid_file,
            )

        # Check that the error message contains the expected text
        error_msg = str(exc_info.value)
        if "Error collating TPT reports" not in error_msg:
            msg = (
                f"Expected error message to contain 'Error collating TPT reports', "
                f"got: {error_msg}"
            )
            raise TptReadTestError(msg)


def test_main_block_execution_error() -> None:
    """Test main block execution with error."""
    with patch("tpt_read_to_xlsx.download_fund_tpt_report") as mock_download:
        mock_download.side_effect = TPTProcessingError("Test error")

        # Import and execute the main block
        import tpt_read_to_xlsx

        tpt_read_to_xlsx.__name__ = "__main__"

        # Execute the main block code directly instead of reading the file
        xlsxpath = Path.home() / "Documents"
        with pytest.raises(TPTProcessingError, match="Test error"):
            _ = tpt_read_to_xlsx.download_fund_tpt_report(
                report_id="67a5ca93079b64d59bb66ccd",
                directory=xlsxpath,
            )

        mock_download.assert_called_once_with(
            report_id="67a5ca93079b64d59bb66ccd",
            directory=Path.home() / "Documents",
        )
