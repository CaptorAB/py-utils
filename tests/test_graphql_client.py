"""Test suite for the graphql_client.py module."""

import json
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest
import requests

import graphql_client


class GraphqlClientTestError(Exception):
    """Custom exception used for signaling test failures."""


TESTERRORMESSAGE = "Code not working as intended."


@pytest.fixture
def dummy_token() -> str:
    """Fixture that provides a dummy JWT token string.

    Returns:
        str: A dummy JWT token.

    """
    return "dummy.jwt.token"


@pytest.fixture
def dummy_decoded() -> dict[str, str]:
    """Fixture that provides a decoded dummy JWT token.

    Returns:
        dict[str, str]: Decoded token content with 'aud' and 'unique_name'.

    """
    return {"aud": "prod", "unique_name": "tester"}


def test_check_internet_success() -> None:
    """Test that _check_internet returns True when socket connection succeeds."""
    with patch("socket.socket.connect") as mock_connect:
        mock_connect.return_value = None
        result = graphql_client._check_internet()
        if result is not True:
            raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_check_internet_failure() -> None:
    """Test that _check_internet returns False on socket connection failure."""
    with patch("socket.socket.connect", side_effect=OSError):
        result = graphql_client._check_internet()
        if result is not False:
            raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_get_dot_config_file_name() -> None:
    """Test that the returned path includes the given filename."""
    result = graphql_client._get_dot_config_file_name("testfile")
    if not isinstance(result, Path) or not str(result).endswith("testfile"):
        raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_write_token_and_get_token(
    tmp_path: Path,
    dummy_token: str,
    dummy_decoded: dict[str, str],
) -> None:
    """Test writing a token to file and retrieving it again.

    Args:
        tmp_path (Path): Temporary path fixture from pytest.
        dummy_token (str): Dummy token to write.
        dummy_decoded (dict): Corresponding decoded token.

    """
    file_path = tmp_path.joinpath("config.json")

    with (
        patch("graphql_client._get_dot_config_file_name", return_value=file_path),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        graphql_client._write_token_to_file(dummy_token, "config.json")

        with file_path.open() as f:
            data = json.load(f)
            if "prod" not in data["tokens"]:
                raise GraphqlClientTestError(TESTERRORMESSAGE)

        result = graphql_client._get_token_from_file("prod", "config.json")
        if result != dummy_token:
            raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_get_token_from_file_missing_file() -> None:
    """Test that FileNotFoundError is raised when the token file does not exist."""
    with patch(
        "graphql_client._get_dot_config_file_name",
        return_value=Path("nonexistent.json"),
    ):
        try:
            graphql_client._get_token_from_file("prod", "nonexistent.json")
        except FileNotFoundError:
            return
        raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_get_token_from_file_missing_db(tmp_path: Path) -> None:
    """Test that DatabaseChoiceError is raised for missing database entry."""
    file_path = tmp_path.joinpath("config.json")
    file_path.write_text(json.dumps({"tokens": {"test": {"token": "123"}}}))

    with patch("graphql_client._get_dot_config_file_name", return_value=file_path):
        try:
            graphql_client._get_token_from_file("prod", "config.json")
        except graphql_client.DatabaseChoiceError:
            return
        raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_token_get_server_valid_flow(
    tmp_path: Path,
    dummy_token: str,
    dummy_decoded: dict[str, str],
) -> None:
    """Test a complete valid flow through _token_get_server."""
    with (
        patch("jwt.decode", return_value=dummy_decoded),
        patch("webbrowser.open"),
        patch("graphql_client._check_internet", return_value=True),
        patch(
            "graphql_client._get_dot_config_file_name",
            return_value=tmp_path.joinpath("config.json"),
        ),
        patch("graphql_client.Queue") as queue_patch,
        patch("graphql_client.make_server") as server_patch,
        patch("threading.Thread") as thread_patch,
    ):
        q = Queue()
        q.put(dummy_token)
        queue_patch.return_value = q

        server_patch.return_value = MagicMock()
        thread_patch.return_value = MagicMock()

        result = graphql_client._token_get_server("prod", "captor.se", "token.json")
        if result != dummy_token:
            raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_token_get_server_invalid_db() -> None:
    """Test that DatabaseChoiceError is raised for an invalid database input."""
    try:
        graphql_client._token_get_server("invalid", "base", "file")
    except graphql_client.DatabaseChoiceError:
        return
    raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_token_get_server_no_internet() -> None:
    """Test that NoInternetError is raised when no internet connection is detected."""
    with patch("graphql_client._check_internet", return_value=False):
        try:
            graphql_client._token_get_server("prod", "base", "file")
        except graphql_client.NoInternetError:
            return
        raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_browser_get_token_from_file(dummy_token: str) -> None:
    """Test that _browser_get_token returns a token from file successfully."""
    with (
        patch("graphql_client._get_token_from_file", return_value=dummy_token),
        patch(
            "requests.get",
            return_value=MagicMock(status_code=200, raise_for_status=lambda: None),
        ),
    ):
        result = graphql_client._browser_get_token("prod", "captor.se", ".captor")
        if result != dummy_token:
            raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_browser_get_token_fallback_success(dummy_token: str) -> None:
    """Test fallback to _token_get_server when token file is missing."""
    with (
        patch("graphql_client._get_token_from_file", side_effect=FileNotFoundError),
        patch("graphql_client._token_get_server", return_value=dummy_token),
        patch(
            "requests.get",
            return_value=MagicMock(status_code=200, raise_for_status=lambda: None),
        ),
    ):
        result = graphql_client._browser_get_token("prod", "captor.se", ".captor")
        if result != dummy_token:
            raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_browser_get_token_refresh_on_http_error(dummy_token: str) -> None:
    """Test that token is refreshed from server when HTTPError is raised."""
    with (
        patch("graphql_client._get_token_from_file", return_value=dummy_token),
        patch("requests.get", side_effect=requests.HTTPError("bad token")),
        patch("graphql_client._token_get_server", return_value="new_token"),
    ):
        result = graphql_client._browser_get_token("prod", "captor.se", ".captor")
        if result != "new_token":
            raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_browser_get_token_connection_error(dummy_token: str) -> None:
    """Test that NoInternetError is raised on connection failure."""
    with (
        patch("graphql_client._get_token_from_file", return_value=dummy_token),
        patch("requests.get", side_effect=requests.ConnectionError),
    ):
        try:
            graphql_client._browser_get_token("prod", "captor.se", ".captor")
        except graphql_client.NoInternetError:
            return
        raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_external_graphql_client_success(
    dummy_token: str,
    dummy_decoded: dict[str, str],
) -> None:
    """Test successful creation of GraphQLClient instance."""
    with (
        patch("graphql_client._browser_get_token", return_value=dummy_token),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        client = graphql_client.GraphQLClient("prod", "captor.se")
        if client.database != "prod" or "graphql" not in client.url:
            raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_external_graphql_client_invalid_db() -> None:
    """Test that GraphQLClient raises DatabaseChoiceError on invalid database input."""
    try:
        graphql_client.GraphQLClient("invalid", "captor.se")
    except graphql_client.DatabaseChoiceError:
        return
    raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_external_graphql_query_success(
    dummy_token: str,
    dummy_decoded: dict[str, str],
) -> None:
    """Test successful execution of a GraphQL query."""
    with (
        patch("graphql_client._browser_get_token", return_value=dummy_token),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        client = graphql_client.GraphQLClient("prod", "captor.se")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"result": "ok"}}
        mock_response.raise_for_status = lambda: None

        with patch("requests.post", return_value=mock_response):
            data, errors = client.query("query { test }")
            if data != {"result": "ok"} or errors is not None:
                raise GraphqlClientTestError(TESTERRORMESSAGE)


def test_external_graphql_query_http_error(
    dummy_token: str,
    dummy_decoded: dict[str, str],
) -> None:
    """Test that query method returns error string on HTTPError."""
    with (
        patch("graphql_client._browser_get_token", return_value=dummy_token),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        client = graphql_client.GraphQLClient("prod", "captor.se")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("boom")

        with patch("requests.post", return_value=mock_response):
            data, errors = client.query("query { test }")
            if data is not None or "boom" not in errors:
                raise GraphqlClientTestError(TESTERRORMESSAGE)
