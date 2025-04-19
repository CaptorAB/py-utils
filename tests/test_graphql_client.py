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

    def __init__(self, message: str = "Code not working as intended.") -> None:
        """Initialize the GraphqlClientTestError.

        Args:
            message (str): Exception message.
                Defaults to 'Code not working as intended.'.

        """
        self.message = message
        super().__init__(message)


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
        result = graphql_client.check_internet()
        if result is not True:
            raise GraphqlClientTestError


def test_check_internet_failure() -> None:
    """Test that _check_internet returns False on socket connection failure."""
    with patch("socket.socket.connect", side_effect=OSError):
        result = graphql_client.check_internet()
        if result is not False:
            raise GraphqlClientTestError


def test_get_dot_config_file_name() -> None:
    """Test that the returned path includes the given filename."""
    result = graphql_client.get_dot_config_file_name(filename="testfile")
    if not isinstance(result, Path) or not str(result).endswith("testfile"):
        raise GraphqlClientTestError


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
    file_path = tmp_path / "config.json"

    with (
        patch("graphql_client.get_dot_config_file_name", return_value=file_path),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        graphql_client.write_token_to_file(
            jwt_token=dummy_token, filename="config.json"
        )

        with file_path.open() as f:
            data = json.load(f)
            if "prod" not in data["tokens"]:
                raise GraphqlClientTestError

        result = graphql_client.get_token_from_file(
            database="prod", filename="config.json"
        )
        if result != dummy_token:
            raise GraphqlClientTestError


def test_get_token_from_file_missing_file() -> None:
    """Test that FileNotFoundError is raised when the token file does not exist."""
    with patch(
        "graphql_client.get_dot_config_file_name",
        return_value=Path("nonexistent.json"),
    ):
        try:
            graphql_client.get_token_from_file(
                database="prod", filename="nonexistent.json"
            )
        except FileNotFoundError:
            return
        raise GraphqlClientTestError


def test_get_token_from_file_missing_db(tmp_path: Path) -> None:
    """Test that DatabaseChoiceError is raised for missing database entry."""
    file_path = tmp_path / "config.json"
    file_path.write_text(data=json.dumps({"tokens": {"test": {"token": "123"}}}))

    with patch("graphql_client.get_dot_config_file_name", return_value=file_path):
        try:
            graphql_client.get_token_from_file(database="prod", filename="config.json")
        except graphql_client.DatabaseChoiceError:
            return
        raise GraphqlClientTestError


def test_token_get_server_valid_flow(
    tmp_path: Path,
    dummy_token: str,
    dummy_decoded: dict[str, str],
) -> None:
    """Test a complete valid flow through _token_get_server."""
    with (
        patch("jwt.decode", return_value=dummy_decoded),
        patch("webbrowser.open"),
        patch("graphql_client.check_internet", return_value=True),
        patch(
            "graphql_client.get_dot_config_file_name",
            return_value=tmp_path / "config.json",
        ),
        patch("graphql_client.Queue") as queue_patch,
        patch("graphql_client.make_server") as server_patch,
        patch("threading.Thread") as thread_patch,
    ):
        q = Queue()
        q.put(item=dummy_token)
        queue_patch.return_value = q

        server_patch.return_value = MagicMock()
        thread_patch.return_value = MagicMock()

        result = graphql_client.token_get_server(
            database="prod", base_url="captor.se", filename="token.json"
        )
        if result != dummy_token:
            raise GraphqlClientTestError


def test_token_get_server_invalid_db() -> None:
    """Test that DatabaseChoiceError is raised for an invalid database input."""
    try:
        graphql_client.token_get_server(
            database="invalid", base_url="base", filename="file"
        )
    except graphql_client.DatabaseChoiceError:
        return
    raise GraphqlClientTestError


def test_token_get_server_no_internet() -> None:
    """Test that NoInternetError is raised when no internet connection is detected."""
    with patch("graphql_client.check_internet", return_value=False):
        try:
            graphql_client.token_get_server(
                database="prod", base_url="base", filename="file"
            )
        except graphql_client.NoInternetError:
            return
        raise GraphqlClientTestError


def test_browser_get_token_from_file(dummy_token: str) -> None:
    """Test that _browser_get_token returns a token from file successfully."""
    with (
        patch("graphql_client.get_token_from_file", return_value=dummy_token),
        patch(
            "requests.get",
            return_value=MagicMock(status_code=200, raise_for_status=lambda: None),
        ),
    ):
        result = graphql_client.browser_get_token(
            database="prod", base_url="captor.se", filename=".captor"
        )
        if result != dummy_token:
            raise GraphqlClientTestError


def test_browser_get_token_fallback_success(dummy_token: str) -> None:
    """Test fallback to _token_get_server when token file is missing."""
    with (
        patch("graphql_client.get_token_from_file", side_effect=FileNotFoundError),
        patch("graphql_client.token_get_server", return_value=dummy_token),
        patch(
            "requests.get",
            return_value=MagicMock(status_code=200, raise_for_status=lambda: None),
        ),
    ):
        result = graphql_client.browser_get_token(
            database="prod", base_url="captor.se", filename=".captor"
        )
        if result != dummy_token:
            raise GraphqlClientTestError


def test_browser_get_token_refresh_on_http_error(dummy_token: str) -> None:
    """Test that token is refreshed from server when HTTPError is raised."""
    with (
        patch("graphql_client.get_token_from_file", return_value=dummy_token),
        patch("requests.get", side_effect=requests.HTTPError("bad token")),
        patch("graphql_client.token_get_server", return_value="new_token"),
    ):
        result = graphql_client.browser_get_token(
            database="prod", base_url="captor.se", filename=".captor"
        )
        if result != "new_token":
            raise GraphqlClientTestError


def test_browser_get_token_connection_error(dummy_token: str) -> None:
    """Test that NoInternetError is raised on connection failure."""
    with (
        patch("graphql_client.get_token_from_file", return_value=dummy_token),
        patch("requests.get", side_effect=requests.ConnectionError),
    ):
        try:
            graphql_client.browser_get_token(
                database="prod", base_url="captor.se", filename=".captor"
            )
        except graphql_client.NoInternetError:
            return
        raise GraphqlClientTestError


def test_external_graphql_client_success(
    dummy_token: str,
    dummy_decoded: dict[str, str],
) -> None:
    """Test successful creation of GraphQLClient instance."""
    with (
        patch("graphql_client.browser_get_token", return_value=dummy_token),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        client = graphql_client.GraphqlClient(database="prod", base_url="captor.se")
        if client.database != "prod" or "graphql" not in client.url:
            raise GraphqlClientTestError


def test_external_graphql_client_invalid_db() -> None:
    """Test that GraphQLClient raises DatabaseChoiceError on invalid database."""
    try:
        graphql_client.GraphqlClient(database="invalid", base_url="captor.se")
    except graphql_client.DatabaseChoiceError:
        return
    raise GraphqlClientTestError


def test_external_graphql_query_success(
    dummy_token: str,
    dummy_decoded: dict[str, str],
) -> None:
    """Test successful execution of a GraphQL query."""
    with (
        patch("graphql_client.browser_get_token", return_value=dummy_token),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        client = graphql_client.GraphqlClient(database="prod", base_url="captor.se")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"result": "ok"}}
        mock_response.raise_for_status = lambda: None

        with patch("requests.post", return_value=mock_response):
            data, errors = client.query("query { test }")
            if data != {"result": "ok"} or errors is not None:
                raise GraphqlClientTestError


def test_external_graphql_query_http_error(
    dummy_token: str,
    dummy_decoded: dict[str, str],
) -> None:
    """Test that query method returns error string on HTTPError."""
    with (
        patch("graphql_client.browser_get_token", return_value=dummy_token),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        client = graphql_client.GraphqlClient(database="prod", base_url="captor.se")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("boom")

        with patch("requests.post", return_value=mock_response):
            data, errors = client.query("query { test }")
            if data is not None or "boom" not in errors:
                raise GraphqlClientTestError


@pytest.fixture
def dummy_decoded_test() -> dict[str, str]:
    """Fixture that provides a decoded dummy JWT token for the 'test' database.

    Returns:
        dict[str, str]: Decoded token content with 'aud' and 'unique_name'.

    """
    return {"aud": "test", "unique_name": "tester2"}


def test_write_token_to_file_merge_branch(
    tmp_path: Path,
    dummy_decoded_test: dict[str, str],
    dummy_token: str,
) -> None:
    """Test merging behavior in _write_token_to_file when a config file already exists.

    Args:
        tmp_path (Path): Temporary filesystem path fixture.
        dummy_decoded_test (dict[str, str]): Decoded token for 'test'.
        dummy_token (str): Dummy JWT token.

    """
    file_path = tmp_path / "config.json"
    initial = {"tokens": {"test": {"token": "old.token", "decoded": {}}}}
    file_path.write_text(json.dumps(initial))

    with (
        patch("jwt.decode", return_value=dummy_decoded_test),
        patch("graphql_client.get_dot_config_file_name", return_value=file_path),
    ):
        graphql_client.write_token_to_file(
            jwt_token=dummy_token, filename="config.json"
        )

    data = json.loads(file_path.read_text())
    msg = "Merged token for 'test' not updated correctly."

    if "test" not in data["tokens"] or data["tokens"]["test"]["token"] != dummy_token:
        raise GraphqlClientTestError(msg)


def test_query_with_variables_and_verify_false(dummy_token: str) -> None:
    """Test GraphqlClient.query with variables and verify=False.

    Query should return no data or errors for an empty response.

    Args:
        dummy_token (str): Dummy JWT token.

    """
    client = graphql_client.GraphqlClient.__new__(graphql_client.GraphqlClient)
    client.token = dummy_token
    client.url = "http://example.com/graphql"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {}
    mock_resp.raise_for_status = lambda: None

    with patch("requests.post", return_value=mock_resp) as mock_post:
        data, errors = client.query("query String", variables={"a": 1}, verify=False)

    mock_post.assert_called_once_with(
        url=client.url,
        json={"query": "query String", "variables": {"a": 1}},
        headers={"Authorization": f"Bearer {dummy_token}", "accept-encoding": "gzip"},
        verify=False,
        timeout=10,
    )

    msg = "Expected no data or errors for empty response."
    if data is not None or errors is not None:
        raise GraphqlClientTestError(msg)


def test_query_connection_error(dummy_token: str) -> None:
    """Test GraphqlClient.query raises NoInternetError on ConnectionError.

    Args:
        dummy_token (str): Dummy JWT token.

    """
    client = graphql_client.GraphqlClient.__new__(graphql_client.GraphqlClient)
    client.token = dummy_token
    client.url = "http://example.com/graphql"

    msg = "ConnectionError did not raise NoInternetError."
    with patch("requests.post", side_effect=requests.ConnectionError):
        try:
            client.query("query {}")
        except graphql_client.NoInternetError:
            return
        raise GraphqlClientTestError(msg)


def test_query_with_response_errors(dummy_token: str) -> None:
    """Test GraphqlClient.query returns errors when 'errors' in the response JSON.

    Args:
        dummy_token (str): Dummy JWT token.

    """
    client = graphql_client.GraphqlClient.__new__(graphql_client.GraphqlClient)
    client.token = dummy_token
    client.url = "http://example.com/graphql"

    err_json = {"errors": ["error1"]}
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = err_json
    mock_resp.raise_for_status = lambda: None

    with patch("requests.post", return_value=mock_resp):
        data, errors = client.query("q")

    msg = "Errors not returned correctly from query."
    if data is not None or errors != ["error1"]:
        raise GraphqlClientTestError(msg)


def test_graphqlclient_test_db_url(
    dummy_token: str, dummy_decoded: dict[str, str]
) -> None:
    """Test GraphqlClient initialization for 'test' database sets the expected URL.

    Args:
        dummy_token (str): Dummy JWT token.
        dummy_decoded (dict[str, str]): Decoded token for 'prod'.

    """
    with (
        patch("graphql_client.browser_get_token", return_value=dummy_token),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        client = graphql_client.GraphqlClient(database="test", base_url="domain.com")

    expected_url = "https://testapi.domain.com/graphql"

    msg = (
        f"Expected database='test' and url='{expected_url}', "
        f"got database='{client.database}', url='{client.url}'."
    )

    if client.database != "test" or client.url != expected_url:
        raise GraphqlClientTestError(msg)


def test_get_token_success() -> None:
    """Test that get_token returns the 'access_token' field on success response."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"access_token": "tok123"}
    msg = "get_token did not return the expected token."

    with patch("requests.post", return_value=mock_resp):
        result = graphql_client.get_token("prod", "u", "p")
    if result != "tok123":
        raise GraphqlClientTestError(msg)


def test_get_token_no_access_token() -> None:
    """Test that get_token returns None when 'access_token' is missing."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {}
    msg = "get_token should return None if 'access_token' missing."

    with patch("requests.post", return_value=mock_resp):
        result = graphql_client.get_token("test", "u", "p")
    if result is not None:
        raise GraphqlClientTestError(msg)


def test_get_token_http_error() -> None:
    """Test that get_token returns None HTTPError raised by requests.post."""
    msg = "get_token should return None on HTTPError."
    with patch("requests.post", side_effect=requests.HTTPError("bad request")):
        result = graphql_client.get_token("prod", "u", "p")
    if result is not None:
        raise GraphqlClientTestError(msg)


def test_get_token_connection_error() -> None:
    """Test that get_token raises NoInternetError on ConnectionError."""
    msg = "get_token did not raise NoInternetError on ConnectionError."
    with patch("requests.post", side_effect=requests.ConnectionError()):
        try:
            graphql_client.get_token("test", "u", "p")
        except graphql_client.NoInternetError:
            return
        raise GraphqlClientTestError(msg)


def test_get_token_invalid_database() -> None:
    """Test that get_token raises DatabaseChoiceError for invalid database value."""
    msg = "get_token did not raise DatabaseChoiceError for invalid database."
    try:
        graphql_client.get_token("invalid", "u", "p")
    except graphql_client.DatabaseChoiceError:
        return
    raise GraphqlClientTestError(msg)
