import json
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest
import requests

import graphql_client


@pytest.fixture
def dummy_token():
    return "dummy.jwt.token"


@pytest.fixture
def dummy_decoded():
    return {"aud": "prod", "unique_name": "tester"}


def test_check_internet_success():
    with patch("socket.socket.connect") as mock_connect:
        mock_connect.return_value = None
        result = graphql_client.check_internet()
        if result is not True:
            raise ValueError("code not working as intended")


def test_check_internet_failure():
    with patch("socket.socket.connect", side_effect=OSError):
        result = graphql_client.check_internet()
        if result is not False:
            raise ValueError("code not working as intended")


def test_get_dot_config_file_name():
    result = graphql_client.get_dot_config_file_name("testfile")
    if not isinstance(result, Path) or not str(result).endswith("testfile"):
        raise ValueError("code not working as intended")


def test_write_token_and_get_token(tmp_path, dummy_token, dummy_decoded):
    file_path = tmp_path / "config.json"

    with (
        patch("graphql_client.get_dot_config_file_name", return_value=file_path),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        graphql_client.write_token_to_file(dummy_token, "config.json")

        with file_path.open() as f:
            data = json.load(f)
            if "prod" not in data["tokens"]:
                raise ValueError("code not working as intended")

        result = graphql_client.get_token_from_file("prod", "config.json")
        if result != dummy_token:
            raise ValueError("code not working as intended")


def test_get_token_from_file_missing_file():
    with patch(
        "graphql_client.get_dot_config_file_name",
        return_value=Path("nonexistent.json"),
    ):
        try:
            graphql_client.get_token_from_file("prod", "nonexistent.json")
        except FileNotFoundError:
            return
        raise ValueError("code not working as intended")


def test_get_token_from_file_missing_db(tmp_path):
    file_path = tmp_path / "config.json"
    file_path.write_text(json.dumps({"tokens": {"test": {"token": "123"}}}))

    with patch("graphql_client.get_dot_config_file_name", return_value=file_path):
        try:
            graphql_client.get_token_from_file("prod", "config.json")
        except graphql_client.DatabaseChoiceError:
            return
        raise ValueError("code not working as intended")


def test_token_get_server_valid_flow(tmp_path, dummy_token, dummy_decoded):
    with (
        patch("jwt.decode", return_value=dummy_decoded),
        patch("webbrowser.open"),
        patch("graphql_client.check_internet", return_value=True),
        patch(
            "graphql_client.get_dot_config_file_name",
            return_value=tmp_path / "token.json",
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

        result = graphql_client.token_get_server("prod", "captor.se", "token.json")
        if result != dummy_token:
            raise ValueError("code not working as intended")


def test_token_get_server_invalid_db():
    try:
        graphql_client.token_get_server("invalid", "base", "file")
    except graphql_client.DatabaseChoiceError:
        return
    raise ValueError("code not working as intended")


def test_token_get_server_no_internet(tmp_path):
    with patch("graphql_client.check_internet", return_value=False):
        try:
            graphql_client.token_get_server("prod", "base", "file")
        except graphql_client.NoInternetError:
            return
        raise ValueError("code not working as intended")


def test_browser_get_token_from_file(dummy_token):
    with (
        patch("graphql_client.get_token_from_file", return_value=dummy_token),
        patch(
            "requests.get",
            return_value=MagicMock(status_code=200, raise_for_status=lambda: None),
        ),
    ):
        result = graphql_client.browser_get_token("prod", "captor.se", ".captor")
        if result != dummy_token:
            raise ValueError("code not working as intended")


def test_browser_get_token_fallback_success(dummy_token):
    with (
        patch("graphql_client.get_token_from_file", side_effect=FileNotFoundError),
        patch("graphql_client.token_get_server", return_value=dummy_token),
        patch(
            "requests.get",
            return_value=MagicMock(status_code=200, raise_for_status=lambda: None),
        ),
    ):
        result = graphql_client.browser_get_token("prod", "captor.se", ".captor")
        if result != dummy_token:
            raise ValueError("code not working as intended")


def test_browser_get_token_refresh_on_http_error(dummy_token):
    with (
        patch("graphql_client.get_token_from_file", return_value=dummy_token),
        patch("requests.get", side_effect=requests.HTTPError("bad token")),
        patch("graphql_client.token_get_server", return_value="new_token"),
    ):
        result = graphql_client.browser_get_token("prod", "captor.se", ".captor")
        if result != "new_token":
            raise ValueError("code not working as intended")


def test_browser_get_token_connection_error(dummy_token):
    with (
        patch("graphql_client.get_token_from_file", return_value=dummy_token),
        patch("requests.get", side_effect=requests.ConnectionError),
    ):
        try:
            graphql_client.browser_get_token("prod", "captor.se", ".captor")
        except graphql_client.NoInternetError:
            return
        raise ValueError("code not working as intended")


def test_external_graphql_client_success(dummy_token, dummy_decoded):
    with (
        patch("graphql_client.browser_get_token", return_value=dummy_token),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        client = graphql_client.ExternalGraphQLClient("prod", "captor.se")
        if client.database != "prod" or "graphql" not in client.url:
            raise ValueError("code not working as intended")


def test_external_graphql_client_invalid_db():
    try:
        graphql_client.ExternalGraphQLClient("invalid", "captor.se")
    except graphql_client.DatabaseChoiceError:
        return
    raise ValueError("code not working as intended")


def test_external_graphql_query_success(dummy_token, dummy_decoded):
    with (
        patch("graphql_client.browser_get_token", return_value=dummy_token),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        client = graphql_client.ExternalGraphQLClient("prod", "captor.se")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"result": "ok"}}
        mock_response.raise_for_status = lambda: None

        with patch("requests.post", return_value=mock_response):
            data, errors = client.query("query { test }")
            if data != {"result": "ok"} or errors is not None:
                raise ValueError("code not working as intended")


def test_external_graphql_query_http_error(dummy_token, dummy_decoded):
    with (
        patch("graphql_client.browser_get_token", return_value=dummy_token),
        patch("jwt.decode", return_value=dummy_decoded),
    ):
        client = graphql_client.ExternalGraphQLClient("prod", "captor.se")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("boom")

        with patch("requests.post", return_value=mock_response):
            data, errors = client.query("query { test }")
            if data is not None or "boom" not in errors:
                raise ValueError("code not working as intended")
