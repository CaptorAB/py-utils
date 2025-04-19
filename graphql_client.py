"""Module defining the GraphqlClient class and its help functions."""

import json
import socket
import threading
import webbrowser
from logging import WARNING, basicConfig, getLogger
from pathlib import Path
from queue import Queue

import jwt as pyjwt
import requests
from werkzeug.serving import make_server
from werkzeug.wrappers import Request, Response

__all__ = ["GraphqlClient", "GraphqlError", "get_token"]

basicConfig(level=WARNING)
logger = getLogger(__name__)


class GraphqlError(Exception):
    """Raised if the Graphql query returns any error(s)."""


class DatabaseChoiceError(Exception):
    """Raised when an invalid database choice is provided."""

    def __init__(
        self, message: str = "Can only handle database prod or test."
    ) -> None:
        """Initialize the DatabaseChoiceError.

        Args:
            message (str): Exception message.
                Defaults to 'Can only handle database prod or test.'.

        """
        self.message = message
        super().__init__(message)


class NoInternetError(Exception):
    """Raised when there is no internet connectivity."""

    def __init__(self, message: str = "No internet connection.") -> None:
        """Initialize the NoInternetError.

        Args:
            message (str): Exception message.
                Defaults to 'No internet connection.'.

        """
        self.message = message
        super().__init__(message)


def check_internet(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    """Check if there is an active internet connection.

    Args:
        host (str): Host to ping. Defaults to "8.8.8.8".
        port (int): Port to use. Defaults to 53.
        timeout (int): Timeout in seconds. Defaults to 3.

    Returns:
        bool: True if internet is available, False otherwise.

    """
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
    except OSError:
        return False
    else:
        return True


def get_dot_config_file_name(filename: str) -> Path:
    """Get the full path to the config file in the user's home directory.

    Args:
        filename (str): Filename to append to home directory.

    Returns:
        Path: Path object to the config file.

    """
    return Path.home() / filename


def write_token_to_file(jwt_token: str, filename: str) -> None:
    """Write the decoded JWT token to a file.

    Args:
        jwt_token (str): JWT token string.
        filename (str): Target file name to write the token to.

    Raises:
        FileNotFoundError: If the file could not be written.

    """
    dot_config_file_name = get_dot_config_file_name(filename=filename)

    data = pyjwt.decode(jwt=jwt_token, options={"verify_signature": False})
    database = data["aud"]
    if dot_config_file_name.exists():
        with dot_config_file_name.open(mode="r", encoding="utf-8") as file_handle:
            local_token = json.load(fp=file_handle)
        local_token["tokens"][database] = {"token": jwt_token, "decoded": data}
    else:
        local_token = {"tokens": {database: {"token": jwt_token, "decoded": data}}}

    with dot_config_file_name.open(mode="w", encoding="utf-8") as file_handle:
        # noinspection PyTypeChecker
        json.dump(obj=local_token, fp=file_handle, indent=2, sort_keys=True)

    if not dot_config_file_name.exists():
        msg = "Writing token to file failed."
        raise FileNotFoundError(msg)

    logger_message = f"Wrote token to file: {dot_config_file_name}."
    logger.info(logger_message)


def get_token_from_file(database: str, filename: str) -> str:
    """Read the token from a local file.

    Args:
        database (str): Database identifier.
        filename (str): File containing the token.

    Returns:
        str: Token string.

    Raises:
        FileNotFoundError: If the token file doesn't exist.
        DatabaseChoiceError: If the database is not supported.

    """
    dot_config_file_name = get_dot_config_file_name(filename=filename)

    if not dot_config_file_name.exists():
        msg = f"File '{dot_config_file_name}' with token not found"
        raise FileNotFoundError(msg)

    with dot_config_file_name.open(mode="r", encoding="utf-8") as file_handle:
        dot_config = json.load(fp=file_handle)

    try:
        token = dot_config["tokens"][database]["token"]
    except KeyError as exc:
        raise DatabaseChoiceError from exc

    logger_message = "get_token_from_file()"
    logger.info(logger_message)

    return token


def token_get_server(
    database: str, base_url: str, filename: str, port: int = 5678
) -> str:
    """Start a temporary server to retrieve a token via browser.

    Args:
        database (str): Database name.
        base_url (str): Base URL for the authentication service.
        filename (str): File to store the token.
        port (int): Local server port. Defaults to 5678.

    Returns:
        str: Retrieved token.

    Raises:
        DatabaseChoiceError: If an unsupported database is given.
        NoInternetError: If no internet connection is available.

    """
    logger_message = "token_get_server()"
    logger.info(logger_message)

    if database == "prod":
        url_str = ""
    elif database == "test":
        url_str = "test"
    else:
        raise DatabaseChoiceError

    if check_internet():
        webbrowser.open(
            url=(
                f"https://{url_str}portal.{base_url}/token?"
                f"redirect_uri=http://localhost:{port}/token"
            ),
            new=2,
        )
    else:
        raise NoInternetError

    @Request.application
    def app(request: Request) -> Response:
        """Local HTTP handler to receive the API key from the browser."""
        queue.put(item=request.args["api_key"])
        return Response(
            response=""" <!DOCTYPE html>
                         <html lang=\"en-US\">
                             <head>
                                 <script>
                                     setTimeout(function () {
                                         window.close();
                                     }, 2000);
                                 </script>
                             </head>
                             <body>
                                 <p>Writing token to local machine</p>
                             </body>
                         </html> """,
            status=200,
            content_type="text/html; charset=UTF-8",
        )

    queue = Queue()
    server = make_server(host="localhost", port=port, app=app)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    token = queue.get(block=True)
    write_token_to_file(jwt_token=token, filename=filename)
    server.shutdown()
    thread.join()

    return token


def browser_get_token(
    database: str,
    base_url: str,
    filename: str,
    timeout: int = 10,
) -> str:
    """Get a valid token from file or trigger browser-based login flow.

    Args:
        database (str): Database name.
        base_url (str): Base URL for the authentication service.
        filename (str): Token storage file name.
        timeout (int): Timeout for token verification request.

    Returns:
        str: Validated token.

    Raises:
        NoInternetError: If internet is not available.

    """
    try:
        token = get_token_from_file(database=database, filename=filename)
    except (FileNotFoundError, DatabaseChoiceError) as exc:
        logger_message = f"Getting token from file failed: {exc}"
        logger.warning(logger_message)
        token = token_get_server(
            database=database, base_url=base_url, filename=filename
        )

    headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}

    try:
        response = requests.get(
            url=f"https://auth.{base_url}/ping",
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger_message = f"Token authorization failed. HTTP error occurred: {exc}"
        logger.warning(logger_message)
        token = token_get_server(
            database=database, base_url=base_url, filename=filename
        )
    except requests.ConnectionError as excc:
        raise NoInternetError from excc

    return token


class GraphqlClient:
    """Captor Graphql Client.

    Class used to authenticate a user and allow it to fetch data from the
    Captor Graphql API.
    """

    def __init__(self, database: str = "prod", base_url: str = "captor.se") -> None:
        """Initialize the GraphQLClient.

        Args:
            database (str): Database name, 'prod' or 'test'. Defaults to 'prod'.
            base_url (str): Base domain name. Defaults to 'captor.se'.

        Raises:
            DatabaseChoiceError: If database is not 'prod' or 'test'.

        """
        filename = f".{base_url.split(sep='.')[0]}"
        self.token = browser_get_token(
            database=database,
            base_url=base_url,
            filename=filename,
        )

        decoded_token = pyjwt.decode(
            jwt=self.token,
            options={"verify_signature": False},
        )
        logger_message = f"token.unique_name: {decoded_token['unique_name']}"
        logger.info(logger_message)

        self.database = database

        if self.database == "prod":
            url_str = ""
        elif self.database == "test":
            url_str = "test"
        else:
            raise DatabaseChoiceError

        self.url = f"https://{url_str}api.{base_url}/graphql"

    def query(
        self,
        query_string: str,
        variables: dict | None = None,
        timeout: int = 10,
        *,
        verify: bool = True,
    ) -> tuple[dict | list | bool | None, dict | list | bool | str | None]:
        """Execute a GraphQL query.

        Args:
            query_string (str): GraphQL query string.
            variables (dict | None): Query variables. Defaults to None.
            timeout (int): Request timeout in seconds. Defaults to 10.
            verify (bool): Whether to verify SSL cert. Defaults to True.

        Returns:
            tuple: Tuple of (data, errors) from the query result.

        Raises:
            NoInternetError: If internet is not available.

        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "accept-encoding": "gzip",
        }
        json_data = {"query": query_string}

        if variables:
            json_data["variables"] = variables

        try:
            response = requests.post(
                url=self.url,
                json=json_data,
                headers=headers,
                verify=verify,
                timeout=timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger_message = f"Query execution failed. HTTP error occurred: {exc}"
            logger.warning(logger_message)
            return None, str(exc)
        except requests.ConnectionError as excc:
            raise NoInternetError from excc

        response_data = response.json()

        data = response_data.get("data", None)
        errors = response_data.get("errors", None)

        return data, errors


def get_token(
    database: str,
    username: str,
    password: str,
    url: str = "https://auth.captor.se/token",
    timeout: int = 10,
) -> str | None:
    """Get a token with a username and password as authentication.

    Args:
        database (str): Database identifier.
        username (str): Username.
        password (str): Password.
        url (str): Web site address / url.
            Defaults to 'https://auth.captor.se/token'
        timeout (int): Timeout in seconds. Defaults to 10.

    Returns:
        str: Token string.

    Raises:
        DatabaseChoiceError: If the database is not supported.
        NoInternetError: If internet is not available.

    """
    if database not in ["prod", "test"]:
        raise DatabaseChoiceError

    try:
        response = requests.post(
            url=url,
            data={
                "client_id": database,
                "username": username,
                "password": password,
                "grant_type": "password",
            },
            timeout=timeout,
        )
        result: str = response.json().get("access_token", None)
    except requests.HTTPError as exc:
        logger_message = (
            f"POST https://auth.captor.se/token failed. HTTP error occurred: {exc}"
        )
        logger.warning(logger_message)
        return None
    except requests.ConnectionError as excc:
        raise NoInternetError from excc
    else:
        return result
