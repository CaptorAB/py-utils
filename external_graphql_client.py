import json
import socket
import threading
import webbrowser
from pathlib import Path
from queue import Queue

import jwt as pyjwt
import requests
from werkzeug.serving import make_server
from werkzeug.wrappers import Request, Response


class DatabaseChoiceException(Exception):
    pass


class NoInternetException(Exception):
    pass


def check_internet(host: str = "8.8.8.8", port: int = 53, timeout: int = 3) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
    except OSError:
        return False
    else:
        return True


def get_dot_config_file_name(filename: str) -> Path:
    return Path.home().joinpath(filename)


def write_token_to_file(jwt_token: str, filename: str) -> None:
    dot_config_file_name = get_dot_config_file_name(filename=filename)

    data = pyjwt.decode(jwt=jwt_token, options={"verify_signature": False})
    db = data["aud"]
    if dot_config_file_name.exists():
        with dot_config_file_name.open(mode="r", encoding="utf-8") as file_handle:
            local_token = json.load(fp=file_handle)
        local_token["tokens"][db] = {"token": jwt_token, "decoded": data}
    else:
        local_token = {"tokens": {db: {"token": jwt_token, "decoded": data}}}

    with dot_config_file_name.open(mode="w", encoding="utf-8") as file_handle:
        # noinspection PyTypeChecker
        json.dump(obj=local_token, fp=file_handle, indent=2, sort_keys=True)

    if not dot_config_file_name.exists():
        msg = "Writing token to file failed."
        raise FileNotFoundError(msg)

    print(f"Wrote token to file: {dot_config_file_name}.")


def get_token_from_file(db: str, filename: str) -> str:
    dot_config_file_name = get_dot_config_file_name(filename=filename)

    if not dot_config_file_name.exists():
        msg = f"File '{dot_config_file_name}' with token not found"
        raise FileNotFoundError(msg)

    with dot_config_file_name.open(mode="r", encoding="utf-8") as file_handle:
        dot_config = json.load(fp=file_handle)

    try:
        token = dot_config["tokens"][db]["token"]
    except KeyError as exc:
        msg = "Can only handle database equal to 'prod' or 'test'."
        raise DatabaseChoiceException(msg) from exc

    print("get_token_from_file()")
    return token


def token_get_server(db: str, base_url: str, filename: str, port: int = 5678) -> str:
    print("token_get_server()")

    if db == "prod":
        url_str = ""
    elif db == "test":
        url_str = "test"
    else:
        msg = "Can only handle database equal to 'prod' or 'test'."
        raise DatabaseChoiceException(msg)

    if check_internet():
        webbrowser.open(
            url=(
                f"https://{url_str}portal.{base_url}/token?"
                f"redirect_uri=http://localhost:{port}/token"
            ),
            new=2,
        )
    else:
        msg = "No internet connection."
        raise NoInternetException(msg)

    @Request.application
    def app(request: Request) -> Response:
        q.put(request.args["api_key"])
        return Response(
            response=""" <!DOCTYPE html>
                         <html lang="en-US">
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

    q = Queue()
    s = make_server(host="localhost", port=port, app=app)
    t = threading.Thread(target=s.serve_forever)
    t.start()
    token = q.get(block=True)
    write_token_to_file(jwt_token=token, filename=filename)
    s.shutdown()
    t.join()

    return token


def browser_get_token(db: str, base_url: str, filename: str, timeout: int = 10) -> str:
    try:
        token = get_token_from_file(db=db, filename=filename)
    except (FileNotFoundError, DatabaseChoiceException) as exc:
        print(f"Getting token from file failed: {exc}")
        token = token_get_server(db=db, base_url=base_url, filename=filename)

    headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}

    try:
        response = requests.get(
            url=f"https://auth.{base_url}/ping", headers=headers, timeout=timeout
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        print(f"Token authorization failed. HTTP error occurred: {exc}")
        token = token_get_server(db=db, base_url=base_url, filename=filename)
    except requests.ConnectionError as excc:
        msg = "No internet connection."
        raise NoInternetException(msg) from excc

    return token


class ExternalGraphQLClient:
    def __init__(self, database: str = "prod", base_url: str = "captor.se") -> None:
        filename = f".{base_url.split(sep='.')[0]}"
        self.token = browser_get_token(
            db=database, base_url=base_url, filename=filename
        )

        decoded_token = pyjwt.decode(
            jwt=self.token, options={"verify_signature": False}
        )
        print("token.unique_name: ", decoded_token["unique_name"])

        self.database = database

        if self.database == "prod":
            url_str = ""
        elif self.database == "test":
            url_str = "test"
        else:
            msg = "Can only handle database prod or test."
            raise DatabaseChoiceException(msg)

        self.url = f"https://{url_str}api.{base_url}/graphql"

    def query(
        self,
        gql_query: str,
        variables: dict | None = None,
        timeout: int = 10,
        *,
        verify: bool = True,
    ):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "accept-encoding": "gzip",
        }
        data = {"query": gql_query}

        if variables:
            data["variables"] = variables

        try:
            response = requests.post(
                url=self.url,
                json=data,
                headers=headers,
                verify=verify,
                timeout=timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            return None, str(exc)

        response_data = response.json()

        data = response_data.get("data", None)
        errors = response_data.get("errors", None)

        return data, errors
