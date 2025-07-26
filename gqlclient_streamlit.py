import jwt as pyjwt
import streamlit as st
from requests import post as requests_post, HTTPError, RequestException

from graphql_client import GraphqlError, get_token_from_file, token_get_server


class StreamlitGraphqlClient:
    """Streamlit-adapted GraphQL client that handles OAuth-style login
    via a local callback server, stores the token in session state, and
    persists between reloads by reading/writing a file in the user's home directory.
    """

    def __init__(self, database: str = "prod", base_url: str = "captor.se") -> None:
        """Initialize the StreamlitGraphqlClient.

        Attempts to load an existing token from Streamlit session state
        or from a local file (~/.streamlit_token). If found, stores it in
        session state for persistence.

        Args:
            database: 'prod' or 'test'
            base_url: e.g. 'captor.se'
        """
        self.database = database
        self.base_url = base_url
        prefix = "" if database == "prod" else "test"
        self.url = f"https://{prefix}api.{base_url}/graphql"
        self.tokenfile = ".captor_streamlit"

        # Load token from session or fallback to file
        token = st.session_state.get("token")
        if not token:
            try:
                token = get_token_from_file(database=database, filename=self.tokenfile)
                st.session_state["token"] = token
            except Exception:
                token = None

        self.token: str | None = token

    def login(self) -> str:
        """Start the local HTTP callback flow:
        - Opens the Captor portal in the browser.
        - Waits for the callback with api_key.
        - Stores the token in session state and local file.

        Returns:
            The raw token string.
        """
        token = token_get_server(
            database=self.database,
            base_url=self.base_url,
            filename=self.tokenfile,
            port=5678,
        )
        st.session_state["token"] = token
        self.token = token

        # Decode JWT metadata if possible
        try:
            decoded = pyjwt.decode(jwt=token, options={"verify_signature": False})
            st.session_state["decoded_token"] = decoded
        except pyjwt.DecodeError:
            pass

        return token

    def query(
        self,
        query_string: str,
        variables: dict[str, object] | None = None,
        timeout: int = 10,
        verify: bool = True,
    ) -> tuple[dict[str, object] | list | None, str | list | None]:
        """Execute a GraphQL query using the current authentication token.

        Args:
            query_string: The GraphQL query text.
            variables: Variables for the query.
            timeout: Request timeout in seconds.
            verify: Whether to verify SSL certificates.

        Returns:
            A tuple where the first element is the data (dict, list, or None)
            and the second element is any errors (str, list, or None).

        Raises:
            GraphqlError: If no token is set or a network error occurs.
        """
        if not self.token:
            raise GraphqlError("Authentication token is missing. Please log in.")

        headers = {
            "Authorization": f"Bearer {self.token}",
            "accept-encoding": "gzip",
        }
        payload: dict[str, object] = {"query": query_string}
        if variables:
            payload["variables"] = variables

        try:
            resp = requests_post(
                url=self.url,
                json=payload,
                headers=headers,
                verify=verify,
                timeout=timeout,
            )
            resp.raise_for_status()
        except HTTPError as he:
            return None, str(he)
        except RequestException as re:
            raise GraphqlError(f"Network error: {re}")

        result = resp.json()
        return result.get("data"), result.get("errors")
