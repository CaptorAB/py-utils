# app.py

import pandas as pd
import streamlit as st

from gqlclient_streamlit import StreamlitGraphqlClient

st.set_page_config(page_title="Captor GraphQL Explorer", layout="wide")
st.title("🔍 Captor GraphQL Explorer")

# Initialize client (and load token from session or file)
if "gql" not in st.session_state:
    st.session_state.gql = StreamlitGraphqlClient()
gql = st.session_state.gql

# Check for token
token = st.session_state.get("token")

# --- Login Flow ---
if not token:
    # Inject a <meta> tag to reload the page every 2s until we pick up the token
    st.markdown("<meta http-equiv='refresh' content='2'>", unsafe_allow_html=True)
    st.warning("🔐 You must log in to access Captor’s GraphQL API.")

    if st.button("🔑 Log in via browser"):
        with st.spinner(
            "Opening Captor portal... complete it there, then return here."
        ):
            try:
                gql.login()
                # Flag for one‐time banner
                st.session_state["just_authenticated"] = True
            except Exception as e:
                st.error(f"Login failed: {e}")

    # Stop here until we have a token
    st.stop()

# --- One‐time Success Banner ---
if st.session_state.pop("just_authenticated", False):
    st.success("👍 You are authenticated!")

# --- Authenticated UI ---
st.success("You’re now logged in and can run queries below.")

st.subheader("Query Parties")
party_name = st.text_input("Party name", "Captor Iris Bond")

if st.button("Fetch Parties"):
    with st.spinner("⏳ Fetching data…"):
        query = """
        query parties($nameIn: [String!]) {
          parties(filter: {nameIn: $nameIn}) {
            longName
            legalEntityIdentifier
          }
        }
        """
        data, error = gql.query(query_string=query, variables={"nameIn": [party_name]})

    if error:
        st.error(f"❗ GraphQL Error: {error}")
    elif not data or not data.get("parties"):
        st.warning("⚠️ No parties found for that name.")
    else:
        st.balloons()
        df = pd.DataFrame(data["parties"])
        st.dataframe(df, use_container_width=True)
