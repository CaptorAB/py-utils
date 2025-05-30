"""Example code for how to fetch data from the Captor Graphql API."""

from pprint import pformat

from graphql_client import GraphqlClient, GraphqlError

if __name__ == "__main__":
    gql = GraphqlClient()

    my_query = """ query parties($nameIn: [String!]) {
                     parties(filter: {nameIn: $nameIn}) {
                       longName
                       legalEntityIdentifier
                     }
                   } """

    my_variables = {"nameIn": ["Captor Iris Bond"]}

    data, error = gql.query(query_string=my_query, variables=my_variables)

    if error:
        raise GraphqlError(str(error))

    print("\n", pformat(data))  # noqa: T201
