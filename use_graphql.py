"""Example code for how to fetch data from the Captor Graphql API."""

from pprint import pformat

from graphql_client import GraphQLClient


class GraphqlError(Exception):
    """Raised if the Graphql query returns any error(s)."""


if __name__ == "__main__":
    gql = GraphQLClient()

    my_query = """ query parties($nameIn: [String!]) {
                     parties(filter: {nameIn: $nameIn}) {
                       longName
                       legalEntityIdentifier
                     }
                   } """

    my_variables = {"nameIn": ["Captor Iris Bond"]}

    output, outputerrors = gql.query(query_string=my_query, variables=my_variables)

    if outputerrors:
        raise GraphqlError(str(outputerrors))

    print("\n", pformat(output))  # noqa: T201
