from pprint import pformat
from external_graphql_client import ExternalGraphQLClient


class GraphqlException(Exception):
    pass


if __name__ == "__main__":
    gql = ExternalGraphQLClient()
    my_query = """ query parties($nameIn: [String!]) { 
                     parties(filter: {nameIn: $nameIn}) { 
                       longName
                       legalEntityIdentifier
                     }
                   } """
    my_variables = {"nameIn": "Captor Iris Bond"}

    output, outputerrors = gql.query(gql_query=my_query, variables=my_variables)

    if outputerrors:
        raise GraphqlException(str(outputerrors))

    print(pformat(output))
