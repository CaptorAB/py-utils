<a href="https://captor.se/"><img src="https://sales.captor.se/captor_logo_sv_1600_icketransparent.png" alt="Captor Fund Management AB" width="81" height="100" align="left" float="right"/></a><br/>

<br><br>



# Captor Python utilities

![Platform](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-blue)
![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![GitHub Action Test Suite](https://github.com/CaptorAB/py-utils/actions/workflows/tests.yml/badge.svg)](https://github.com/CaptorAB/py-utils/actions/workflows/tests.yml)
[![Coverage](https://cdn.jsdelivr.net/gh/CaptorAB/py-utils@master/coverage.svg)](https://github.com/CaptorAB/py-utils/actions/workflows/tests.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://beta.ruff.rs/docs/)
[![GitHub License](https://img.shields.io/github/license/CaptorAB/py-utils)](https://github.com/CaptorAB/py-utils/blob/master/LICENSE)

This project assumes that you will be able to install a compatible version and install the dependencies listed in the requirements.txt file.

## GraphQL

### [graphql_client.py](https://github.com/CaptorAB/py-utils/blob/master/graphql_client.py)

An API client necessary to access the [Captor Graphql API](https://api.captor.se/graphql). It requires that you are a registered user that can be authenticated.

### [use_graphql.py](https://github.com/CaptorAB/py-utils/blob/master/use_graphql.py)

Contains a simple example of how to access data through the Captor Graphql API.

## Open API

### [tpt_read_to_xlsx.py](https://github.com/CaptorAB/py-utils/blob/master/tpt_read_to_xlsx.py)

Contains examples of how to extract TPT report data from the [Captor open API](https://api.captor.se/public/api/).

## Portfolio tools from [openseries](https://github.com/CaptorAB/openseries)

### [portfoliotool.py](https://github.com/CaptorAB/py-utils/blob/master/portfoliotool.py)

Contains an example of how to extract timeseries data of NAV prices for Captor funds and use the openseries package to calculate various key metrics.

### [portfoliosimulation.py](https://github.com/CaptorAB/py-utils/blob/master/portfoliosimulation.py)

Contains an example on how to use openseries to simulate portfolios.

## Attribution analysis

### [attribution.py](https://github.com/CaptorAB/py-utils/blob/master/attribution.py)

Contains functions that fetch data on Captor funds to be used to do grouped instruments performance attribution. 
The function that performs the calculation contains three different methods to link cumulative returns. 
Use 'simple' for shorter time periods and one of the other methods for longer periods.
The output is visualized in a Plotly area diagramme. 
Below are examples of how to use. The files can be amended for other funds as needed.

### [aster_short.py](https://github.com/CaptorAB/py-utils/blob/master/aster_short.py)

Utilises the attribution.py module to do performance analysis on the fund Captor Aster Global Credit Short-Term.

### [iris_bond.py](https://github.com/CaptorAB/py-utils/blob/master/iris_bond.py)

Utilises the attribution.py module to do performance analysis on the fund Captor Iris Bond.
