<a href="https://captor.se/"><img src="https://sales.captor.se/captor_logo_sv_1600_icketransparent.png" alt="Captor Fund Management AB" width="81" height="100" align="left" float="right"/></a><br/>

<br><br>



# Captor Python utilities

## Python version
usee
At Captor we are running python 3.13 at the moment. However, these utilities should run in all versions from 3.10. This project assumes that you will be able to install a compatible version and install the dependencies listed in the requirements.txt file.

### external_graphql_client.py

An API client necessary to access the Captor Graphql API. It requires that your a registered user that can be authenticated.

### use_graphql.py

Contains a simple example of how to access data through the (Captor Graphql API)[https://api.captor.se/graphql].

### tpt_read_to_xlsx.py

Contains examples of how to extract TPT report data from the (Captor open API)[https://api.captor.se/public/api/].

### portfoliotool.py

Contains an example of how to extract timeseries data of NAV prices for Captor funds and use the (openseries)[https://github.com/CaptorAB/openseries] package to calculate various key metrics.

### portfoliosimulation.py

Contains an example on how to use openseries to simulate portfolios.
