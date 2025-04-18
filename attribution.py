"""Portfolio Attribution Module.

Provides functionality for:
  a) Fetching performance data from a GraphQL API
  b) Computing daily and cumulative group level return attributions
     with simple, logreturn, or Carino/Menchero linking methods
  c) Rendering an interactive area chart of attributions and total
     portfolio returns using OpenSeries and Plotly

This module defines Pydantic models for validation, helper functions
to query and validate data, and the `compute_grouped_attribution_with_cumulative`
and `attrib_area` routines for analysis and visualization.
"""

import datetime as dt
import math
from pathlib import Path
from typing import Any, Literal

from openseries import OpenFrame, OpenTimeSeries
from pandas import concat
from plotly.graph_objs import Figure
from plotly.offline import plot

from graphql_client import GraphqlClient, GraphqlError


class PortfolioValueZeroError(Exception):
    """Raised if the portfolio value is zero."""


class UnknownCompoundMethodError(Exception):
    """Raised if the compound method is unknown."""


class CannotCompoundReturnError(Exception):
    """Raised if the return cannot be compounded."""


def get_party_name(graphql: GraphqlClient, party_id: str) -> str:
    """Retrieve the long name of a party from the GraphQL API.

    Args:
        graphql: A configured GraphqlClient instance.
        party_id: The GraphQL ID of the party to query.

    Returns:
        The 'longName' field of the party.

    Raises:
        GraphqlError: If the GraphQL API returns an error.

    """
    query = "query party($_id: GraphQLObjectId) { party(_id: $_id) { longName } }"
    variables = {"_id": party_id}
    data, error = graphql.query(query_string=query, variables=variables)

    if error:
        msg = str(error)
        raise GraphqlError(msg)

    return data["party"]["longName"]


def get_performance(
    graphql: GraphqlClient,
    client_id: str,
    start_dt: dt.date | None = None,
    end_dt: dt.date | None = None,
    *,
    look_through: bool = False,
) -> dict:
    """Fetch performance data for a client via GraphQL and validate it.

    Args:
        graphql: A configured GraphqlClient instance.
        client_id: The GraphQL ID of the client/fund.
        start_dt: Optional start date for the performance filter.
        end_dt: Optional end date for the performance filter.
        look_through: Whether to include underlying holdings in performance.

    Returns:
        A dict representing the 'performance2' field from the API response.

    Raises:
        GraphqlError: If the GraphQL API returns an error or the response
            fails Pydantic validation.

    """
    query = """ query performance2(
                  $clientId: GraphQLObjectId!,
                  $startDate: GraphQLDateString,
                  $endDate: GraphQLDateString,
                  $lookThrough: Boolean = false
                ) {
                  performance2(
                    clientId: $clientId,
                    lookThrough: $lookThrough
                    filter: {
                      startDate: $startDate
                      endDate: $endDate
                    }
                  ) {
                    dates
                    series
                    instrumentPerformances {
                      instrument{ modelType currency }
                      values
                      cashFlows
                    }
                  }
                } """

    variables = {
        "clientId": client_id,
        "startDate": start_dt.strftime("%Y-%m-%d") if start_dt else None,
        "endDate": end_dt.strftime("%Y-%m-%d") if end_dt else None,
        "lookThrough": look_through,
    }

    data, error = graphql.query(query_string=query, variables=variables)

    if error:
        msg = str(error)
        raise GraphqlError(msg)

    return data["performance2"]


def compute_grouped_attribution_with_cumulative(
    data: dict[str, Any],
    group_by: str,
    group_values: list[str],
    method: Literal["simple", "logreturn", "carino_menchero"],
) -> tuple[
    dict[str, list[dict[str, str | float]]],
    dict[str, list[dict[str, str | float]]],
    list[dict[str, str | float]],
]:
    """Compute daily and cumulative return attribution by group.

    Args:
        data: A dict containing:
            - "dates": list of date strings (length n_days)
            - "series": list of total portfolio returns (length n_days)
            - "instrumentPerformances": list of dicts each with:
                * "values": list of floats (length n_days)
                * "cashFlows": list of floats (length n_days)
                * "instrument": dict with key matching `group_by`
        group_by: The instrument field to group on (e.g. "currency", "modelType").
        group_values: Specific categories to pull out; all others into "Other".
        method: The cumulative method to apply:
            - "simple": arithmetic running-sum
            - "logreturn": geometric via log returns
            - "carino_menchero": Carino/Menchero linking adjustment

    Returns:
        A tuple of three elements:
        1. daily_series: dict mapping each group name to a list of dicts
           {"date": <date_str>, "value": <daily contribution>}.
        2. cumulative_series: dict mapping each group name to a list of
           dicts {"date": <date_str>, "value": <cumulative contribution>}.
        3. total_series: list of dicts representing the raw portfolio series
           {"date": <date_str>, "value": <series value>}.

    Raises:
        PortfolioValueZeroError: Raised if the portfolio value is zero.

        CannotCompoundReturnError: Raised if the return cannot be compounded.

        UnknownCompoundMethodError: Raised if the return cannot be compounded.

    """
    dates = data.get("dates")
    series = data.get("series")
    performances = data.get("instrumentPerformances")
    n_days = len(dates)
    groups = [*group_values, "Other"]

    # Compute daily contributions
    daily_contribs: dict[str, list[float]] = {grp: [0.0] * n_days for grp in groups}
    for t in range(1, n_days):
        total_prev_value = sum(perf["values"][t - 1] for perf in performances)
        if total_prev_value == 0.0:
            msg = f"Total portfolio value is zero on day index {t - 1}"
            raise PortfolioValueZeroError(msg)
        for perf in performances:
            prev_value = perf["values"][t - 1]
            curr_value = perf["values"][t]
            flow = perf["cashFlows"][t]
            category = perf["instrument"][group_by]
            grp = category if category in group_values else "Other"
            delta = curr_value - prev_value - flow
            daily_contribs[grp][t] += delta / total_prev_value

    # Compute cumulative contributions per method
    cumulative_contribs: dict[str, list[float]] = {
        grp: [0.0] * n_days for grp in groups
    }

    if method == "simple":
        for grp in groups:
            for t in range(1, n_days):
                cumulative_contribs[grp][t] = (
                    cumulative_contribs[grp][t - 1] + daily_contribs[grp][t]
                )

    elif method == "logreturn":
        for grp in groups:
            running_log = 0.0
            for t in range(1, n_days):
                ret = daily_contribs[grp][t]
                if ret <= -1.0:
                    msg = f"Return {ret} at day index {t} cannot be compounded"
                    raise CannotCompoundReturnError(msg)
                running_log += math.log1p(ret)
                cumulative_contribs[grp][t] = math.expm1(running_log)

    elif method == "carino_menchero":
        portfolio_daily_returns = [0.0] * n_days
        for t in range(1, n_days):
            portfolio_daily_returns[t] = sum(daily_contribs[grp][t] for grp in groups)

        for t in range(1, n_days):
            cum_return_factor = math.prod(
                1.0 + portfolio_daily_returns[i] for i in range(1, t + 1)
            )
            total_cum_return = cum_return_factor - 1.0
            total_link_factor = (
                math.log1p(total_cum_return) / total_cum_return
                if total_cum_return != 0.0
                else 1.0
            )

            for grp in groups:
                linked_sum = 0.0
                for tau in range(1, t + 1):
                    port_ret = portfolio_daily_returns[tau]
                    period_link = (
                        math.log1p(port_ret) / port_ret if port_ret != 0.0 else 1.0
                    )
                    contrib = daily_contribs[grp][tau]
                    linked_sum += contrib * (period_link / total_link_factor)
                cumulative_contribs[grp][t] = linked_sum

    else:
        msg = f"Unknown method '{method}'"
        raise UnknownCompoundMethodError(msg)

    # Format output as list of {"date": ..., "value": ...}
    total_series = [{"date": dates[t], "value": series[t]} for t in range(n_days)]
    daily_series: dict[str, list[dict[str, str | float]]] = {}
    cumulative_series: dict[str, list[dict[str, str | float]]] = {}
    for grp in groups:
        daily_series[grp] = [
            {"date": dates[t], "value": daily_contribs[grp][t]} for t in range(n_days)
        ]
        cumulative_series[grp] = [
            {"date": dates[t], "value": cumulative_contribs[grp][t]}
            for t in range(n_days)
        ]

    return daily_series, cumulative_series, total_series


def attrib_area(
    data: OpenFrame,
    series: OpenTimeSeries,
    filename: str,
    title: str | None = None,
    title_font_size: int = 32,
    tick_fmt: str = ".2%",
    directory: str | None = None,
    *,
    values_in_legend: bool = True,
    add_logo: bool = True,
    auto_open: bool = True,
) -> tuple[Figure, Path]:
    """Create and save an area chart of attribution series with Plotly.

    Args:
        data: OpenFrame containing group time series data.
        series: OpenTimeSeries of total portfolio series.
        filename: Base filename (without extension) for the saved plot.
        title: Optional chart title.
        title_font_size: Font size for the title text.
        tick_fmt: Format string for axis ticks and legend values.
        directory: Directory to write the HTML file. Defaults to ~/Documents.
        values_in_legend: If True, append returns to legend labels.
        add_logo: If True, include the default logo in the chart.
        auto_open: If True, open the HTML file after saving.

    Returns:
        A tuple (figure, filepath) where figure is the Plotly Figure object
        and filepath is the Path to the saved HTML file.

    """
    if directory is None:
        directory = Path.home() / "Documents"

    areaframe = data.from_deepcopy()
    areaseries = series.from_deepcopy()
    areaseries.to_cumret()

    if values_in_legend:
        total = []
        for serie, ret in zip(
            areaframe.constituents, areaframe.value_ret, strict=False
        ):
            total.append(ret)
            serie.set_new_label(f"{serie.label}: {ret:{tick_fmt}}")

    areaframe.tsdf = concat([x.tsdf for x in areaframe.constituents], axis="columns")
    areaframe.merge_series(how="inner").value_nan_handle(method="drop")

    figure, plotfile = areaframe.plot_series(
        auto_open=False,
        tick_fmt=tick_fmt,
        directory=directory,
        filename=f"{filename}.html",
        add_logo=add_logo,
    )

    figure.update_traces(
        fill="tonexty",
        mode="none",
        stackgroup="one",
        hovertemplate=(
            f"<extra></extra>Value: %{{y:{tick_fmt}}}<br>Date: %{{x|{'%Y-%m-%d'}}}"
        ),
        hoverlabel={
            "bgcolor": "white",
            "bordercolor": "white",
            "font": {"color": "#01579B"},
        },
    )

    if values_in_legend:
        series_name = str(areaseries.label)
        areaseries.set_new_label(f"{series_name}: {areaseries.value_ret:{tick_fmt}}")

    areaseries.tsdf = areaseries.tsdf.sub(1.0)

    figure.add_scatter(
        x=areaseries.tsdf.index,
        y=areaseries.tsdf.iloc[:, 0],
        hovertemplate=(
            f"<extra></extra>Value: %{{y:{tick_fmt}}}<br>Date: %{{x|{'%Y-%m-%d'}}}"
        ),
        marker={"size": 10},
        mode="markers",
        name=areaseries.label,
    )
    figure.update_layout(
        font={"size": 16},
        legend={
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.2,
            "xanchor": "center",
            "orientation": "h",
        },
        margin={"b": 100},
    )

    if title is not None:
        figure.update_layout(
            title={"text": f"<b>{title}</b>", "font": {"size": title_font_size}}
        )

    plot(
        figure_or_data=figure,
        filename=plotfile,
        auto_open=auto_open,
        link_text="",
        include_plotlyjs="cdn",
    )

    return figure, Path(plotfile)
