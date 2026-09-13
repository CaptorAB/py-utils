"""Captor Iris Bond attribution analysis module."""

import datetime as dt

from openseries import (
    OpenFrame,
    OpenTimeSeries,
    get_previous_business_day_before_today,
    report_html,
)

from attribution import (
    attribution_area,
    attribution_waterfall,
    bar_freq_for_period,
    compute_grouped_attribution_with_cumulative,
    get_party_name,
    get_performance,
    get_timeserie,
)
from graphql_client import GraphqlClient

if __name__ == "__main__":
    gql_client = GraphqlClient()

    fund_id = "58e64b9523d2772e1859b705"
    fund_name = get_party_name(graphql=gql_client, party_id=fund_id)

    start = dt.date(2023, 9, 10)
    end = get_previous_business_day_before_today()
    perfdata = get_performance(
        graphql=gql_client, client_id=fund_id, start_dt=start, end_dt=end
    )

    _, cumperf, totserie, baseccy = compute_grouped_attribution_with_cumulative(
        data=perfdata,
        group_by="modelType",
        group_values=["Bond", "Swap", "Swaption", "Balance"],
        method="simple",
        fees_and_costs_label="Fees and Costs",
        graphql=gql_client,
    )

    navserie = OpenTimeSeries.from_arrays(
        name=fund_name,
        dates=[item["date"] for item in totserie],
        values=[item["value"] for item in totserie],
        baseccy=baseccy,
    )

    frame = OpenFrame(
        constituents=[
            OpenTimeSeries.from_arrays(
                name="Cash" if key == "Balance" else key,
                dates=[item["date"] for item in value],
                values=[item["value"] for item in value],
                baseccy=baseccy,
            )
            for key, value in cumperf.items()
        ]
    )
    frame.tsdf = frame.tsdf.add(1.0)

    _, _ = attribution_area(
        data=frame,
        series=navserie,
        title=fund_name,
        tick_fmt=".3%",
        filename=f"{fund_name.replace(' ', '').replace('-', '')}_area",
    )

    _, _ = attribution_waterfall(
        data=frame,
        title=fund_name,
        tick_fmt=".2%",
        filename=f"{fund_name.replace(' ', '').replace('-', '')}_waterfall",
    )

    compare_id = "63892890473ba6918f4ee954"
    compare_name = "1.6 x Govt Bond index"
    compareserie = get_timeserie(
        graphql=gql_client, timeseries_id=compare_id, name=compare_name
    )

    compare = OpenFrame(constituents=[navserie, compareserie])
    compare.trunc_frame()
    report_html(
        data=compare,
        bar_freq=bar_freq_for_period(
            start_idx=compare.first_idx, end_idx=compare.last_idx
        ),
        title="Captor Iris Bond",
        filename=f"{fund_name.replace(' ', '').replace('-', '')}_report.html",
        auto_open=True,
    )
