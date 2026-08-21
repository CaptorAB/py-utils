"""Captor Global Fixed Income attribution analysis module."""

from openseries import (
    OpenFrame,
    OpenTimeSeries,
    get_previous_business_day_before_today,
    report_html,
)

from attribution import (
    attribution_area,
    attribution_waterfall,
    compute_grouped_attribution_with_cumulative,
    get_party_name,
    get_performance,
    get_timeserie,
)
from graphql_client import GraphqlClient

if __name__ == "__main__":
    gql_client = GraphqlClient()

    fund_id = "649d36255de897ec4079bbae"
    fund_name = get_party_name(graphql=gql_client, party_id=fund_id)

    start = None
    end = get_previous_business_day_before_today()
    perfdata = get_performance(
        graphql=gql_client, client_id=fund_id, start_dt=start, end_dt=end
    )

    _, cumperf, totserie, baseccy = compute_grouped_attribution_with_cumulative(
        data=perfdata,
        group_by="modelType",
        group_values=["Bond", "CdsIndex", "CdsBasket"],
        method="carino_menchero",
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
                name=key,
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
        filename=f"{fund_name.replace(' ', '').replace('-', '')}_waterfall",
    )

    compare_id = "658d5593d5913159f8909a60"
    compare_name = "Portfolio of Global IG 70 and HY 30 indices hedged SEK"
    compareserie = get_timeserie(
        graphql=gql_client, timeseries_id=compare_id, name=compare_name
    )

    compare = OpenFrame(constituents=[navserie, compareserie])
    compare.trunc_frame()
    report_html(
        data=compare,
        bar_freq="BQE",
        title=f"{fund_name}",
        filename=f"{fund_name.replace(' ', '').replace('-', '')}_report.html",
        auto_open=True,
    )
