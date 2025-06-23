"""Captor Aster Global Credit Short Term attribution analysis module."""

import datetime as dt
from zoneinfo import ZoneInfo

from openseries import OpenFrame, OpenTimeSeries, date_offset_foll, report_html

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

    fund_id = "62690a20071ef07765246144"
    fund_name = get_party_name(graphql=gql_client, party_id=fund_id)

    zone = ZoneInfo("Europe/Stockholm")
    today = dt.datetime.now(tz=zone).date()
    start = date_offset_foll(raw_date=today, months_offset=-33)
    perfdata = get_performance(graphql=gql_client, client_id=fund_id, start_dt=start)

    _, cumperf, totserie, baseccy = compute_grouped_attribution_with_cumulative(
        data=perfdata,
        group_by="modelType",
        group_values=["CdsIndex"],
        method="logreturn",
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
        filename=f"{fund_name.replace(' ', '').replace('-', '')}_waterfall",
    )

    compare_id = "637fe3c9d5c64606a55e3617"
    compare_name = "Sweden Corporate FRN index"
    compareserie = get_timeserie(
        graphql=gql_client, timeseries_id=compare_id, name=compare_name
    )

    compare = OpenFrame(constituents=[navserie, compareserie])
    compare.trunc_frame()
    report_html(
        data=compare,
        bar_freq="BQE",
        title="Captor Aster Global Credit Short Term",
        filename=f"{fund_name.replace(' ', '').replace('-', '')}_report.html",
        auto_open=True,
    )
