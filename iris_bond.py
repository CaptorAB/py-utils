"""Captor Aster Global Credit Short Term attribution analysis module."""

import datetime as dt
from zoneinfo import ZoneInfo

from openseries import OpenFrame, OpenTimeSeries, date_offset_foll

from attribution import (
    attribution_area,
    compute_grouped_attribution_with_cumulative,
    get_party_name,
    get_performance,
)
from graphql_client import GraphqlClient

if __name__ == "__main__":
    gql_client = GraphqlClient()

    fund_id = "58e64b9523d2772e1859b705"
    fund_name = get_party_name(graphql=gql_client, party_id=fund_id)
    baseccy = "SEK"

    zone = ZoneInfo("Europe/Stockholm")
    today = dt.datetime.now(tz=zone).date()
    start = date_offset_foll(raw_date=today, months_offset=-3)
    perfdata = get_performance(graphql=gql_client, client_id=fund_id, start_dt=start)

    _, cumperf, totserie = compute_grouped_attribution_with_cumulative(
        data=perfdata,
        group_by="modelType",
        group_values=["Bond", "Swap"],
        method="simple",
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
        auto_open=True,
        add_logo=True,
    )
