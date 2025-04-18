"""Captor Aster Global Credit Short Term attribution analysis module."""

import datetime as dt

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

    fund_id = "62690a20071ef07765246144"
    fund_name = get_party_name(graphql=gql_client, party_id=fund_id)
    baseccy = "SEK"

    start = date_offset_foll(dt.date.today(), months_offset=-1)
    perfdata = get_performance(graphql=gql_client, client_id=fund_id, start_dt=start)

    _, cumperf, totserie = compute_grouped_attribution_with_cumulative(
        data=perfdata, group_by="modelType", group_values=["CdsIndex"], method="simple"
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
