"""Captor Aster Global High Yield attribution analysis module."""

import datetime as dt

from openseries import (
    OpenFrame,
    OpenTimeSeries,
    ValueType,
)
from pandas import DataFrame, concat

from attribution import (
    attribution_area,
    attribution_waterfall,
    compute_grouped_attribution_with_cumulative,
    get_party_name,
    get_performance,
)
from graphql_client import GraphqlClient

if __name__ == "__main__":
    gql_client = GraphqlClient()
    auto_open = True

    fund_id = "62690582071ef0776524606c"
    fund_name = get_party_name(graphql=gql_client, party_id=fund_id)

    start = dt.date(2025, 12, 30)
    perfdata = get_performance(graphql=gql_client, client_id=fund_id, start_dt=start)

    _, cumperf, totserie, baseccy = compute_grouped_attribution_with_cumulative(
        data=perfdata,
        group_by="modelType",
        group_values=["CdsIndex", "Bond", "CdsBasket", "Balance"],
        method="carino_menchero",
        fees_and_costs_label="Fees & costs",
    )

    navserie = OpenTimeSeries.from_arrays(
        name=fund_name,
        dates=[item["date"] for item in totserie],
        values=[item["value"] for item in totserie],
        baseccy=baseccy,
    )

    cds = DataFrame()

    for key, value in cumperf.items():
        if key in ["CdsIndex", "CdsBasket"]:
            tmp = OpenTimeSeries.from_arrays(
                name=key,
                dates=[item["date"] for item in value],
                values=[item["value"] for item in value],
                baseccy=baseccy,
            )
            tmp.tsdf = tmp.tsdf.add(1.0)
            tmp.value_to_ret()
            cds = concat([cds, tmp.tsdf], axis="columns", sort=True)

    cds["cds"] = cds.sum(axis="columns")
    cds_series = OpenTimeSeries.from_df(
        dframe=cds.loc[:, "cds"], valuetype=ValueType.RTRN
    )
    cds_series.to_cumret()
    cds_series.set_new_label("CdsIndex")
    cds_series.tsdf = cds_series.tsdf.sub(1.0)

    frame = OpenFrame(constituents=[cds_series])

    for key, value in cumperf.items():
        if key not in ["CdsIndex", "CdsBasket"]:
            label = "Cash" if key == "Balance" else key
            tmp = OpenTimeSeries.from_arrays(
                name=label,
                dates=[item["date"] for item in value],
                values=[item["value"] for item in value],
                baseccy=baseccy,
            )
            frame.add_timeseries(tmp)

    frame.tsdf = frame.tsdf.add(1.0)

    _, _ = attribution_area(
        data=frame,
        series=navserie,
        title=fund_name,
        tick_fmt=".2%",
        filename=f"{fund_name.replace(' ', '').replace('-', '')}_area",
        auto_open=auto_open,
    )
    _, _ = attribution_waterfall(
        data=frame,
        title=fund_name,
        filename=f"{fund_name.replace(' ', '').replace('-', '')}_waterfall",
        auto_open=auto_open,
    )
