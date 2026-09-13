"""Captor Aster Global Credit attribution analysis module."""

import datetime as dt

from openseries import OpenFrame, OpenTimeSeries, ValueType, report_html
from pandas import DataFrame, concat

from attribution import (
    attribution_waterfall,
    bar_freq_for_period,
    compute_grouped_attribution_with_cumulative,
    compute_two_portfolio_diversification_series,
    get_party_name,
    get_performance,
    get_timeserie,
    returns_with_diversification_plot,
)
from graphql_client import GraphqlClient

if __name__ == "__main__":
    gql_client = GraphqlClient()
    auto_open = True

    fund_id = "605b2e5cc34cf5001154c90d"
    fund_name = get_party_name(graphql=gql_client, party_id=fund_id)
    filename_base = fund_name.replace(" ", "").replace("-", "")
    rolling_window = 63
    cds_scaling_factor = 1.0

    start = dt.date(2023, 9, 10)
    perfdata = get_performance(graphql=gql_client, client_id=fund_id, start_dt=start)
    _, cds_vs_other_cumulative, _ = compute_two_portfolio_diversification_series(
        data=perfdata,
        group_by="modelType",
        cds_like_groups=("CdsIndex", "CdsBasket"),
        rolling_window=rolling_window,
        cds_scaling_factor=cds_scaling_factor,
    )
    diversification_values = [
        point["value"]
        for idx, point in enumerate(
            cds_vs_other_cumulative["Rolling diversification benefit"]
        )
        if idx >= rolling_window - 1
    ]
    diversification_mean = sum(diversification_values) / len(diversification_values)
    print(  # noqa: T201
        "Mean rolling diversification benefit "
        f"(window={rolling_window}, cds_scaling_factor="
        f"{cds_scaling_factor:.2f}): {diversification_mean:.2%}",
    )

    diversification_label = "Rolling diversification benefit"
    two_portfolio_df = DataFrame(
        {
            label: [item["value"] for item in values]
            for label, values in cds_vs_other_cumulative.items()
        },
        index=[
            item["date"] for item in cds_vs_other_cumulative[diversification_label]
        ],
    )
    two_portfolio_df.loc[
        two_portfolio_df.index[: rolling_window - 1],
        diversification_label,
    ] = float("nan")
    _, _ = returns_with_diversification_plot(
        plot_df=two_portfolio_df,
        filename=f"{filename_base}_cds_other_diversification.html",
        title=f"{fund_name} - CDS vs IR instruments and diversification",
        diversification_column=diversification_label,
        auto_open=auto_open,
    )

    _, cumperf, totserie, baseccy = compute_grouped_attribution_with_cumulative(
        data=perfdata,
        group_by="modelType",
        group_values=["CdsIndex", "Swap", "Bond", "Swaption", "CdsBasket", "Balance"],
        method="carino_menchero",
        fees_and_costs_label="Fees & costs",
        graphql=gql_client,
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

    _, _ = attribution_waterfall(
        data=frame,
        title=fund_name,
        tick_fmt=".2%",
        filename=f"{filename_base}_waterfall",
        auto_open=auto_open,
    )

    navserie = OpenTimeSeries.from_arrays(
        name=fund_name,
        dates=[item["date"] for item in totserie],
        values=[item["value"] for item in totserie],
        baseccy=baseccy,
    )
    compare_id = "6391a977e6a359fc24e82ba4"
    compare_name = "1.4 x Bloomberg Global Agg Corp hedged SEK"
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
        title="Captor Aster Global Credit",
        filename=f"{filename_base}_report.html",
        auto_open=auto_open,
    )
