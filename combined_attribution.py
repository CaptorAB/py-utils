"""Combined attribution for Captor Aster HY, Aster Global Credit and Iris Bond.

Fetches performance for the three funds from a common start date, attributes
it to instrument type groups and saves the result to an xlsx file in
~/Documents.
"""

import datetime as dt
from pathlib import Path
from typing import Any

from openseries import get_previous_business_day_before_today
from pandas import DataFrame, ExcelWriter, Series, concat, to_datetime

from attribution import (
    compute_grouped_attribution_with_cumulative,
    get_party_name,
    get_performance,
)
from graphql_client import GraphqlClient

FUNDS: list[dict[str, Any]] = [
    {
        "fund_id": "62690582071ef0776524606c",
        "group_values": ["CdsIndex", "Bond", "CdsBasket", "Balance"],
    },
    {
        "fund_id": "605b2e5cc34cf5001154c90d",
        "group_values": [
            "CdsIndex",
            "Swap",
            "Bond",
            "Swaption",
            "CdsBasket",
            "Balance",
        ],
    },
    {
        "fund_id": "58e64b9523d2772e1859b705",
        "group_values": ["Bond", "Swap", "Swaption", "Balance"],
    },
]
START = dt.date(2022, 12, 6)
METHOD = "carino_menchero"
FEES_AND_COSTS_LABEL = "Fees & costs"
TOTAL_LABEL = "Fund total"
MERGED_GROUPS: dict[str, tuple[str, ...]] = {
    "Cds": ("CdsIndex", "CdsBasket"),
    "Swap": ("Swap", "Swaption"),
    "Bond": ("Bond", "Cash"),
}


def _to_frame(series_by_group: dict[str, list[dict[str, Any]]]) -> DataFrame:
    """Convert a mapping of group -> [{date, value}] into a date indexed DataFrame."""
    frame = DataFrame(
        {
            ("Cash" if group == "Balance" else group): [p["value"] for p in points]
            for group, points in series_by_group.items()
        },
        index=to_datetime([p["date"] for p in next(iter(series_by_group.values()))]),
    )
    frame.index.name = "Date"

    for label, groups in MERGED_GROUPS.items():
        columns = [col for col in groups if col in frame.columns]
        if columns:
            position = min(frame.columns.get_loc(col) for col in columns)
            merged = frame[columns].sum(axis="columns")
            frame = frame.drop(columns=columns)
            frame.insert(position, label, merged)
    return frame


def fund_attribution(
    gql_client: GraphqlClient,
    fund_id: str,
    group_values: list[str],
    start: dt.date,
    end: dt.date,
) -> tuple[str, DataFrame, DataFrame]:
    """Return fund name and daily and cumulative attribution for one fund.

    Both frames include a column with the fund's own total return.
    """
    fund_name = get_party_name(graphql=gql_client, party_id=fund_id)
    perfdata = get_performance(
        graphql=gql_client, client_id=fund_id, start_dt=start, end_dt=end
    )
    daily, cumulative, total, _ = compute_grouped_attribution_with_cumulative(
        data=perfdata,
        group_by="modelType",
        group_values=group_values,
        method=METHOD,
        fees_and_costs_label=FEES_AND_COSTS_LABEL,
        graphql=gql_client,
    )

    daily_df = _to_frame(daily)
    cumulative_df = _to_frame(cumulative)

    nav = Series(
        [p["value"] for p in total],
        index=to_datetime([p["date"] for p in total]),
    )
    daily_df[TOTAL_LABEL] = nav.pct_change().fillna(0.0).to_numpy()
    cumulative_df[TOTAL_LABEL] = (nav / nav.iloc[0] - 1.0).to_numpy()

    return fund_name, daily_df, cumulative_df


def trading_day_mask(daily: DataFrame) -> Series:
    """Return a boolean mask of the dates to keep in the output.

    The API returns calendar days. Days where no column has any contribution
    (weekends, common holidays) are dropped, which leaves every cumulative
    value unchanged. The first date is always kept as the base row.
    """
    mask = daily.ne(0.0).any(axis="columns")
    if not mask.empty:
        mask.iloc[0] = True
    return mask


if __name__ == "__main__":  # pragma: no cover
    gql_client = GraphqlClient()
    end = get_previous_business_day_before_today()

    daily_frames: dict[str, DataFrame] = {}
    cumulative_frames: dict[str, DataFrame] = {}
    for fund in FUNDS:
        name, daily_df, cumulative_df = fund_attribution(
            gql_client=gql_client,
            fund_id=fund["fund_id"],
            group_values=fund["group_values"],
            start=START,
            end=end,
        )
        daily_frames[name] = daily_df
        cumulative_frames[name] = cumulative_df

    daily_all = concat(
        daily_frames, axis="columns", names=["Fund", "Group"], sort=True
    )
    cumulative_all = concat(
        cumulative_frames, axis="columns", names=["Fund", "Group"], sort=True
    )

    trading_days = trading_day_mask(daily_all)
    daily_all = daily_all.loc[trading_days]
    cumulative_all = cumulative_all.loc[trading_days]

    summary = DataFrame(
        {name: frame.iloc[-1] for name, frame in cumulative_frames.items()}
    )
    summary = concat([summary.drop(index=TOTAL_LABEL), summary.loc[[TOTAL_LABEL]]])
    summary.loc["Sum of groups"] = summary.drop(index=TOTAL_LABEL).sum()
    summary.loc["Residual"] = summary.loc[TOTAL_LABEL] - summary.loc["Sum of groups"]
    summary.index.name = (
        f"Cumulative {START:%Y-%m-%d} to "
        f"{cumulative_all.index[-1]:%Y-%m-%d} ({METHOD})"
    )

    filename = (
        Path.home()
        / "Documents"
        / f"combined_attribution_{START:%Y%m%d}_{end:%Y%m%d}.xlsx"
    )
    with ExcelWriter(filename, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary")
        cumulative_all.to_excel(writer, sheet_name="Cumulative")
        daily_all.to_excel(writer, sheet_name="Daily")
        for name in cumulative_frames:
            cumulative_all[name].to_excel(writer, sheet_name=name[:31])

    print(f"Saved {filename}")  # noqa: T201
    print(summary.to_string(float_format="{:.4%}".format))  # noqa: T201
