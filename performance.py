"""Module for performance report."""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
from openseries import OpenFrame, OpenTimeSeries, ValueType

from graphql_client import GraphqlClient, GraphqlError

CLIENT = ""
EXCLUDED_ACCOUNTS = []
MONTHS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def get_account_performance(
    account_id: str,
    gql_client: GraphqlClient,
    start_dt: dt.date | None = None,
    end_dt: dt.date | None = None,
    *,
    look_through: bool = False,
) -> OpenTimeSeries | None:
    """Fetch and compute performance series for an account from the GraphQL API."""
    variables = {"accountId": account_id, "lookThrough": look_through}

    query = """ query accountPerformance(
                $accountId: GraphQLObjectId!
                $startDate: GraphQLDateString
                $endDate: GraphQLDateString
                $lookThrough: Boolean
                ) {
                accountPerformance(
                    accountId: $accountId
                    lookThrough: $lookThrough
                    filter: {startDate: $startDate, endDate: $endDate}
                ) {
                    currency
                    dates
                    values
                    cashFlows
                }
                } """
    if start_dt:
        variables.update({"startDate": start_dt.strftime("%Y-%m-%d")})
    if end_dt:
        variables.update({"endDate": end_dt.strftime("%Y-%m-%d")})

    data, error = gql_client.query(query_string=query, variables=variables)
    if error:
        raise GraphqlError(str(error))

    try:
        dates = data["accountPerformance"]["dates"]
    except TypeError:
        msg = f"Account {account_id} has no valid performance data."
        sys.stderr.write(f"{msg}\n")
        return None
    else:
        values = data["accountPerformance"]["values"]
        cashflows = data["accountPerformance"]["cashFlows"]

        portfolio_df = pd.DataFrame(
            {
                "value": values,
                "cashflow": cashflows,
            },
            index=pd.to_datetime(dates),
        ).sort_index()

        if portfolio_df.empty:
            msg = f"Account {account_id} has no performance data."
            sys.stderr.write(f"{msg}\n")
            return None

        prev_value = portfolio_df["value"].shift(1)
        portfolio_return = pd.Series(index=portfolio_df.index, dtype="float64")
        nonzero_prev_value = prev_value != 0.0
        portfolio_return.loc[nonzero_prev_value] = (
            portfolio_df.loc[nonzero_prev_value, "value"]
            - portfolio_df.loc[nonzero_prev_value, "cashflow"]
        ) / prev_value.loc[nonzero_prev_value] - 1.0
        # Treat first row or zero prior value as a neutral return.
        portfolio_df["portfolio_return"] = portfolio_return.fillna(0.0)

        filtered_dates = portfolio_df.index.strftime("%Y-%m-%d").tolist()
        cumulative_values = (1.0 + portfolio_df["portfolio_return"]).cumprod().tolist()

        return OpenTimeSeries.from_arrays(
            name=account_id,
            dates=filtered_dates,
            values=cumulative_values,
            baseccy=data["accountPerformance"]["currency"],
            valuetype=ValueType.PRICE,
        )


def _select_price_close_timeseries(
    time_series: list[dict[str, str | list[str] | list[float]]],
) -> dict[str, str | list[str] | list[float]]:
    """Return the time series item for Price(Close).

    Args:
        time_series: A list of time series dicts from GraphQL.

    Returns:
        The time series dict matching Price(Close).

    Raises:
        ValueError: When a Price(Close) time series is missing.
    """
    series = next(
        (item for item in time_series if item.get("type") == "Price(Close)"), None
    )
    if series is None:
        err_msg = "Missing Price(Close) time series for benchmark instrument."
        raise ValueError(err_msg)
    return series


def _model_index_benchmark_for_account(
    account: dict,
) -> OpenTimeSeries:
    """Build the model index benchmark series for an account.

    For Sum accounts uses account["modelIndexBenchmark"].
    For Physical accounts uses the benchmark in account["benchmarks"]
    where mainBenchmark is true.
    """
    if account["type"] == "Physical":
        main_bmk = next(
            (bmk for bmk in account["benchmarks"] if bmk.get("mainBenchmark")), None
        )
        if main_bmk and (
            price_ts := _select_price_close_timeseries(
                time_series=main_bmk["instrument"]["timeSeries"]
            )
        ):
            name = (
                f"{main_bmk['comment'] or main_bmk['instrument']['longName']} "
                f"(Index for {account['description']})"
            )
            return OpenTimeSeries.from_arrays(
                name=name,
                baseccy=main_bmk["currency"],
                timeseries_id=price_ts["_id"],
                instrument_id=main_bmk["instrument"]["_id"],
                dates=price_ts["dates"],
                values=[float(val) for val in price_ts["values"]],
            ).running_adjustment(adjustment=main_bmk["offset"])
    mib = account["modelIndexBenchmark"]
    return OpenTimeSeries.from_arrays(
        name=mib["name"]
        if mib["name"] != "ModelWeightedIndex"
        else f"ModelWeightedIndex ({account['description']})",
        baseccy=mib["currency"],
        dates=[item["date"] for item in mib["timeSeries"]["items"]],
        values=[float(item["value"]) for item in mib["timeSeries"]["items"]],
    )


def get_accounts(
    gql: GraphqlClient, client_id: str
) -> dict[str, dict[str, str | OpenTimeSeries | list[OpenTimeSeries]]]:
    """Get accounts from the Captor database.

    Args:
        gql: The GraphqlClient instance.
        client_id: The client ID.

    Returns:
        A dictionary of accounts.
    """
    query = """ query party($clientId: GraphQLObjectId) {
                  party(_id: $clientId) {
                    firstTradeDate
                    accounts {
                      name
                      _id
                      description
                      type
                      benchmarks {
                        offset
                        currency
                        comment
                        mainBenchmark
                        instrument {
                          _id
                          name
                          longName
                          timeSeries {
                            _id
                            type
                            dates
                            values
                          }
                        }
                      }
                      modelIndexBenchmark {
                        name
                        currency
                        timeSeries {
                          items {
                            date
                            value
                          }
                        }
                      }
                    }
                  }
                } """
    variables = {"clientId": client_id}

    data, error = gql.query(query, variables=variables)

    if error:
        raise GraphqlError(str(error))

    return {
        account["_id"]: {
            "description": account["description"],
            "type": account["type"],
            "firstTradeDate": data["party"]["firstTradeDate"],
            "benchmarks": [
                OpenTimeSeries.from_arrays(
                    name=bmk["comment"] or bmk["instrument"]["longName"],
                    baseccy=bmk["currency"],
                    timeseries_id=price_ts["_id"],
                    instrument_id=bmk["instrument"]["_id"],
                    dates=price_ts["dates"],
                    values=[float(val) for val in price_ts["values"]],
                ).running_adjustment(adjustment=bmk["offset"])
                for bmk in account["benchmarks"]
                if (
                    price_ts := _select_price_close_timeseries(
                        time_series=bmk["instrument"]["timeSeries"]
                    )
                )
                and (account["type"] != "Physical" or not bmk.get("mainBenchmark"))
            ],
            "modelIndexBenchmark": _model_index_benchmark_for_account(account),
        }
        for account in data["party"]["accounts"]
    }


def performance_report(
    graphql: GraphqlClient,
    client: str,
    start_dt: dt.date | None = None,
    end_dt: dt.date | None = None,
    excluded_accounts: list[str] | None = None,
) -> OpenFrame:
    """Build an OpenFrame of cumulative return series for all client accounts.

    Fetches performance data for each account from the GraphQL API, aligns each
    account with its benchmarks and model index benchmark, truncates to the
    requested date range (and the account's first trade date), and rebases all
    series to 1 at the start of the period.

    Args:
        graphql: GraphQL client for querying account and performance data.
        client: Client ID (party _id) whose accounts to include.
        start_dt: Optional start date for the performance period.
        end_dt: Optional end date for the performance period.
        excluded_accounts: Account IDs to exclude from the report.

    Returns:
        OpenFrame of cumulative return time series for each included account,
        their benchmarks, and model index benchmark, all rebased to 1.

    Raises:
        GraphqlError: If the account or performance GraphQL query fails.

    Note:
        Accounts in excluded_accounts are skipped. If a date-filtered
        performance query raises GraphqlError, it is retried without date
        filters. Accounts with no valid performance data are skipped and
        reported to stderr.
    """
    if excluded_accounts is None:
        excluded_accounts = []

    accounts = get_accounts(gql=graphql, client_id=client)

    constituents = []
    errors = []
    for account_id, account_data in accounts.items():
        start = start_dt
        end = end_dt

        if account_id in excluded_accounts:
            continue

        try:
            tmp_performance = get_account_performance(
                account_id=account_id,
                gql_client=graphql,
                start_dt=start,
                end_dt=end,
            )
        except GraphqlError:
            start = None
            end = None
            tmp_performance = get_account_performance(
                account_id=account_id,
                gql_client=graphql,
                start_dt=None,
                end_dt=None,
            )

        if not isinstance(tmp_performance, OpenTimeSeries):
            continue

        if start:
            start = max(start, tmp_performance.first_idx)
        if end:
            end = min(end, tmp_performance.last_idx)

        performance = get_account_performance(
            account_id=account_id,
            gql_client=graphql,
            start_dt=start,
            end_dt=end,
        ).set_new_label(lvl_zero=account_data["description"])

        if performance is None:
            errors.append(f"- Account '{account_data['description']}'")
            continue

        trunc_start = max(
            v
            for v in [
                start,
                start_dt,
                performance.first_idx,
                dt.datetime.strptime(account_data["firstTradeDate"], "%Y-%m-%d")
                .astimezone()
                .date(),
            ]
            if v is not None
        )

        trunc_end = min(
            v for v in [end, end_dt, performance.last_idx] if v is not None
        )

        tmp_frame = OpenFrame(
            constituents=[performance]
            + account_data["benchmarks"]
            + [account_data["modelIndexBenchmark"]]
        )
        tmp_frame.trunc_frame(start_cut=trunc_start, end_cut=trunc_end)

        constituents.extend(tmp_frame.constituents)

    frame = OpenFrame(constituents=constituents)
    frame.to_cumret()

    if len(errors) > 0:
        sys.stderr.write(f"Errors: {errors}\n")

    return frame


if __name__ == "__main__":
    graphql = GraphqlClient()

    start = None  # dt.date(2024, 12, 30)
    end = None  # dt.date(2025, 12, 30)

    frame = performance_report(
        graphql=graphql,
        client=CLIENT,
        excluded_accounts=EXCLUDED_ACCOUNTS,
        start_dt=start,
        end_dt=end,
    )

    thisyr = frame.last_idx.year
    thismth = frame.last_idx.month
    results = frame.value_ret_calendar_period(year=thisyr)
    results.name = f"YTD {thisyr}"
    results.index = results.index.droplevel(level=1)

    mtd = frame.value_ret_calendar_period(year=thisyr, month=thismth)
    mtd.name = f"MTD {MONTHS[thismth]} {thisyr}"
    mtd.index = mtd.index.droplevel(level=1)
    results = pd.concat([results, mtd], axis="columns")

    for period, label in zip(
        [12, 36, 60],
        ["1 year", "3 year", "5 year"],
        strict=True,
    ):
        retrn = frame.geo_ret_func(months_from_last=period)
        retrn.name = label
        retrn.index = retrn.index.droplevel(level=1)
        results = pd.concat([results, retrn], axis="columns")

    first = frame.first_idx.strftime("%Y%m%d")
    last = frame.last_idx.strftime("%Y%m%d")
    filename_performance = f"performance_{first}_{last}"
    filename_summary = f"summary_{first}_{last}"

    # frame.plot_series(filename=f"{filename}.html")  # noqa: ERA001

    frame.to_xlsx(filename=f"{filename_performance}.xlsx")

    results_xlsx = Path.home() / "Documents" / f"{filename_summary}.xlsx"
    results.to_excel(excel_writer=results_xlsx, engine="openpyxl")
