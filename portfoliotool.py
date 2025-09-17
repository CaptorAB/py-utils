"""openseries examples."""

import requests
from openseries import OpenFrame, OpenTimeSeries, ValueType
from pandas import set_option


class MissingIsinsError(Exception):
    """Raised when make_fund_basket does not return timeseries for all ISINs."""


def make_fund_basket(positions: dict[str, float], timeout: int = 10) -> OpenFrame:
    """Create an OpenFrame basket of funds based on provided ISIN positions.

    Args:
        positions (dict[str, float]): Dictionary of ISINs and their corresponding
            weights.
        timeout (int): Timeout in seconds for the HTTP request. Defaults to 10.

    Returns:
        OpenFrame: An OpenFrame object containing the weighted fund time series.

    Raises:
        MissingIsinsError: If any ISINs from the input positions are missing in the
            fetched data.

    """
    response = requests.get(
        url="https://api.captor.se/public/api/nav",
        timeout=timeout,
    )
    response.raise_for_status()

    found, weights, series = set(), [], []
    result = response.json()
    for data in result:
        if data["isin"] in positions:
            found.add(data["isin"])
            weights.append(positions[data["isin"]])
            series.append(
                OpenTimeSeries.from_arrays(
                    name=data["longName"],
                    isin=data["isin"],
                    baseccy=data["currency"],
                    dates=data["dates"],
                    values=data["navPerUnit"],
                    valuetype=ValueType.PRICE,
                ),
            )

    if len(set(positions.keys()) - found) != 0:
        msg = (
            "Request for NAV series failed. "
            f"Missing ISINs are: {set(positions.keys()) - found}"
        )
        raise MissingIsinsError(msg)

    return OpenFrame(constituents=series, weights=weights)


if __name__ == "__main__":
    set_option("display.max_rows", None)
    set_option("display.max_columns", None)
    set_option("display.width", None)
    set_option("display.max_colwidth", None)

    funds = {
        "SE0015243886": 0.2,
        "SE0011337195": 0.2,
        "SE0011670843": 0.2,
        "SE0017832280": 0.2,
        "SE0017832330": 0.2,
    }
    basket = make_fund_basket(positions=funds)
    basket.value_nan_handle().trunc_frame().to_cumret()

    portfolio = OpenTimeSeries.from_df(basket.make_portfolio(name="Portfolio"))
    basket.add_timeseries(portfolio)

    figure, plotfile = basket.plot_series(
        tick_fmt=".1%", filename="portfolioplot.html", title="Captor Funds"
    )

    prop_df = basket.all_properties(
        properties=[
            "arithmetic_ret",
            "vol",
            "ret_vol_ratio",
            "sortino_ratio",
            "worst_month",
            "cvar_down",
            "first_indices",
            "last_indices",
        ],
    )
    prop_df.columns = prop_df.columns.droplevel(level=1)

    formats = [
        "{:.2%}",
        "{:.2%}",
        "{:.2f}",
        "{:.2f}",
        "{:.2%}",
        "{:.2%}",
        "{:%Y-%m-%d}",
        "{:%Y-%m-%d}",
    ]
    for item, f in zip(prop_df.index, formats, strict=False):
        prop_df.loc[item] = prop_df.loc[item].apply(
            lambda x, fmt=f: x if isinstance(x, str) else fmt.format(x),
        )

    print("\n", prop_df)  # noqa: T201
