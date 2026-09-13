"""Pytest suite for attribution.py module without using bare asserts.

Defines AttributionTestError for signaling test failures.
Covers get_party_name, get_performance,
compute_grouped_attribution_with_cumulative, diversification helpers,
attribution_area, and attribution_waterfall.
Targets Python 3.14 and follows Ruff standards.
"""

import datetime as dt
import math
import warnings
from pathlib import Path
from shutil import rmtree as shutil_rmtree
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from openseries import OpenFrame

    from graphql_client import GraphqlClient

try:
    from typing import Self
except ImportError:
    from typing import Self

import pandas as pd
import pytest

import attribution as am
from attribution import (
    CannotCompoundReturnError,
    MissingGroupValueWarning,
    PortfolioValueZeroError,
    UnknownCompoundMethodError,
    UnknownGroupValueError,
    ZeroGroupContributionWarning,
)


class AttributionTestError(Exception):
    """Custom exception to signal test failure in attribution tests."""


class DummyGraphqlClient:
    """Stub for GraphqlClient to simulate API responses."""

    def __init__(self, data: Any, error: Any) -> None:
        """Initialize with preset data and error.

        Args:
            data: The data to return from query.
            error: The error to return from query.

        """
        self._data = data
        self._error = error

    def query(self, query_string: str, variables: dict[str, Any]) -> Any:
        """Simulate a GraphQL query returning (data, error).

        Returns:
            A tuple (data, error).

        """
        return self._data, self._error


class TestAttribution:
    """Test suite for attribution module functionality."""

    @pytest.fixture
    def graphql_client_success(self) -> DummyGraphqlClient:
        """Fixture for a successful DummyGraphqlClient stub."""
        data = {"party": {"longName": "Sample Fund"}}
        return DummyGraphqlClient(data, None)

    @pytest.fixture
    def graphql_client_error(self) -> DummyGraphqlClient:
        """Fixture for an error DummyGraphqlClient stub."""
        return DummyGraphqlClient(None, "error occurred")

    @pytest.fixture
    def sample_data(self) -> dict[str, Any]:
        """Fixture providing sample data for attribution tests."""
        perf1 = {
            "values": [100.0, 120.0],
            "cashFlows": [0.0, 0.0],
            "instrument": {"modelType": "G1", "currency": "EUR"},
        }
        perf2 = {
            "values": [200.0, 180.0],
            "cashFlows": [0.0, 0.0],
            "instrument": {"modelType": "G2", "currency": "USD"},
        }
        dates = ["d1", "d2"]
        series = [0.0, (120.0 - 100.0 + 180.0 - 200.0) / 300.0]
        return {
            "dates": dates,
            "series": series,
            "instrumentPerformances": [perf1, perf2],
        }

    def test_get_party_name_success(
        self, graphql_client_success: DummyGraphqlClient
    ) -> None:
        """Test get_party_name returns correct longName on success."""
        result = am.get_party_name(
            graphql=cast("GraphqlClient", graphql_client_success), party_id="id123"
        )
        msg = f"Expected 'Sample Fund', got '{result}'"
        if result != "Sample Fund":
            raise AttributionTestError(msg)

    def test_get_party_name_error(
        self, graphql_client_error: DummyGraphqlClient
    ) -> None:
        """Test get_party_name raises GraphqlError on API error."""
        raised = False
        try:
            am.get_party_name(
                graphql=cast("GraphqlClient", graphql_client_error), party_id="id123"
            )
        except am.GraphqlError:
            raised = True
        msg = "GraphqlError was not raised for error response"
        if not raised:
            raise AttributionTestError(msg)

    def test_get_performance_success(self) -> None:
        """Test get_performance returns performance dict on success."""
        payload = {
            "performance": {
                "dates": ["2025-01-01"],
                "series": [0.0],
                "instrumentPerformances": [
                    {
                        "values": [100.0],
                        "cashFlows": [0.0],
                        "instrument": {"modelType": "M", "currency": "X"},
                    }
                ],
            }
        }
        client = DummyGraphqlClient(payload, None)
        result = am.get_performance(
            graphql=cast("GraphqlClient", client), client_id="c1"
        )
        expected_series = [0.0]
        rec_series = result.get("series")
        msg = f"Expected series {expected_series}, got {rec_series}"
        if rec_series != expected_series:
            raise AttributionTestError(msg)

    def test_get_performance_error(self) -> None:
        """Test get_performance raises GraphqlError on API error."""
        client = DummyGraphqlClient(None, "fetch failed")
        raised = False
        try:
            am.get_performance(graphql=cast("GraphqlClient", client), client_id="c2")
        except am.GraphqlError:
            raised = True
        msg = "GraphqlError was not raised on performance fetch error"
        if not raised:
            raise AttributionTestError(msg)

    def test_compute_simple_method(self, sample_data: dict[str, Any]) -> None:
        """Test simple method sums daily contributions correctly."""
        daily, cumu, total, _ = am.compute_grouped_attribution_with_cumulative(
            data=sample_data,
            group_by="modelType",
            group_values=["G1"],
            method="simple",
        )

        expected_g1 = (120.0 - 100.0) / 300.0
        expected_other = (180.0 - 200.0) / 300.0

        msg1 = f"G1 daily wrong: {daily['G1'][1]}"
        if not math.isclose(daily["G1"][1]["value"], expected_g1, rel_tol=1e-9):
            raise AttributionTestError(msg1)

        msg2 = f"Other daily wrong: {daily['Other'][1]}"
        if not math.isclose(daily["Other"][1]["value"], expected_other, rel_tol=1e-9):
            raise AttributionTestError(msg2)

        msg3 = f"G1 cumulative wrong: {cumu['G1'][1]}"
        if not math.isclose(cumu["G1"][1]["value"], expected_g1, rel_tol=1e-9):
            raise AttributionTestError(msg3)

        expected_total = [
            {"date": "d1", "value": 0.0},
            {"date": "d2", "value": sample_data["series"][1]},
        ]

        msg4 = f"Total mismatched: {total}"
        if total != expected_total:
            raise AttributionTestError(msg4)

    def test_compute_logreturn_method(self, sample_data: dict[str, Any]) -> None:
        """Test logreturn method compounds via log1p correctly."""
        _, cumu, _, _ = am.compute_grouped_attribution_with_cumulative(
            data=sample_data,
            group_by="modelType",
            group_values=["G1"],
            method="logreturn",
        )
        expected = (120.0 - 100.0) / 300.0

        msg = f"G1 logreturn wrong: {cumu['G1'][1]}"
        if not math.isclose(cumu["G1"][1]["value"], expected, rel_tol=1e-9):
            raise AttributionTestError(msg)

    def test_compute_logreturn_error(self, sample_data: dict[str, Any]) -> None:
        """Test logreturn raises CannotCompoundReturnError when return <= -1."""
        bad_perf = {
            "values": [100.0, 0.0],
            "cashFlows": [0.0, 0.0],
            "instrument": {"modelType": "G1", "currency": "EUR"},
        }
        data_bad = {**sample_data, "instrumentPerformances": [bad_perf]}
        raised = False
        try:
            am.compute_grouped_attribution_with_cumulative(
                data=data_bad,
                group_by="modelType",
                group_values=["G1"],
                method="logreturn",
            )
        except CannotCompoundReturnError:
            raised = True

        msg = "logreturn did not raise CannotCompoundReturnError on invalid return"
        if not raised:
            raise AttributionTestError(msg)

    def test_compute_carino_menchero_method(self, sample_data: dict[str, Any]) -> None:
        """Test Carino/Menchero linking for a single period."""
        _, cumu, _, _ = am.compute_grouped_attribution_with_cumulative(
            data=sample_data,
            group_by="modelType",
            group_values=["G1"],
            method="carino_menchero",
        )
        expected = (120.0 - 100.0) / 300.0

        msg = f"Carino/Menchero wrong: {cumu['G1'][1]}"
        if not math.isclose(cumu["G1"][1]["value"], expected, rel_tol=1e-9):
            raise AttributionTestError(msg)

    def test_compute_unknown_method(self, sample_data: dict[str, Any]) -> None:
        """Test unknown method raises UnknownCompoundMethodError."""
        raised = False
        try:
            am.compute_grouped_attribution_with_cumulative(
                data=sample_data,
                group_by="modelType",
                group_values=["G1"],
                method="invalid",  # type: ignore[arg-type]
            )
        except UnknownCompoundMethodError:
            raised = True

        msg = "Invalid method did not raise UnknownCompoundMethodError"
        if not raised:
            raise AttributionTestError(msg)

    def test_compute_zero_total_prev(self, sample_data: dict[str, Any]) -> None:
        """Test simple method raises PortfolioValueZeroError."""
        zero_perf = {
            "values": [0.0, 0.0],
            "cashFlows": [0.0, 0.0],
            "instrument": {"modelType": "G1", "currency": "EUR"},
        }
        data_zero = {**sample_data, "instrumentPerformances": [zero_perf]}
        raised = False
        try:
            am.compute_grouped_attribution_with_cumulative(
                data=data_zero,
                group_by="modelType",
                group_values=["G1"],
                method="simple",
            )
        except PortfolioValueZeroError:
            raised = True

        msg = "Zero total prev did not raise PortfolioValueZeroError"
        if not raised:
            raise AttributionTestError(msg)

    def test_compute_fxswap_error(self, sample_data: dict[str, Any]) -> None:
        """Test FxSwap error handling when no foreign currency leg is found."""
        fxswap_perf = {
            "values": [100.0, 120.0],
            "cashFlows": [0.0, 0.0],
            "instrument": {
                "modelType": "FxSwap",
                "_id": "fx123",
                "currency": "EUR",
                "model": {
                    "legs": [
                        {"currency": "EUR"}  # Only one leg with same currency
                    ]
                },
            },
        }
        data_fxswap = {**sample_data, "instrumentPerformances": [fxswap_perf]}
        raised = False
        try:
            am.compute_grouped_attribution_with_cumulative(
                data=data_fxswap,
                group_by="currency",
                group_values=["EUR"],
                method="simple",
                consider_fxswap=True,
            )
        except am.FxLegError:
            raised = True

        msg = "FxLegError was not raised for FxSwap with no foreign currency leg"
        if not raised:
            raise AttributionTestError(msg)

    def test_get_graphql_enum_values_success(self) -> None:
        """Test get_graphql_enum_values returns sorted enum names."""
        payload = {
            "__type": {
                "enumValues": [{"name": "Swap"}, {"name": "Bond"}],
            }
        }
        client = DummyGraphqlClient(payload, None)
        result = am.get_graphql_enum_values(
            graphql=cast("GraphqlClient", client),
            type_name="InstrumentModelTypeEnum",
            include_deprecated=True,
        )
        expected = ["Bond", "Swap"]
        msg = f"Expected {expected}, got {result}"
        if result != expected:
            raise AttributionTestError(msg)

    def test_get_graphql_enum_values_error(self) -> None:
        """Test get_graphql_enum_values raises GraphqlError on API error."""
        client = DummyGraphqlClient(None, "fetch failed")
        raised = False
        try:
            am.get_graphql_enum_values(
                graphql=cast("GraphqlClient", client),
                type_name="InstrumentModelTypeEnum",
            )
        except am.GraphqlError:
            raised = True
        msg = "GraphqlError was not raised on enum introspection error"
        if not raised:
            raise AttributionTestError(msg)

    def test_get_graphql_enum_values_type_missing(self) -> None:
        """Test get_graphql_enum_values raises when the type is missing."""
        client = DummyGraphqlClient({"__type": None}, None)
        raised = False
        try:
            am.get_graphql_enum_values(
                graphql=cast("GraphqlClient", client),
                type_name="NotAnEnum",
            )
        except am.GraphqlError:
            raised = True
        msg = "GraphqlError was not raised for a missing GraphQL type"
        if not raised:
            raise AttributionTestError(msg)

    def test_get_graphql_enum_values_not_enum(self) -> None:
        """Test get_graphql_enum_values raises when enumValues is missing."""
        client = DummyGraphqlClient({"__type": {"enumValues": None}}, None)
        raised = False
        try:
            am.get_graphql_enum_values(
                graphql=cast("GraphqlClient", client),
                type_name="Instrument",
            )
        except am.GraphqlError:
            raised = True
        msg = "GraphqlError was not raised when the type is not an enum"
        if not raised:
            raise AttributionTestError(msg)

    def test_get_graphql_enum_values_non_dict_data(self) -> None:
        """Test get_graphql_enum_values raises when data is not a dict."""
        client = DummyGraphqlClient(["not", "a", "dict"], None)
        raised = False
        try:
            am.get_graphql_enum_values(
                graphql=cast("GraphqlClient", client),
                type_name="InstrumentModelTypeEnum",
            )
        except am.GraphqlError:
            raised = True
        msg = "GraphqlError was not raised for a non-dict introspection payload"
        if not raised:
            raise AttributionTestError(msg)

    def test_unknown_group_value_raises(self, sample_data: dict[str, Any]) -> None:
        """Test an unknown schema enum value raises UnknownGroupValueError."""
        client = DummyGraphqlClient(
            {"__type": {"enumValues": [{"name": "G1"}, {"name": "G2"}]}},
            None,
        )
        raised = False
        message = ""
        try:
            am.compute_grouped_attribution_with_cumulative(
                data=sample_data,
                group_by="modelType",
                group_values=["Bnd"],
                graphql=cast("GraphqlClient", client),
            )
        except UnknownGroupValueError as exc:
            raised = True
            message = str(exc)
        msg = "UnknownGroupValueError was not raised for an unknown modelType"
        if not raised:
            raise AttributionTestError(msg)
        if "Bnd" not in message or "G1" not in message:
            msg2 = f"Error message missing choices or unknown value: {message}"
            raise AttributionTestError(msg2)

    def test_valid_group_value_missing_from_fund(
        self, sample_data: dict[str, Any]
    ) -> None:
        """Test a valid unused enum value warns and keeps a zero series."""
        client = DummyGraphqlClient(
            {
                "__type": {
                    "enumValues": [
                        {"name": "G1"},
                        {"name": "G2"},
                        {"name": "G3"},
                    ]
                }
            },
            None,
        )
        with pytest.warns(MissingGroupValueWarning, match="G3"):
            daily, _, _, _ = am.compute_grouped_attribution_with_cumulative(
                data=sample_data,
                group_by="modelType",
                group_values=["G1", "G3"],
                graphql=cast("GraphqlClient", client),
            )
        expected_zero = 0.0
        rec_zero = daily["G3"][1]["value"]
        msg = f"Expected unused G3 series to be {expected_zero}, got {rec_zero}"
        if not math.isclose(rec_zero, expected_zero, rel_tol=1e-9):
            raise AttributionTestError(msg)

    def test_missing_group_value_without_graphql(
        self, sample_data: dict[str, Any]
    ) -> None:
        """Test a missing value warns with present types when schema is unknown."""
        with pytest.warns(MissingGroupValueWarning, match="does not appear"):
            am.compute_grouped_attribution_with_cumulative(
                data=sample_data,
                group_by="modelType",
                group_values=["G1", "G3"],
            )

    def test_zero_contribution_warns(self, sample_data: dict[str, Any]) -> None:
        """Test a present group with no P&L emits ZeroGroupContributionWarning."""
        zero_pnl = {
            "values": [100.0, 100.0],
            "cashFlows": [0.0, 0.0],
            "instrument": {"modelType": "G1", "currency": "EUR"},
        }
        data_zero = {**sample_data, "instrumentPerformances": [zero_pnl]}
        with pytest.warns(ZeroGroupContributionWarning, match="G1"):
            am.compute_grouped_attribution_with_cumulative(
                data=data_zero,
                group_by="modelType",
                group_values=["G1"],
            )

    def test_unmapped_group_by_with_graphql(self, sample_data: dict[str, Any]) -> None:
        """Test graphql is ignored when group_by has no schema enum mapping."""
        client = DummyGraphqlClient(
            {"__type": {"enumValues": [{"name": "unused"}]}},
            None,
        )
        data = {
            **sample_data,
            "instrumentPerformances": [
                {
                    "values": [100.0, 120.0],
                    "cashFlows": [0.0, 0.0],
                    "instrument": {
                        "modelType": "G1",
                        "currency": "EUR",
                        "name": "A",
                    },
                }
            ],
        }
        with pytest.warns(MissingGroupValueWarning, match="does not appear"):
            am.compute_grouped_attribution_with_cumulative(
                data=data,
                group_by="name",
                group_values=["B"],
                graphql=cast("GraphqlClient", client),
            )

    def test_currency_group_with_graphql(self, sample_data: dict[str, Any]) -> None:
        """Test currency grouping uses CurrencyEnum introspection."""
        client = DummyGraphqlClient(
            {
                "__type": {
                    "enumValues": [
                        {"name": "EUR"},
                        {"name": "USD"},
                        {"name": "SEK"},
                    ]
                }
            },
            None,
        )
        with pytest.warns(MissingGroupValueWarning, match="SEK"):
            am.compute_grouped_attribution_with_cumulative(
                data=sample_data,
                group_by="currency",
                group_values=["EUR", "SEK"],
                graphql=cast("GraphqlClient", client),
            )

    def test_single_day_skips_zero_contribution_warning(
        self, sample_data: dict[str, Any]
    ) -> None:
        """Test a single-date series does not emit a zero-contribution warning."""
        data = {
            **sample_data,
            "dates": ["d1"],
            "series": [0.0],
            "instrumentPerformances": [
                {
                    "values": [100.0],
                    "cashFlows": [0.0],
                    "instrument": {"modelType": "G1", "currency": "EUR"},
                }
            ],
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            am.compute_grouped_attribution_with_cumulative(
                data=data,
                group_by="modelType",
                group_values=["G1"],
            )
        zero_warned = any(
            issubclass(item.category, ZeroGroupContributionWarning) for item in caught
        )
        msg = "ZeroGroupContributionWarning was emitted for a single-date series"
        if zero_warned:
            raise AttributionTestError(msg)

    def test_two_portfolio_diversification_series(self) -> None:
        """Test CDS vs other split, scaling, and rolling diversification."""
        data = {
            "dates": ["d1", "d2", "d3"],
            "series": [0.0, 0.01, 0.02],
            "instrumentPerformances": [
                {
                    "values": [100.0, 110.0, 121.0],
                    "cashFlows": [0.0, 0.0, 0.0],
                    "instrument": {"modelType": "CdsIndex", "currency": "EUR"},
                },
                {
                    "values": [200.0, 190.0, 185.0],
                    "cashFlows": [0.0, 0.0, 0.0],
                    "instrument": {"modelType": "Bond", "currency": "EUR"},
                },
            ],
        }
        daily, cumu, total = am.compute_two_portfolio_diversification_series(
            data=data,
            rolling_window=2,
            cds_scaling_factor=2.0,
        )
        expected_cds_d1 = 2.0 * (10.0 / 300.0)
        rec_cds = daily["CDS"][1]["value"]
        msg1 = f"Scaled CDS daily wrong: {rec_cds}"
        if not math.isclose(rec_cds, expected_cds_d1, rel_tol=1e-9):
            raise AttributionTestError(msg1)

        expected_other_d1 = -10.0 / 300.0
        rec_other = daily["Other instruments"][1]["value"]
        msg2 = f"Other daily wrong: {rec_other}"
        if not math.isclose(rec_other, expected_other_d1, rel_tol=1e-9):
            raise AttributionTestError(msg2)

        rec_cum_cds = cumu["CDS"][2]["value"]
        expected_cum_cds = expected_cds_d1 + 2.0 * (11.0 / 300.0)
        msg3 = f"CDS cumulative wrong: {rec_cum_cds}"
        if not math.isclose(rec_cum_cds, expected_cum_cds, rel_tol=1e-9):
            raise AttributionTestError(msg3)

        first_div = cumu["Rolling diversification benefit"][0]["value"]
        msg4 = f"Warmup diversification should be 0.0, got {first_div}"
        if first_div != 0.0:
            raise AttributionTestError(msg4)

        expected_total = [
            {"date": "d1", "value": 0.0},
            {"date": "d2", "value": 0.01},
            {"date": "d3", "value": 0.02},
        ]
        msg5 = f"Total series mismatched: {total}"
        if total != expected_total:
            raise AttributionTestError(msg5)

    def test_two_portfolio_diversification_invalid_window(self) -> None:
        """Test rolling_window less than 2 raises ValueError."""
        raised = False
        try:
            am.compute_two_portfolio_diversification_series(
                data={"dates": ["d1"], "series": [0.0], "instrumentPerformances": []},
                rolling_window=1,
            )
        except ValueError:
            raised = True
        msg = "rolling_window=1 did not raise ValueError"
        if not raised:
            raise AttributionTestError(msg)

    def test_two_portfolio_diversification_invalid_scale(self) -> None:
        """Test non-positive cds_scaling_factor raises ValueError."""
        raised = False
        try:
            am.compute_two_portfolio_diversification_series(
                data={"dates": ["d1"], "series": [0.0], "instrumentPerformances": []},
                cds_scaling_factor=0.0,
            )
        except ValueError:
            raised = True
        msg = "cds_scaling_factor=0.0 did not raise ValueError"
        if not raised:
            raise AttributionTestError(msg)

    def test_two_portfolio_diversification_zero_value(self) -> None:
        """Test zero previous portfolio value raises PortfolioValueZeroError."""
        data = {
            "dates": ["d1", "d2"],
            "series": [0.0, 0.0],
            "instrumentPerformances": [
                {
                    "values": [0.0, 0.0],
                    "cashFlows": [0.0, 0.0],
                    "instrument": {"modelType": "CdsIndex", "currency": "EUR"},
                }
            ],
        }
        raised = False
        try:
            am.compute_two_portfolio_diversification_series(
                data=data, rolling_window=2
            )
        except PortfolioValueZeroError:
            raised = True
        msg = "Zero total prev did not raise PortfolioValueZeroError"
        if not raised:
            raise AttributionTestError(msg)

    def test_bar_freq_for_period(self) -> None:
        """Test bar frequency thresholds and reversed date order."""
        monthly = am.bar_freq_for_period(dt.date(2024, 1, 15), dt.date(2024, 12, 15))
        if monthly != "BME":
            msg = f"12-month span should be BME, got {monthly}"
            raise AttributionTestError(msg)

        quarterly = am.bar_freq_for_period(dt.date(2023, 1, 1), dt.date(2024, 2, 1))
        if quarterly != "BQE":
            msg = f"13-month span should be BQE, got {quarterly}"
            raise AttributionTestError(msg)

        at_quarterly_cap = am.bar_freq_for_period(
            dt.date(2020, 1, 1), dt.date(2023, 1, 1)
        )
        if at_quarterly_cap != "BQE":
            msg = f"36-month span should be BQE, got {at_quarterly_cap}"
            raise AttributionTestError(msg)

        yearly = am.bar_freq_for_period(dt.date(2020, 1, 1), dt.date(2023, 2, 1))
        if yearly != "BYE":
            msg = f"37-month span should be BYE, got {yearly}"
            raise AttributionTestError(msg)

        swapped = am.bar_freq_for_period(dt.date(2024, 12, 15), dt.date(2024, 1, 15))
        if swapped != "BME":
            msg = f"Reversed 12-month span should be BME, got {swapped}"
            raise AttributionTestError(msg)

        from_timestamp = am.bar_freq_for_period(
            pd.Timestamp("2024-01-15"),
            dt.datetime(2024, 6, 15, tzinfo=dt.UTC),
        )
        if from_timestamp != "BME":
            msg = f"Timestamp inputs should be BME, got {from_timestamp}"
            raise AttributionTestError(msg)

    def test_diversification_color_bounds(self) -> None:
        """Test color-scale bounds from data, overrides, and collapsed range."""
        values = pd.Series([0.2, float("nan"), 0.8])
        lower, upper = am._diversification_color_bounds(values, None, None)
        if not math.isclose(lower, 0.2, rel_tol=1e-9) or not math.isclose(
            upper, 0.8, rel_tol=1e-9
        ):
            msg = f"Observed bounds wrong: {(lower, upper)}"
            raise AttributionTestError(msg)

        pinned = am._diversification_color_bounds(values, 0.0, 1.0)
        if pinned != (0.0, 1.0):
            msg = f"Pinned bounds wrong: {pinned}"
            raise AttributionTestError(msg)

        empty_lower, empty_upper = am._diversification_color_bounds(
            pd.Series([float("nan")]), None, None
        )
        if empty_lower != 0.0 or empty_upper != 1.0:
            msg = f"Empty-series bounds wrong: {(empty_lower, empty_upper)}"
            raise AttributionTestError(msg)

        pad_lower, pad_upper = am._diversification_color_bounds(
            pd.Series([0.5, 0.5]), None, None
        )
        if pad_lower >= pad_upper:
            msg = f"Collapsed bounds were not padded: {(pad_lower, pad_upper)}"
            raise AttributionTestError(msg)


class DummySeries:
    """Stub for OpenTimeSeries-like object used in attribution_area."""

    def __init__(self, label: str, tsdf: pd.DataFrame) -> None:
        """Initialize dummy series with label and dataframe.

        Args:
            label: Series label.
            tsdf: Time series dataframe.

        """
        self.label = label
        self.tsdf = tsdf

    def from_deepcopy(self) -> Self:
        """Return self for chaining."""
        return self

    def to_cumret(self) -> None:
        """Stub method to convert to cumulative returns."""

    @property
    def value_ret(self) -> float:
        """Stub method to convert to cumulative returns."""
        return 0.005

    def set_new_label(self, new_label: str) -> None:
        """Update the series label.

        Args:
            new_label: New label string.

        """
        self.label = new_label


class DummyFigure:
    """Stub for Plotly Figure-like object returned by plot_series."""

    def __init__(self) -> None:
        """Initialize with default values."""
        self.traces = []
        self.layout = {}

    def update_traces(self, **kwargs: Any) -> None:
        """Mock update_traces by storing the update."""
        self.traces.append(kwargs)

    def add_scatter(self, **kwargs: Any) -> None:
        """Mock add_scatter by storing the scatter data."""
        self.traces.append({"type": "scatter", **kwargs})

    def update_layout(self, **kwargs: Any) -> None:
        """Mock update_layout by storing the layout."""
        self.layout.update(kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Mock to_dict to return the figure data."""
        return {"data": self.traces, "layout": self.layout}


def mock_plot(figure_or_data: Any, **kwargs: Any) -> str:
    """Mock plotly.plot to return a div string."""
    return "<div>Mock Plotly Plot</div>"


@pytest.fixture
def mock_plotly(monkeypatch: Any) -> None:
    """Fixture to mock plotly functionality."""

    def mock_plot_html(
        figure: Any,
        plotfile: Path,
        title: str | None = None,
        output_type: str = "file",
        include_plotlyjs: str = "cdn",
        *,
        auto_open: bool = False,
        add_logo: bool = True,
    ) -> str:
        """Mock plot_html to return the plotfile path as string."""
        return str(plotfile)

    monkeypatch.setattr(am, "plot_html", mock_plot_html)


class DummyFrame:
    """Stub for OpenFrame to test attribution_area."""

    def __init__(
        self, constituents: list[DummySeries], value_ret: list[float]
    ) -> None:
        """Initialize with constituents and returns.

        Args:
            constituents: List of DummySeries objects.
            value_ret: List of return values.

        """
        self.constituents = constituents
        self._value_ret = value_ret
        self.tsdf = pd.DataFrame()
        self._value_ret_series = pd.Series(self._value_ret, index=["test"])

    def from_deepcopy(self) -> Self:
        """Return a copy of self."""
        return self

    def merge_series(self, how: str) -> Self:
        """Merge series with specified method."""
        return self

    def value_nan_handle(self, method: str) -> Self:
        """Handle NaN values with specified method."""
        return self

    @property
    def value_ret(self) -> pd.Series:
        """Return value returns as a pandas Series."""
        return self._value_ret_series

    @value_ret.setter
    def value_ret(self, value: pd.Series) -> None:
        """Set value returns Series.

        Args:
            value: New value returns Series.

        """
        self._value_ret_series = value

    @staticmethod
    def plot_series(
        tick_fmt: str,  # noqa: ARG004
        directory: Path,
        filename: str,
        output_type: Literal["file", "div"] = "file",  # noqa: ARG004
        *,
        add_logo: bool,  # noqa: ARG004
        auto_open: bool,  # noqa: ARG004
    ) -> tuple[DummyFigure, Path]:
        """Plot series and return figure and filepath."""
        return DummyFigure(), directory / filename


def test_attribution_area(tmp_path: Path, monkeypatch: Any, mock_plotly: Any) -> None:
    """Test attribution_area returns figure and correct file path."""
    dates = [dt.date(2025, 1, i + 1) for i in range(3)]
    dataframe = pd.DataFrame({0: [1.0, 2.0, 3.0]}, index=pd.DatetimeIndex(dates))
    dummy_series = DummySeries(label="S", tsdf=dataframe)
    dummy_frame = DummyFrame(constituents=[dummy_series], value_ret=[0.1])

    monkeypatch.setattr(am, "concat", lambda dfs, axis: dataframe)  # noqa: ARG005

    fig, path_ret = am.attribution_area(
        data=dummy_frame,
        series=dummy_series,
        filename="out",
        title="T",
        tick_fmt=".1%",
        directory=tmp_path,
        values_in_legend=True,
        add_logo=False,
        auto_open=False,
    )

    msg1 = "attribution_area did not return DummyFigure"
    if not isinstance(fig, DummyFigure):
        raise AttributionTestError(msg1)

    # Verify figure contains expected traces
    fig_dict = fig.to_dict()
    msg2 = "Figure missing area traces"
    if not any("stackgroup" in trace for trace in fig_dict["data"]):
        raise AttributionTestError(msg2)

    msg3 = "Figure missing scatter trace"
    if not any(trace.get("type") == "scatter" for trace in fig_dict["data"]):
        raise AttributionTestError(msg3)

    expected_path = tmp_path / "out.html"
    msg4 = f"attribution_area path wrong: {path_ret}"
    if path_ret != str(expected_path):
        raise AttributionTestError(msg4)


def test_attribution_waterfall() -> None:
    """Test attribution_waterfall basic functionality."""
    # Create test data with proper index
    value_data = pd.DataFrame({"value": [1.0, 2.0]}, index=["A", "B"])
    series = DummySeries("test", value_data)
    frame = DummyFrame([series], [0.1])
    frame.value_ret = pd.Series([0.1, 0.2], index=["A", "B"])

    # Test basic waterfall
    fig, path = am.attribution_waterfall(
        data=cast("OpenFrame", frame),
        filename="test",
        auto_open=False,
    )

    # Verify figure structure
    fig_dict = fig.to_dict()
    msg1 = "Figure missing waterfall traces"
    if not fig_dict["data"]:
        raise AttributionTestError(msg1)

    msg2 = "Expected path to end with .html"
    if not str(path).endswith(".html"):
        raise AttributionTestError(msg2)

    Path(path).unlink()


def test_attribution_waterfall_custom_dir() -> None:
    """Test attribution_waterfall with custom directory."""
    # Create test data with proper index
    value_data = pd.DataFrame({"value": [1.0, 2.0]}, index=["A", "B"])
    series = DummySeries("test", value_data)
    frame = DummyFrame([series], [0.1])
    frame.value_ret = pd.Series([0.1, 0.2], index=["A", "B"])

    # Test with custom directory
    custom_dir = Path(__file__).parent / "custom"
    custom_dir.mkdir()
    fig, path = am.attribution_waterfall(
        data=cast("OpenFrame", frame),
        filename="test",
        directory=custom_dir,
        auto_open=False,
    )

    msg1 = f"Expected path to contain 'custom', got {path}"
    if "custom" not in str(path):
        raise AttributionTestError(msg1)

    # Verify figure structure
    fig_dict = fig.to_dict()
    msg2 = "Figure missing waterfall traces"
    if not fig_dict["data"]:
        raise AttributionTestError(msg2)

    shutil_rmtree(custom_dir)


def test_attribution_waterfall_title() -> None:
    """Test attribution_waterfall with custom title."""
    # Create test data with proper index
    value_data = pd.DataFrame({"value": [1.0, 2.0]}, index=["A", "B"])
    series = DummySeries("test", value_data)
    frame = DummyFrame([series], [0.1])
    frame.value_ret = pd.Series([0.1, 0.2], index=["A", "B"])

    # Test with custom title
    custom_title = "Custom Title"
    fig, _ = am.attribution_waterfall(
        data=cast("OpenFrame", frame),
        filename="test",
        title=custom_title,
        auto_open=False,
        output_type="div",
    )

    # Verify figure has title in layout
    fig_dict = fig.to_dict()
    msg1 = "Figure missing title in layout"
    if not any("title" in key for key in fig_dict["layout"]):
        raise AttributionTestError(msg1)


def test_attribution_waterfall_tick_fmt() -> None:
    """Test attribution_waterfall uses a custom tick format."""
    value_data = pd.DataFrame({"value": [1.0, 2.0]}, index=["A", "B"])
    series = DummySeries("test", value_data)
    frame = DummyFrame([series], [0.1])
    frame.value_ret = pd.Series([0.1, 0.2], index=["A", "B"])

    fig, path = am.attribution_waterfall(
        data=cast("OpenFrame", frame),
        filename="test_tick",
        tick_fmt=".3%",
        auto_open=False,
        output_type="div",
    )
    yaxis = fig.to_dict()["layout"].get("yaxis", {})
    rec_fmt = yaxis.get("tickformat")
    msg = f"Expected tickformat '.3%', got {rec_fmt}"
    if rec_fmt != ".3%":
        raise AttributionTestError(msg)
    if not path:
        raise AttributionTestError("Waterfall plot returned an empty path")


def test_returns_with_diversification_plot(tmp_path: Path, monkeypatch: Any) -> None:
    """Test diversification plot writes HTML and includes both panels."""
    monkeypatch.setattr(am, "plot_html", lambda **kwargs: str(kwargs["plotfile"]))
    plot_df = pd.DataFrame(
        {
            "CDS": [0.0, 0.01, 0.03],
            "Other instruments": [0.0, -0.005, -0.01],
            "Rolling diversification benefit": [float("nan"), 0.2, 0.4],
        },
        index=["d1", "d2", "d3"],
    )
    fig, path = am.returns_with_diversification_plot(
        plot_df=plot_df,
        filename="div.html",
        title="T",
        diversification_column="Rolling diversification benefit",
        color_min=0.0,
        color_max=1.0,
        directory=tmp_path,
        auto_open=False,
        add_logo=False,
    )
    names = [trace.name for trace in fig.data]
    msg1 = f"Missing expected traces: {names}"
    if "CDS" not in names or "Rolling diversification benefit" not in names:
        raise AttributionTestError(msg1)
    if "mean benefit" not in names:
        raise AttributionTestError(f"Missing mean benefit trace: {names}")
    expected = tmp_path / "div.html"
    msg2 = f"Plot path wrong: {path}"
    if path != str(expected):
        raise AttributionTestError(msg2)


def test_returns_with_diversification_plot_default_column(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Test last column is used when diversification_column is omitted."""
    monkeypatch.setattr(am, "plot_html", lambda **kwargs: str(kwargs["plotfile"]))
    plot_df = pd.DataFrame(
        {"CDS": [0.0, 0.01], "benefit": [0.1, 0.2]},
        index=["d1", "d2"],
    )
    fig, _ = am.returns_with_diversification_plot(
        plot_df=plot_df,
        filename="div.html",
        directory=tmp_path,
        auto_open=False,
    )
    names = [trace.name for trace in fig.data]
    if "benefit" not in names:
        raise AttributionTestError(f"Default column not plotted: {names}")


def test_returns_with_diversification_plot_missing_column() -> None:
    """Test a missing diversification column raises ValueError."""
    plot_df = pd.DataFrame({"CDS": [0.0, 0.01]}, index=["d1", "d2"])
    raised = False
    try:
        am.returns_with_diversification_plot(
            plot_df=plot_df,
            filename="div.html",
            diversification_column="missing",
            auto_open=False,
        )
    except ValueError:
        raised = True
    if not raised:
        raise AttributionTestError("Missing column did not raise ValueError")


def test_returns_with_diversification_plot_no_return_series() -> None:
    """Test a diversification-only DataFrame raises ValueError."""
    plot_df = pd.DataFrame({"benefit": [0.1, 0.2]}, index=["d1", "d2"])
    raised = False
    try:
        am.returns_with_diversification_plot(
            plot_df=plot_df,
            filename="div.html",
            diversification_column="benefit",
            auto_open=False,
        )
    except ValueError:
        raised = True
    if not raised:
        raise AttributionTestError("No return series did not raise ValueError")


def test_returns_with_diversification_plot_home_documents(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Test default directory uses ~/Documents when it exists."""
    documents = tmp_path / "Documents"
    documents.mkdir()
    monkeypatch.setattr(am.Path, "home", classmethod(lambda _cls: tmp_path))
    monkeypatch.setattr(am, "plot_html", lambda **kwargs: str(kwargs["plotfile"]))
    plot_df = pd.DataFrame(
        {"CDS": [0.0, 0.01], "benefit": [0.1, 0.2]},
        index=["d1", "d2"],
    )
    _, path = am.returns_with_diversification_plot(
        plot_df=plot_df,
        filename="div.html",
        auto_open=False,
    )
    expected = documents / "div.html"
    if path != str(expected):
        msg = f"Expected Documents path {expected}, got {path}"
        raise AttributionTestError(msg)


def test_returns_with_diversification_plot_fallback_dir(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Test fallback directory when Documents does not exist."""
    missing_home = tmp_path / "nohome"
    missing_home.mkdir()
    monkeypatch.setattr(am.Path, "home", classmethod(lambda _cls: missing_home))
    monkeypatch.setattr(am, "plot_html", lambda **kwargs: str(kwargs["plotfile"]))
    plot_df = pd.DataFrame(
        {"CDS": [0.0, 0.01], "benefit": [0.1, 0.2]},
        index=["d1", "d2"],
    )
    _, path = am.returns_with_diversification_plot(
        plot_df=plot_df,
        filename="div.html",
        auto_open=False,
    )
    if not str(path).endswith("div.html"):
        msg = f"Fallback path unexpected: {path}"
        raise AttributionTestError(msg)
