"""Pytest suite for attribution.py module without using bare asserts.

Defines AttributionTestError for signaling test failures.
Covers get_party_name, get_performance,
compute_grouped_attribution_with_cumulative, and attribution_area.
Targets Python 3.13 and follows Ruff standards.
"""

# ruff: noqa: ANN401

import datetime as dt
import math
from pathlib import Path
from typing import Any, Self

import pandas as pd
import pytest

import attribution as am
from attribution import (
    CannotCompoundReturnError,
    PortfolioValueZeroError,
    UnknownCompoundMethodError,
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

    # noinspection PyUnusedLocal
    def query(self, query_string: str, variables: dict[str, Any]) -> Any:  # noqa: ARG002
        """Simulate a GraphQL query returning (data, error).

        Returns:
            A tuple (data, error).

        """
        return self._data, self._error


@pytest.fixture
def graphql_client_success() -> DummyGraphqlClient:
    """Fixture for a successful DummyGraphqlClient stub.

    Returns:
        DummyGraphqlClient with no error.

    """
    data = {"party": {"longName": "Sample Fund"}}
    return DummyGraphqlClient(data, None)


@pytest.fixture
def graphql_client_error() -> DummyGraphqlClient:
    """Fixture for an error DummyGraphqlClient stub.

    Returns:
        DummyGraphqlClient with an error.

    """
    return DummyGraphqlClient(None, "error occurred")


def test_get_party_name_success(graphql_client_success: DummyGraphqlClient) -> None:
    """Test get_party_name returns correct longName on success."""
    # noinspection PyTypeChecker
    result = am.get_party_name(graphql=graphql_client_success, party_id="id123")
    msg = f"Expected 'Sample Fund', got '{result}'"
    if result != "Sample Fund":
        raise AttributionTestError(msg)


def test_get_party_name_error(graphql_client_error: DummyGraphqlClient) -> None:
    """Test get_party_name raises GraphqlError on API error."""
    raised = False
    try:
        # noinspection PyTypeChecker
        am.get_party_name(graphql=graphql_client_error, party_id="id123")
    except am.GraphqlError:
        raised = True
    msg = "GraphqlError was not raised for error response"
    if not raised:
        raise AttributionTestError(msg)


def test_get_performance_success() -> None:
    """Test get_performance returns performance2 dict on success."""
    payload = {
        "performance2": {
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
    # noinspection PyTypeChecker
    result = am.get_performance(graphql=client, client_id="c1")
    expected_series = [0.0]
    rec_series = result.get("series")
    msg = f"Expected series {expected_series}, got {rec_series}"
    if rec_series != expected_series:
        raise AttributionTestError(msg)


def test_get_performance_error() -> None:
    """Test get_performance raises GraphqlError on API error."""
    client = DummyGraphqlClient(None, "fetch failed")
    raised = False
    try:
        # noinspection PyTypeChecker
        am.get_performance(graphql=client, client_id="c2")
    except am.GraphqlError:
        raised = True
    msg = "GraphqlError was not raised on performance fetch error"
    if not raised:
        raise AttributionTestError(msg)


@pytest.fixture
def sample_data() -> dict[str, Any]:
    """Fixture providing sample data for attribution tests.

    Returns:
        A dict with dates, series, and instrumentPerformances.

    """
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
    return {"dates": dates, "series": series, "instrumentPerformances": [perf1, perf2]}


def test_compute_simple_method(sample_data: dict[str, Any]) -> None:
    """Test simple method sums daily contributions correctly."""
    daily, cumu, total = am.compute_grouped_attribution_with_cumulative(
        data=sample_data,
        group_by="modelType",
        group_values=["G1"],
        method="simple",
    )

    expected_g1 = (120.0 - 100.0) / 300.0
    expected_other = (180.0 - 200.0) / 300.0

    msg1 = f"G1 daily wrong: {daily['G1'][1]}"
    # noinspection PyTypeChecker
    if not math.isclose(daily["G1"][1]["value"], expected_g1, rel_tol=1e-9):
        raise AttributionTestError(msg1)

    msg2 = f"Other daily wrong: {daily['Other'][1]}"
    # noinspection PyTypeChecker
    if not math.isclose(daily["Other"][1]["value"], expected_other, rel_tol=1e-9):
        raise AttributionTestError(msg2)

    msg3 = f"G1 cumulative wrong: {cumu['G1'][1]}"
    # noinspection PyTypeChecker
    if not math.isclose(cumu["G1"][1]["value"], expected_g1, rel_tol=1e-9):
        raise AttributionTestError(msg3)

    expected_total = [
        {"date": "d1", "value": 0.0},
        {"date": "d2", "value": sample_data["series"][1]},
    ]

    msg4 = f"Total mismatched: {total}"
    if total != expected_total:
        raise AttributionTestError(msg4)


def test_compute_logreturn_method(sample_data: dict[str, Any]) -> None:
    """Test logreturn method compounds via log1p correctly."""
    _, cumu, _ = am.compute_grouped_attribution_with_cumulative(
        data=sample_data,
        group_by="modelType",
        group_values=["G1"],
        method="logreturn",
    )
    expected = (120.0 - 100.0) / 300.0

    msg = f"G1 logreturn wrong: {cumu['G1'][1]}"

    # noinspection PyTypeChecker
    if not math.isclose(cumu["G1"][1]["value"], expected, rel_tol=1e-9):
        raise AttributionTestError(msg)


def test_compute_logreturn_error(sample_data: dict[str, Any]) -> None:
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


def test_compute_carino_menchero_method(sample_data: dict[str, Any]) -> None:
    """Test Carino/Menchero linking for a single period."""
    _, cumu, _ = am.compute_grouped_attribution_with_cumulative(
        data=sample_data,
        group_by="modelType",
        group_values=["G1"],
        method="carino_menchero",
    )
    expected = (120.0 - 100.0) / 300.0

    msg = f"Carino/Menchero wrong: {cumu['G1'][1]}"

    # noinspection PyTypeChecker
    if not math.isclose(cumu["G1"][1]["value"], expected, rel_tol=1e-9):
        raise AttributionTestError(msg)


def test_compute_unknown_method(sample_data: dict[str, Any]) -> None:
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


def test_compute_zero_total_prev(sample_data: dict[str, Any]) -> None:
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

    def update_traces(self, **kwargs: Any) -> None:
        """Stub update_traces method."""

    def add_scatter(self, **kwargs: Any) -> None:
        """Stub add_scatter method."""

    def update_layout(self, **kwargs: Any) -> None:
        """Stub update_layout method."""


class DummyFrame:
    """Stub for OpenFrame-like object used in attribution_area."""

    def __init__(
        self, constituents: list[DummySeries], value_ret: list[float]
    ) -> None:
        """Initialize with constituent series and their returns.

        Args:
            constituents: List of DummySeries.
            value_ret: Corresponding returns for legend.

        """
        self.constituents = constituents
        self.value_ret = value_ret
        self.tsdf: pd.DataFrame

    def from_deepcopy(self) -> Self:
        """Return self for chaining."""
        return self

    # noinspection PyUnusedLocal
    def merge_series(self, how: str) -> Self:  # noqa: ARG002
        """Stub merge_series method."""
        return self

    # noinspection PyUnusedLocal
    def value_nan_handle(self, method: str) -> Self:  # noqa: ARG002
        """Stub value_nan_handle method."""
        return self

    # noinspection PyUnusedLocal
    @staticmethod
    def plot_series(
        tick_fmt: str,  # noqa: ARG004
        directory: Path,
        filename: str,
        *,
        add_logo: bool,  # noqa: ARG004
        auto_open: bool,  # noqa: ARG004
    ) -> Any:
        """Stub plot_series returning a figure and filepath.

        Returns:
            Tuple of (DummyFigure, str(filepath)).

        """
        filepath = directory / filename
        return DummyFigure(), str(filepath)


def test_attribution_area(tmp_path: Path, monkeypatch: Any) -> None:
    """Test attribution_area returns figure and correct file path."""
    dates = [dt.date(2025, 1, i + 1) for i in range(3)]
    dataframe = pd.DataFrame({0: [1.0, 2.0, 3.0]}, index=pd.DatetimeIndex(dates))
    dummy_series = DummySeries(label="S", tsdf=dataframe)
    dummy_frame = DummyFrame(constituents=[dummy_series], value_ret=[0.1])

    monkeypatch.setattr(am, "concat", lambda dfs, axis: dataframe)  # noqa: ARG005
    monkeypatch.setattr(am, "plot", lambda **kwargs: None)  # noqa: ARG005

    # noinspection PyTypeChecker
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
    expected_path = tmp_path / "out.html"

    msg2 = f"attribution_area path wrong: {path_ret}"
    if path_ret != expected_path:
        raise AttributionTestError(msg2)
