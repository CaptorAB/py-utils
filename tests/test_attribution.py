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
from shutil import rmtree as shutil_rmtree
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from openseries import OpenFrame

    from graphql_client import GraphqlClient

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

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
        # noinspection PyUnresolvedReferences
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


# noinspection PyUnusedLocal
def mock_plot(figure_or_data: Any, **kwargs: Any) -> str:  # noqa: ARG001
    """Mock plotly.plot to return a div string."""
    return "<div>Mock Plotly Plot</div>"


@pytest.fixture
def mock_plotly(monkeypatch: Any) -> None:
    """Fixture to mock plotly functionality."""
    monkeypatch.setattr(am, "plot", mock_plot)


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

    # noinspection PyUnusedLocal
    def merge_series(self, how: str) -> Self:  # noqa: ARG002
        """Merge series with specified method."""
        return self

    # noinspection PyUnusedLocal
    def value_nan_handle(self, method: str) -> Self:  # noqa: ARG002
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

    # noinspection PyUnusedLocal
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


def test_attribution_area(tmp_path: Path, monkeypatch: Any, mock_plotly: Any) -> None:  # noqa: ARG001
    """Test attribution_area returns figure and correct file path."""
    dates = [dt.date(2025, 1, i + 1) for i in range(3)]
    dataframe = pd.DataFrame({0: [1.0, 2.0, 3.0]}, index=pd.DatetimeIndex(dates))
    dummy_series = DummySeries(label="S", tsdf=dataframe)
    dummy_frame = DummyFrame(constituents=[dummy_series], value_ret=[0.1])

    monkeypatch.setattr(am, "concat", lambda dfs, axis: dataframe)  # noqa: ARG005

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
    if path_ret != expected_path:
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

    path.unlink()


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
