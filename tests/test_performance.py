"""Pytest suite for performance.py module.

Uses DummyGraphqlClient to mock GraphQL API responses. Targets Python 3.11+
and follows Ruff standards.
"""

import datetime as dt
from typing import TYPE_CHECKING, Any, cast

import pytest
from openseries import OpenTimeSeries

import performance as pm
from graphql_client import GraphqlError
from performance import (
    get_account_performance,
    get_accounts,
    performance_report,
)

if TYPE_CHECKING:
    from graphql_client import GraphqlClient


class PerformanceTestError(Exception):
    """Custom exception to signal test failure in performance tests."""


class DummyGraphqlClient:
    """Stub for GraphqlClient to simulate API responses."""

    def __init__(self, data: Any, error: Any) -> None:
        """Initialize with preset data and error."""
        self._data = data
        self._error = error

    def query(
        self,
        query_string: str,
        variables: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Simulate a GraphQL query returning (data, error)."""
        return self._data, self._error


# --- _select_price_close_timeseries ---


def test_select_price_close_found() -> None:
    """Test _select_price_close_timeseries returns correct item when present."""
    time_series = [
        {"type": "Price(Open)", "dates": [], "values": []},
        {
            "type": "Price(Close)",
            "_id": "ts1",
            "dates": ["2025-01-01"],
            "values": [100.0],
        },
    ]
    result = pm._select_price_close_timeseries(time_series)
    msg = f"Expected Price(Close) item, got {result}"
    if result.get("type") != "Price(Close)":
        raise PerformanceTestError(msg)


def test_select_price_close_missing() -> None:
    """Test _select_price_close_timeseries raises ValueError when absent."""
    time_series = [{"type": "Price(Open)", "dates": [], "values": []}]
    err_msg = "ValueError was not raised for missing Price(Close)"
    try:
        pm._select_price_close_timeseries(time_series)
    except ValueError as e:
        if "Missing Price(Close)" not in str(e):
            msg = f"Expected Missing Price(Close) message: {e}"
            raise PerformanceTestError(msg) from e
        return
    raise PerformanceTestError(err_msg)


# --- get_account_performance ---


def test_get_account_performance_success() -> None:
    """Test get_account_performance returns OpenTimeSeries on valid data."""
    data = {
        "accountPerformance": {
            "currency": "SEK",
            "dates": ["2025-01-01", "2025-01-02"],
            "values": [100.0, 105.0],
            "cashFlows": [0.0, 0.0],
        }
    }
    client = DummyGraphqlClient(data, None)
    result = get_account_performance(
        account_id="acc1",
        gql_client=cast("GraphqlClient", client),
    )
    msg = "Expected OpenTimeSeries, got None or wrong type"
    if not isinstance(result, OpenTimeSeries):
        raise PerformanceTestError(msg)
    label_msg = f"Expected name 'acc1', got {result.label}"
    if result.label != "acc1":
        raise PerformanceTestError(label_msg)


def test_get_account_performance_with_dates() -> None:
    """Test get_account_performance passes start/end dates to query."""
    data = {
        "accountPerformance": {
            "currency": "EUR",
            "dates": ["2025-03-01", "2025-03-02"],
            "values": [200.0, 210.0],
            "cashFlows": [0.0, 0.0],
        }
    }
    client = DummyGraphqlClient(data, None)
    result = get_account_performance(
        account_id="acc2",
        gql_client=cast("GraphqlClient", client),
        start_dt=dt.date(2025, 3, 1),
        end_dt=dt.date(2025, 3, 31),
    )
    msg = "Expected OpenTimeSeries with date filter"
    if not isinstance(result, OpenTimeSeries):
        raise PerformanceTestError(msg)


def test_get_account_performance_graphql_error() -> None:
    """Test get_account_performance raises GraphqlError on API error."""
    client = DummyGraphqlClient(None, "API error")
    not_raised_msg = "GraphqlError was not raised"
    try:
        get_account_performance(
            account_id="acc1",
            gql_client=cast("GraphqlClient", client),
        )
    except GraphqlError as e:
        if "API error" not in str(e):
            msg = f"Expected 'API error' in exception: {e}"
            raise PerformanceTestError(msg) from e
        return
    raise PerformanceTestError(not_raised_msg)


def test_get_account_performance_none_dates() -> None:
    """Test get_account_performance returns None when dates are missing."""
    data = {"accountPerformance": None}
    client = DummyGraphqlClient(data, None)
    result = get_account_performance(
        account_id="acc1",
        gql_client=cast("GraphqlClient", client),
    )
    msg = "Expected None when accountPerformance has no dates"
    if result is not None:
        raise PerformanceTestError(msg)


def test_get_account_performance_empty_portfolio() -> None:
    """Test get_account_performance returns None when portfolio is empty."""
    data = {
        "accountPerformance": {
            "currency": "USD",
            "dates": [],
            "values": [],
            "cashFlows": [],
        }
    }
    client = DummyGraphqlClient(data, None)
    result = get_account_performance(
        account_id="acc1",
        gql_client=cast("GraphqlClient", client),
    )
    msg = "Expected None when portfolio data is empty"
    if result is not None:
        raise PerformanceTestError(msg)


def test_get_account_performance_with_cashflow() -> None:
    """Test get_account_performance computes return correctly with cashflows."""
    data = {
        "accountPerformance": {
            "currency": "SEK",
            "dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "values": [100.0, 50.0, 60.0],
            "cashFlows": [0.0, 50.0, 0.0],
        }
    }
    client = DummyGraphqlClient(data, None)
    result = get_account_performance(
        account_id="acc1",
        gql_client=cast("GraphqlClient", client),
    )
    msg = "Expected OpenTimeSeries with cashflow-adjusted returns"
    if not isinstance(result, OpenTimeSeries):
        raise PerformanceTestError(msg)


# --- _model_index_benchmark_for_account ---


def test_model_index_benchmark_sum_account() -> None:
    """Test _model_index_benchmark_for_account uses modelIndexBenchmark for Sum."""
    account = {
        "type": "Sum",
        "description": "Test Sum",
        "modelIndexBenchmark": {
            "name": "MSCI World",
            "currency": "USD",
            "timeSeries": {
                "items": [
                    {"date": "2025-01-01", "value": 100.0},
                    {"date": "2025-01-02", "value": 101.0},
                ]
            },
        },
    }
    result = pm._model_index_benchmark_for_account(account)
    msg = "Expected OpenTimeSeries from modelIndexBenchmark"
    if not isinstance(result, OpenTimeSeries):
        raise PerformanceTestError(msg)
    if "MSCI World" not in result.label:
        raise PerformanceTestError(
            f"Expected 'MSCI World' in name, got {result.label}"
        )


def test_model_index_benchmark_model_weighted_index() -> None:
    """Test _model_index_benchmark uses account description for ModelWeightedIndex."""
    account = {
        "type": "Sum",
        "description": "My Account",
        "modelIndexBenchmark": {
            "name": "ModelWeightedIndex",
            "currency": "SEK",
            "timeSeries": {
                "items": [
                    {"date": "2025-01-01", "value": 1.0},
                    {"date": "2025-01-02", "value": 1.01},
                ]
            },
        },
    }
    result = pm._model_index_benchmark_for_account(account)
    msg = "Expected 'ModelWeightedIndex (My Account)' in name"
    if result.label != "ModelWeightedIndex (My Account)":
        raise PerformanceTestError(msg)


def test_model_index_benchmark_physical_account() -> None:
    """Test _model_index_benchmark uses main benchmark for Physical accounts."""
    account = {
        "type": "Physical",
        "description": "Physical Account",
        "benchmarks": [
            {
                "mainBenchmark": True,
                "comment": "S&P 500",
                "currency": "USD",
                "offset": 0.0,
                "instrument": {
                    "_id": "inst1",
                    "longName": "S&P 500 Index",
                    "timeSeries": [
                        {
                            "_id": "ts1",
                            "type": "Price(Close)",
                            "dates": ["2025-01-01", "2025-01-02"],
                            "values": [100.0, 101.0],
                        }
                    ],
                },
            },
        ],
        "modelIndexBenchmark": {
            "name": "MIB",
            "currency": "USD",
            "timeSeries": {"items": []},
        },
    }
    result = pm._model_index_benchmark_for_account(account)
    type_msg = "Expected OpenTimeSeries from Physical main benchmark"
    if not isinstance(result, OpenTimeSeries):
        raise PerformanceTestError(type_msg)
    label_msg = f"Expected S&P 500 in name, got {result.label}"
    if "S&P 500" not in result.label and "S&P 500 Index" not in result.label:
        raise PerformanceTestError(label_msg)


def test_model_index_benchmark_physical_fallback_to_model_index() -> None:
    """Test Physical account falls back to modelIndexBenchmark when no main bmk."""
    account = {
        "type": "Physical",
        "description": "Physical No Main Bmk",
        "benchmarks": [
            {
                "mainBenchmark": False,
                "comment": "Non-main benchmark",
                "currency": "USD",
                "offset": 0.0,
                "instrument": {
                    "_id": "inst2",
                    "longName": "Index 2",
                    "timeSeries": [
                        {
                            "_id": "ts2",
                            "type": "Price(Close)",
                            "dates": ["2025-01-01"],
                            "values": [100.0],
                        }
                    ],
                },
            },
        ],
        "modelIndexBenchmark": {
            "name": "Physical MIB",
            "currency": "EUR",
            "timeSeries": {
                "items": [
                    {"date": "2025-01-01", "value": 1.0},
                    {"date": "2025-01-02", "value": 1.01},
                ]
            },
        },
    }
    result = pm._model_index_benchmark_for_account(account)
    msg = "Expected fallback to modelIndexBenchmark for Physical account"
    if result.label != "Physical MIB":
        raise PerformanceTestError(msg)


# --- get_accounts ---


def test_get_accounts_success() -> None:
    """Test get_accounts returns dict of accounts on success."""
    data = {
        "party": {
            "firstTradeDate": "2024-01-01",
            "accounts": [
                {
                    "_id": "acc1",
                    "name": "Account 1",
                    "description": "Desc 1",
                    "type": "Sum",
                    "benchmarks": [],
                    "modelIndexBenchmark": {
                        "name": "BM",
                        "currency": "SEK",
                        "timeSeries": {
                            "items": [
                                {"date": "2025-01-01", "value": 1.0},
                                {"date": "2025-01-02", "value": 1.01},
                            ]
                        },
                    },
                },
            ],
        }
    }
    client = DummyGraphqlClient(data, None)
    result = get_accounts(
        gql=cast("GraphqlClient", client),
        client_id="client123",
    )
    msg = "Expected dict with account acc1"
    if "acc1" not in result:
        raise PerformanceTestError(msg)
    acc = result["acc1"]
    desc_msg = f"Expected description Desc 1, got {acc['description']}"
    if acc["description"] != "Desc 1":
        raise PerformanceTestError(desc_msg)
    mib_msg = "Expected modelIndexBenchmark in account"
    if "modelIndexBenchmark" not in acc:
        raise PerformanceTestError(mib_msg)


def test_get_accounts_graphql_error() -> None:
    """Test get_accounts raises GraphqlError on API error."""
    client = DummyGraphqlClient(None, "query failed")
    not_raised_msg = "GraphqlError was not raised"
    try:
        get_accounts(
            gql=cast("GraphqlClient", client),
            client_id="client123",
        )
    except GraphqlError as e:
        if "query failed" not in str(e):
            msg = f"Expected 'query failed' in exception: {e}"
            raise PerformanceTestError(msg) from e
        return
    raise PerformanceTestError(not_raised_msg)


def test_get_accounts_with_benchmarks() -> None:
    """Test get_accounts includes benchmarks when present."""
    data = {
        "party": {
            "firstTradeDate": "2024-01-01",
            "accounts": [
                {
                    "_id": "acc1",
                    "name": "A1",
                    "description": "Desc",
                    "type": "Sum",
                    "benchmarks": [
                        {
                            "mainBenchmark": False,
                            "comment": "Bmk 1",
                            "currency": "USD",
                            "offset": 0.0,
                            "instrument": {
                                "_id": "i1",
                                "longName": "Index 1",
                                "timeSeries": [
                                    {
                                        "_id": "ts1",
                                        "type": "Price(Close)",
                                        "dates": ["2025-01-01"],
                                        "values": [100.0],
                                    }
                                ],
                            },
                        },
                    ],
                    "modelIndexBenchmark": {
                        "name": "MIB",
                        "currency": "SEK",
                        "timeSeries": {
                            "items": [
                                {"date": "2025-01-01", "value": 1.0},
                            ]
                        },
                    },
                },
            ],
        }
    }
    client = DummyGraphqlClient(data, None)
    result = get_accounts(
        gql=cast("GraphqlClient", client),
        client_id="c1",
    )
    acc = result["acc1"]
    msg = "Expected at least one benchmark in account"
    if len(acc["benchmarks"]) < 1:
        raise PerformanceTestError(msg)


# --- performance_report ---


def test_performance_report_success() -> None:
    """Test performance_report returns OpenFrame on success."""
    party_data = {
        "party": {
            "firstTradeDate": "2024-01-01",
            "accounts": [
                {
                    "_id": "acc1",
                    "name": "A1",
                    "description": "Account One",
                    "type": "Sum",
                    "benchmarks": [],
                    "modelIndexBenchmark": {
                        "name": "MIB",
                        "currency": "SEK",
                        "timeSeries": {
                            "items": [
                                {"date": "2025-01-01", "value": 1.0},
                                {"date": "2025-01-02", "value": 1.02},
                            ]
                        },
                    },
                },
            ],
        }
    }
    perf_data = {
        "accountPerformance": {
            "currency": "SEK",
            "dates": ["2025-01-01", "2025-01-02"],
            "values": [100.0, 102.0],
            "cashFlows": [0.0, 0.0],
        }
    }

    def query_side_effect(
        query_string: str,
        variables: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        if "party" in query_string:
            return party_data, None
        return perf_data, None

    client = DummyGraphqlClient(None, None)
    client.query = query_side_effect  # type: ignore[method-assign]

    frame = performance_report(
        graphql=cast("GraphqlClient", client),
        client="client1",
        start_dt=dt.date(2025, 1, 1),
        end_dt=dt.date(2025, 1, 31),
    )
    frame_msg = "Expected OpenFrame from performance_report"
    if not hasattr(frame, "constituents"):
        raise PerformanceTestError(frame_msg)
    constituents_msg = "Expected at least one constituent in frame"
    if len(frame.constituents) < 1:
        raise PerformanceTestError(constituents_msg)


def test_performance_report_excluded_accounts() -> None:
    """Test performance_report skips excluded accounts but includes others."""
    party_data = {
        "party": {
            "firstTradeDate": "2024-01-01",
            "accounts": [
                {
                    "_id": "excluded_acc",
                    "name": "Excluded",
                    "description": "Excluded Account",
                    "type": "Sum",
                    "benchmarks": [],
                    "modelIndexBenchmark": {
                        "name": "MIB",
                        "currency": "SEK",
                        "timeSeries": {
                            "items": [{"date": "2025-01-01", "value": 1.0}],
                        },
                    },
                },
                {
                    "_id": "included_acc",
                    "name": "Included",
                    "description": "Included Account",
                    "type": "Sum",
                    "benchmarks": [],
                    "modelIndexBenchmark": {
                        "name": "MIB2",
                        "currency": "SEK",
                        "timeSeries": {
                            "items": [
                                {"date": "2025-01-01", "value": 1.0},
                                {"date": "2025-01-02", "value": 1.01},
                            ]
                        },
                    },
                },
            ],
        }
    }
    perf_data = {
        "accountPerformance": {
            "currency": "SEK",
            "dates": ["2025-01-01", "2025-01-02"],
            "values": [100.0, 101.0],
            "cashFlows": [0.0, 0.0],
        }
    }

    def query_side_effect(
        query_string: str,
        variables: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        if "party" in query_string:
            return party_data, None
        return perf_data, None

    client = DummyGraphqlClient(None, None)
    client.query = query_side_effect  # type: ignore[method-assign]

    frame = performance_report(
        graphql=cast("GraphqlClient", client),
        client="c1",
        excluded_accounts=["excluded_acc"],
    )
    labels = [c.label for c in frame.constituents]
    excluded_msg = f"Excluded account in frame: {labels}"
    if "Excluded Account" in labels:
        raise PerformanceTestError(excluded_msg)
    included_msg = f"Expected included account in frame: {labels}"
    if "Included Account" not in labels and "MIB2" not in str(labels):
        raise PerformanceTestError(included_msg)


def test_performance_report_graphql_retry() -> None:
    """Test performance_report retries without date filter on GraphqlError."""
    party_data = {
        "party": {
            "firstTradeDate": "2024-01-01",
            "accounts": [
                {
                    "_id": "acc1",
                    "name": "A1",
                    "description": "Acc 1",
                    "type": "Sum",
                    "benchmarks": [],
                    "modelIndexBenchmark": {
                        "name": "MIB",
                        "currency": "SEK",
                        "timeSeries": {
                            "items": [
                                {"date": "2025-01-01", "value": 1.0},
                                {"date": "2025-01-02", "value": 1.01},
                            ]
                        },
                    },
                },
            ],
        }
    }
    perf_data = {
        "accountPerformance": {
            "currency": "SEK",
            "dates": ["2025-01-01", "2025-01-02"],
            "values": [100.0, 101.0],
            "cashFlows": [0.0, 0.0],
        }
    }
    call_count = 0

    def query_side_effect(
        query_string: str,
        variables: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        nonlocal call_count
        if "party" in query_string:
            return party_data, None
        call_count += 1
        if call_count == 1 and variables and variables.get("startDate"):
            return None, "date filter error"
        return perf_data, None

    client = DummyGraphqlClient(None, None)
    client.query = query_side_effect  # type: ignore[method-assign]

    frame = performance_report(
        graphql=cast("GraphqlClient", client),
        client="c1",
        start_dt=dt.date(2025, 1, 1),
        end_dt=dt.date(2025, 1, 31),
    )
    msg = "Expected retry to succeed and produce frame"
    if len(frame.constituents) < 1:
        raise PerformanceTestError(msg)


def test_performance_report_skips_one_account_without_performance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test performance_report skips only accounts without performance."""
    party_data = {
        "party": {
            "firstTradeDate": "2024-01-01",
            "accounts": [
                {
                    "_id": "acc_no_perf",
                    "name": "NoPerf",
                    "description": "No Performance",
                    "type": "Sum",
                    "benchmarks": [],
                    "modelIndexBenchmark": {
                        "name": "MIB1",
                        "currency": "SEK",
                        "timeSeries": {
                            "items": [
                                {"date": "2025-01-01", "value": 1.0},
                            ]
                        },
                    },
                },
                {
                    "_id": "acc_ok",
                    "name": "A1",
                    "description": "Acc 1",
                    "type": "Sum",
                    "benchmarks": [],
                    "modelIndexBenchmark": {
                        "name": "MIB2",
                        "currency": "SEK",
                        "timeSeries": {
                            "items": [
                                {"date": "2025-01-01", "value": 1.0},
                            ]
                        },
                    },
                },
            ],
        }
    }

    def query_side_effect(
        query_string: str,
        variables: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        if "party" in query_string:
            return party_data, None
        perf_data = {
            "accountPerformance": {
                "currency": "SEK",
                "dates": ["2025-01-01"],
                "values": [100.0],
                "cashFlows": [0.0],
            }
        }
        return perf_data, None

    client = DummyGraphqlClient(None, None)
    client.query = query_side_effect  # type: ignore[method-assign]

    def fake_get_account_performance(
        account_id: str,
        gql_client: "GraphqlClient",
        start_dt: dt.date | None = None,
        end_dt: dt.date | None = None,
        *,
        look_through: bool = False,
    ) -> OpenTimeSeries | None:
        if account_id == "acc_no_perf":
            return None
        return OpenTimeSeries.from_arrays(
            name="Acc 1",
            dates=["2025-01-01", "2025-01-02"],
            values=[1.0, 1.01],
        )

    monkeypatch.setattr(
        pm,
        "get_account_performance",
        fake_get_account_performance,
    )

    frame = performance_report(
        graphql=cast("GraphqlClient", client),
        client="c1",
        start_dt=dt.date(2025, 1, 1),
        end_dt=dt.date(2025, 1, 31),
    )
    msg = "Expected only accounts with performance to be included in frame"
    labels = [c.label for c in frame.constituents]
    if "No Performance" in labels and "Acc 1" not in labels:
        raise PerformanceTestError(msg)
