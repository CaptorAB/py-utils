"""Pytest suite for combined_attribution.py module without using bare asserts.

Defines CombinedAttributionTestError for signaling test failures.
Covers the group merging in _to_frame and trading_day_mask.
"""

from typing import Any

import pandas as pd

import combined_attribution as ca


class CombinedAttributionTestError(Exception):
    """Custom exception to signal test failure in combined attribution tests."""


def _series(dates: list[str], values: list[float]) -> list[dict[str, Any]]:
    """Build a [{date, value}] list as returned by the attribution functions."""
    return [{"date": d, "value": v} for d, v in zip(dates, values, strict=True)]


class TestCombinedAttribution:
    """Test suite for combined attribution helpers."""

    dates = ("2024-01-01", "2024-01-02")

    def test_to_frame_merges_groups(self) -> None:
        """Test that all merged groups are summed and their parts removed."""
        dates = list(self.dates)
        frame = ca._to_frame(
            {
                "CdsIndex": _series(dates, [0.0, 0.10]),
                "Swap": _series(dates, [0.0, 0.20]),
                "Bond": _series(dates, [0.0, 0.30]),
                "Swaption": _series(dates, [0.0, 0.40]),
                "CdsBasket": _series(dates, [0.0, 0.50]),
                "Balance": _series(dates, [0.0, 0.60]),
                "Fees & costs": _series(dates, [0.0, -0.01]),
            }
        )

        expected_columns = ["Cds", "Swap", "Bond", "Fees & costs"]
        if list(frame.columns) != expected_columns:
            msg = f"Expected columns {expected_columns}, got {list(frame.columns)}"
            raise CombinedAttributionTestError(msg)

        expected = {"Cds": 0.60, "Swap": 0.60, "Bond": 0.90, "Fees & costs": -0.01}
        for column, value in expected.items():
            if abs(frame[column].iloc[-1] - value) > 1e-12:  # noqa: PLR2004
                msg = f"Expected {column}={value}, got {frame[column].iloc[-1]}"
                raise CombinedAttributionTestError(msg)

        if not isinstance(frame.index, pd.DatetimeIndex):
            msg = f"Expected DatetimeIndex, got {type(frame.index)}"
            raise CombinedAttributionTestError(msg)

    def test_to_frame_partial_groups(self) -> None:
        """Test a fund holding only some of a merged group's parts."""
        dates = list(self.dates)
        frame = ca._to_frame(
            {
                "Bond": _series(dates, [0.0, 0.30]),
                "Swaption": _series(dates, [0.0, 0.40]),
                "Balance": _series(dates, [0.0, 0.60]),
            }
        )

        expected_columns = ["Bond", "Swap"]
        if list(frame.columns) != expected_columns:
            msg = f"Expected columns {expected_columns}, got {list(frame.columns)}"
            raise CombinedAttributionTestError(msg)

        if abs(frame["Swap"].iloc[-1] - 0.40) > 1e-12:  # noqa: PLR2004
            msg = f"Expected Swap=0.40, got {frame['Swap'].iloc[-1]}"
            raise CombinedAttributionTestError(msg)

    def test_trading_day_mask(self) -> None:
        """Test that only days without any contribution are dropped."""
        daily = pd.DataFrame(
            {
                ("Fund A", "Bond"): [0.0, 0.01, 0.0, 0.0, 0.00001],
                ("Fund B", "Bond"): [0.0, 0.02, 0.0, 0.03, 0.0],
            },
            index=pd.to_datetime(
                [
                    "2024-01-05",  # start date, all zero: kept as base
                    "2024-01-08",
                    "2024-01-06",  # all zero: dropped
                    "2024-01-09",  # only Fund B moves: kept
                    "2024-01-07",  # tiny move on a weekend: kept
                ]
            ),
        )

        mask = ca.trading_day_mask(daily)
        expected = [True, True, False, True, True]
        if mask.tolist() != expected:
            msg = f"Expected mask {expected}, got {mask.tolist()}"
            raise CombinedAttributionTestError(msg)

    def test_trading_day_mask_empty(self) -> None:
        """Test that an empty frame gives an empty mask."""
        mask = ca.trading_day_mask(pd.DataFrame())
        if not mask.empty:
            msg = f"Expected empty mask, got {mask.tolist()}"
            raise CombinedAttributionTestError(msg)
