"""Functions to fetch TPT data from the Captor open API."""

from inspect import stack
from pathlib import Path
from re import match as re_match

import requests
from pandas import DataFrame, concat


def sort_key(col_name: str) -> tuple[float | int, str]:
    """Generate a sorting key for column names based on leading numbers and letters.

    Args:
        col_name (str): The column name to extract sorting components from.

    Returns:
        tuple[float | int, str]: A tuple of (number, string) used for sorting. If no
        match is found, returns (inf, original column name).

    """
    match = re_match(r"(\d+)([A-Za-z]*)_", col_name)
    if match:
        number = int(match.group(1))
        letter = match.group(2)
        return number, letter

    return float("inf"), col_name


def replace_small_values(
    x: float | str | None,
    threshold: float = 0.000001,
) -> float | int | str | None:
    """Replace very small float values with zero.

    Args:
        x (float | str | None): Value to check.
        threshold (float): Threshold below which float values are replaced with 0.0.
            Defaults to 0.000001.

    Returns:
        float | int | str | None: Zero if value is small float, otherwise unchanged.

    """
    if isinstance(x, float) and abs(x) < threshold:
        return 0.0
    return x


def download_fund_tpt_report(
    report_id: str,
    directory: Path | None = None,
    timeout: int = 10,
) -> Path | None:
    """Download a fund TPT report as an Excel file.

    Args:
        report_id (str): Unique identifier for the TPT report.
        directory (Path | None): Directory to save the Excel file in. If None, saves
            to ~/Documents or the script's directory.
        timeout (int): Timeout in seconds for the HTTP request. Defaults to 10.

    Returns:
        Path | None: Path to the saved Excel file, or None if saving failed.

    """
    url = f"https://api.captor.se/public/api/tpts/{report_id}"
    response = requests.get(url=url, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    filename = f"{data['name']}.xlsx"

    dataframe = DataFrame(data=data["data"])
    sorted_columns = sorted(dataframe.columns, key=sort_key)
    dataframe = dataframe[sorted_columns].apply(
        lambda col: col.map(replace_small_values)
    )

    if directory:
        dirpath = directory
    elif Path.home().joinpath("Documents").exists():
        dirpath = Path.home().joinpath("Documents")
    else:
        dirpath = Path(stack()[1].filename).parent

    sheetfile = dirpath.joinpath(filename)
    dataframe.to_excel(excel_writer=sheetfile, engine="openpyxl", index=False)

    if sheetfile.exists():
        print(f"\nReport written to Xlsx file: {sheetfile}")  # noqa: T201
        return sheetfile

    return None


def collate_fund_tpt_reports(
    report_ids: list[str],
    sheetfile: Path,
    timeout: int = 10,
) -> Path | None:
    """Download and merge multiple fund TPT reports into one Excel file.

    Args:
        report_ids (list[str]): List of report IDs to download and collate.
        sheetfile (Path): Destination path for the output Excel file.
        timeout (int): Timeout in seconds for each HTTP request. Defaults to 10.

    Returns:
        Path | None: Path to the saved Excel file, or None if saving failed.

    """
    base_url = "https://api.captor.se/public/api/tpts/{}"

    dataframe = DataFrame()

    for report_id in report_ids:
        url = base_url.format(report_id)
        response = requests.get(url=url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        new_df = DataFrame(data["data"])
        sorted_columns = sorted(new_df.columns, key=sort_key)
        new_df = new_df[sorted_columns].apply(
            lambda col: col.map(replace_small_values),
        )
        dataframe = concat(objs=[dataframe, new_df])

    dataframe.to_excel(excel_writer=sheetfile, engine="openpyxl", index=False)

    if sheetfile.exists():
        return sheetfile

    return None


if __name__ == "__main__":
    xlsxpath = Path.home().joinpath("Documents")
    _ = download_fund_tpt_report(
        report_id="67a5ca93079b64d59bb66ccd",
        directory=xlsxpath,
    )
