from inspect import stack
from pathlib import Path
from re import match as re_match

import requests
from pandas import DataFrame, concat


class GraphqlException(Exception):
    pass


def sort_key(col_name: str) -> tuple[float | int, str]:
    match = re_match(r"(\d+)([A-Za-z]*)_", col_name)
    if match:
        number = int(match.group(1))
        letter = match.group(2)
        return number, letter

    return float("inf"), col_name


def replace_small_values(
    x: float | str | None, threshold: float = 0.000001
) -> float | int | str | None:
    if isinstance(x, float) and abs(x) < threshold:
        return 0.0
    return x


def download_fund_tpt_report(
    report_id: str, directory: Path | None = None, timeout: int = 10
) -> Path | None:
    url = f"https://api.captor.se/public/api/tpts/{report_id}"
    response = requests.get(url=url, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    filename = f"{data['name']}.xlsx"

    df = DataFrame(data=data["data"])
    sorted_columns = sorted(df.columns, key=sort_key)
    df = df[sorted_columns].apply(lambda col: col.map(replace_small_values))

    if directory:
        dirpath = directory
    elif Path.home().joinpath("Documents").exists():
        dirpath = Path.home() / "Documents"
    else:
        dirpath = Path(stack()[1].filename).parent

    sheetfile = dirpath / filename
    df.to_excel(excel_writer=sheetfile, engine="openpyxl", index=False)

    if sheetfile.exists():
        print(f"\nReport written to Xlsx file: {sheetfile}")
        return sheetfile

    return None


def collate_fund_tpt_reports(
    report_ids: list[str], sheetfile: Path, timeout: int = 10
) -> Path | None:
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
            lambda col: col.map(replace_small_values)
        )
        dataframe = concat(objs=[dataframe, new_df])

    dataframe.to_excel(excel_writer=sheetfile, engine="openpyxl", index=False)

    if sheetfile.exists():
        return sheetfile

    return None


if __name__ == "__main__":
    xlsxpath = Path.home().joinpath("Documents")
    _ = download_fund_tpt_report(
        report_id="67a5ca93079b64d59bb66ccd", directory=xlsxpath
    )
