from inspect import stack
from pandas import DataFrame
from pathlib import Path
from re import match as re_match
import requests


def download_fund_tpt_report(report_id: str, directory: str | None = None, timeout: int = 10):
    url = f"https://api.captor.se/public/api/tpts/{report_id}"
    response = requests.get(url=url, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    filename = f"{data['name']}.xlsx"

    def sort_key(col_name: str):
        match = re_match(r"(\d+)([A-Za-z]*)_", col_name)
        if match:
            number = int(match.group(1))
            letter = match.group(2)
            return number, letter

        return float("inf"), col_name

    df = DataFrame(data=data["data"])
    sorted_columns = sorted(df.columns, key=sort_key)
    df = df[sorted_columns]

    if directory:
        dirpath = Path(directory).resolve()
    elif Path.home().joinpath("Documents").exists():
        dirpath = Path.home().joinpath("Documents")
    else:
        dirpath = Path(stack()[1].filename).parent

    sheetfile = dirpath.joinpath(filename)

    df.to_excel(excel_writer=sheetfile, engine="openpyxl", index=False)

    if sheetfile.exists():
        print(f"\nReport written to Xlsx file: {sheetfile}")


if __name__ == "__main__":
    rpt_id = "67a5ca34079b64d59bb669df"
    download_fund_tpt_report(report_id=rpt_id)
