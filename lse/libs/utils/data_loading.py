import base64
import io
import logging

import pandas as pd
from pandas.errors import ParserError

logger = logging.getLogger()


def _massage_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = df[col].apply(lambda x: float(x.replace(",", ".")))
            logger.debug(f"Casting {df[col]} string to float")
        except (ValueError, AttributeError):
            continue

    return df


def parse_spreadsheet_contents(content: str, filename: str) -> pd.DataFrame:
    if "csv" in filename:
        parser_fct = pd.read_csv
    elif "xls" in filename:
        parser_fct = pd.read_excel
    else:
        raise Exception(f"{filename} is not supported")

    content_type, content_string = content.split(",")
    decoded = base64.b64decode(content_string)
    try:
        df = parser_fct(io.StringIO(decoded.decode("utf-8")), sep=None)
    except ParserError:
        df = parser_fct(io.StringIO(decoded.decode("utf-8")), sep=";")
    except Exception as exc:
        logger.exception("Error reading spreadsheet")
        raise Exception(f"Could not read any data from the {filename} file") from exc

    if len(df.columns) == 1:
        df = parser_fct(io.StringIO(decoded.decode("utf-8")), sep=";")

    return _massage_dataframe(df)
