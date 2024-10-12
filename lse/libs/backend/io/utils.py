import io

import pandas as pd

FEATHER_COMPRESSION = "lz4"


def serialize_to_feather(data: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    data.to_feather(buffer, compression=FEATHER_COMPRESSION)
    buffer.seek(0)

    return buffer


def read_feathered_dataframe(data: io.BytesIO) -> pd.DataFrame:
    return pd.read_feather(data)
