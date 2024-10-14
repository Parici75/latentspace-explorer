from __future__ import annotations

from typing import Any

import pandas as pd
from dash import html

from lse.libs.backend import AppBackend
from lse.libs.backend.io import OfflineUserSession, OnlineUserSession
from lse.libs.backend.io.data_models import UserSessionParameters


def build_download_button(uri: str) -> html.Form:
    """Generates a Download button pointing to a uri."""
    button = html.Form(
        action=uri,
        method="get",
        children=[html.Button(type="submit", children=["Download"], className="button")],
    )
    return button


def load_online_session(session_id: str) -> AppBackend:
    """Loads online user session"""

    return AppBackend(user_session=OnlineUserSession(session_id=session_id))


def load_offline_session(
    session_id: str, user_session_parameters: UserSessionParameters
) -> AppBackend:
    """Loads offline user session"""
    return AppBackend(
        user_session=OfflineUserSession(
            session_id=session_id, user_session_parameters=user_session_parameters
        )
    )


def update_data_cache(session_id: str, selected_data: dict[str, Any] | None) -> None:
    app_backend = load_online_session(session_id=session_id)
    load_online_session(session_id).user_session.data_wrapper.cache_data(  # type: ignore
        get_datapoints_from_selection_or_hover(
            data=app_backend.user_session.filtered_data, selected_data=selected_data
        )
    )


def get_datapoints_from_selection_or_hover(
    data: pd.DataFrame, selected_data: dict[str, Any] | None
) -> pd.DataFrame | pd.Series:
    """Selects data points through their {obj}`VariableReference.UNIQUE_ID` stored in Plotly Trace's `custom_data` field."""

    if selected_data is None:
        return data
    if len(selected_data["points"]) == 0:
        return data

    idx = pd.IndexSlice
    datapoint_ids = [datapoint["customdata"] for datapoint in selected_data["points"]]
    data_points = data.loc[idx[datapoint_ids, :]]

    return data_points
