import io

from lse.libs.backend.io.utils import serialize_to_feather
from lse.libs.dash_core.callbacks.utils import load_online_session


def _reconstruct_data(
    session_id: str, signature_variables: list[str], var_explained: float
) -> io.BytesIO:
    app_backend = load_online_session(session_id)
    reconstructed_data = app_backend.reconstruct_data(
        signature_variables=signature_variables, var_explained=var_explained
    )

    return serialize_to_feather(reconstructed_data)
