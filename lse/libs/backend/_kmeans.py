from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from lse.libs.data_models import ClusterType
from lse.libs.exceptions import DataProcessingError
from lse.libs.metaparameters import MAX_N_CLUSTERS


class KMeansClassifier:
    @staticmethod
    def get_kmeans_clusters(
        data: pd.DataFrame,
        n_clusters: int,
    ) -> pd.Series:
        if n_clusters > len(data):
            raise ValueError(
                "The number of clusters can not be higher than the number of samples !!"
            )
        # Kmeans clustering of score
        return pd.Series(
            KMeans(n_clusters=n_clusters, n_init="auto").fit(data).labels_,
            name=ClusterType.PLOT,
            index=data.index,
        ).astype(str)

    @staticmethod
    def get_optimal_n_kmeans_clusters(
        data: pd.DataFrame,
    ) -> int:
        def _fit_model(n_clusters: int) -> float:
            labels = KMeans(n_clusters=n_clusters).fit_predict(data)

            return silhouette_score(data, labels)

        scores = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_fit_model)(n_clusters)
            for n_clusters in range(2, min(MAX_N_CLUSTERS + 1, len(data)))
        )
        if len(scores) == 0:
            raise DataProcessingError("Could not compute optimal number of clusters")

        return int(np.argmax(scores) + 2)
