import numpy as np
import pandas as pd
import networkx as nx
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder

from .preprocessing import make_clustering_embedding


def sample_embedding(embed, sample_n=20000, random_state=42):
    rng = np.random.default_rng(random_state)
    sample_n = min(sample_n, len(embed))
    sample_idx = rng.choice(len(embed), sample_n, replace=False)
    return sample_idx, embed[sample_idx]


def run_clustering_experiment(
    x,
    labels,
    num_cols,
    cat_cols,
    embed_components=64,
    sample_n=20000,
    random_state=42,
):
    embed = make_clustering_embedding(
        x,
        num_cols=num_cols,
        cat_cols=cat_cols,
        n_components=embed_components,
        random_state=random_state,
    )
    sample_idx, embed_sample = sample_embedding(embed, sample_n=sample_n, random_state=random_state)

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state)
    gmm_labels = gmm.fit_predict(embed)
    gmm_labels_sample = gmm_labels[sample_idx]

    graph = kneighbors_graph(embed_sample, n_neighbors=20, mode="connectivity", include_self=False)
    communities = list(nx.community.greedy_modularity_communities(nx.from_scipy_sparse_array(graph.maximum(graph.T))))
    community_labels = np.empty(len(sample_idx), dtype=int)
    for i, nodes in enumerate(communities):
        community_labels[list(nodes)] = i

    summary = pd.DataFrame(
        [
            {
                "method": "GMM",
                "n_clusters": np.unique(gmm_labels_sample).size,
                "silhouette": silhouette_score(embed_sample, gmm_labels_sample),
                "calinski_harabasz": calinski_harabasz_score(embed_sample, gmm_labels_sample),
                "davies_bouldin": davies_bouldin_score(embed_sample, gmm_labels_sample),
            },
            {
                "method": "Community",
                "n_clusters": np.unique(community_labels).size,
                "silhouette": silhouette_score(embed_sample, community_labels),
                "calinski_harabasz": calinski_harabasz_score(embed_sample, community_labels),
                "davies_bouldin": davies_bouldin_score(embed_sample, community_labels),
            },
        ]
    )

    umap_xy = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    ).fit_transform(embed_sample)
    tsne_xy = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(embed_sample)

    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(pd.Series(labels).iloc[sample_idx].astype(str))

    projection = pd.DataFrame(
        {
            "sample_idx": sample_idx,
            "umap_x": umap_xy[:, 0],
            "umap_y": umap_xy[:, 1],
            "tsne_x": tsne_xy[:, 0],
            "tsne_y": tsne_xy[:, 1],
            "gmm_label": gmm_labels_sample,
            "community_label": community_labels,
            "true_label": pd.Series(labels).iloc[sample_idx].astype(str).to_numpy(),
            "true_label_encoded": y_true_encoded,
        }
    )
    return summary, projection, list(label_encoder.classes_)
