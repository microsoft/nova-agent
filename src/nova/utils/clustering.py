import scanpy as sc

from nova.utils.summarize import print_log


def reduce_data(
    adata: sc.AnnData,
    wsi_name: str,
    log: list[str],
    scale: bool = True,
    compute_neighbors: bool = True,
    neighbors_key_added: str = 'neighbors',
    compute_pca: bool = False,
    pca_key_added: str = 'X_pca',
    compute_umap: bool = False,
    umap_key_added: str = 'X_umap',
    compute_tsne: bool = False,
    tsne_key_added: str = 'X_tsne',
) -> tuple[list[str], sc.AnnData]:
    """
    Scale data, compute UMAP, PCA, neighbors, and t-SNE for the given AnnData object.
    Stores results in the adata object under specified keys.
    """
    if scale:
        try:
            # in place adata X is scaled
            sc.pp.scale(adata)
            log.append(f"\n Successfully scaled features for WSI '{wsi_name}' in place.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to scale features for WSI '{wsi_name}'. "
                f"Ensure the adata object contains valid numeric features. Original error: {e}"
                f"Operation log: {print_log(log)}"
            ) from e

    if compute_umap or compute_tsne:
        if not compute_neighbors:
            compute_neighbors = True
            log.append("\n Neighbors will be computed as PCA, UMAP, and t-SNE require neighbors graph.")

    if compute_neighbors:
        try:
            sc.pp.neighbors(
                adata=adata,
                key_added=neighbors_key_added,
            )
            log.append(
                f"\n Successfully computed neighbors for WSI '{wsi_name}' and stored in '{neighbors_key_added}' in adata.obsm."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute neighbors for WSI '{wsi_name}'. "
                f"Ensure the data is suitable for neighbor graph computation. Original error: {e}"
                f"Operation log: {print_log(log)}"
            ) from e

    if compute_pca:
        try:
            sc.pp.pca(
                data=adata,
                key_added=pca_key_added,
            )
            log.append(
                f"\n Successfully computed PCA for WSI '{wsi_name}' and stored in '{pca_key_added}' in adata.obsm."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute PCA for WSI '{wsi_name}'. "
                f"Check that the scaled features are present and suitable for PCA. Original error: {e}"
                f"Operation log: {print_log(log)}"
            ) from e

    if compute_umap:
        try:
            sc.tl.umap(adata=adata, key_added=umap_key_added)
            log.append(
                f"\n Successfully computed UMAP for WSI '{wsi_name}' and stored in '{umap_key_added}' in adata.obsm"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute UMAP for WSI '{wsi_name}'. Original error: {e}Operation log: {print_log(log)}"
            ) from e

    if compute_tsne:
        try:
            sc.tl.tsne(adata=adata, key_added=tsne_key_added)
            log.append(
                f"\n Successfully computed t-SNE for WSI '{wsi_name}' and stored in '{tsne_key_added}' in adata.obsm."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute t-SNE for WSI '{wsi_name}'. "
                f"Ensure the neighbor graph is present and the data is suitable for t-SNE embedding. Original error: {e}"
                f"Operation log: {print_log(log)}"
            ) from e

    return log, adata


def run_leiden_clustering(
    adata: sc.AnnData,
    leiden_key: str,
    leiden_resolution: float,
    wsi_name: str,
    log: list[str],
    neighbors_key_added: str = 'neighbors',
) -> sc.AnnData:
    """
    Run leiden clustering on the given AnnData object.
    Stores results in the adata object under the specified leiden_key.

    Requires the neighbor graph to be computed beforehand.
    """

    if neighbors_key_added not in adata.uns:
        raise RuntimeError(
            f"Neighbors graph not found in adata.uns under key '{neighbors_key_added}'. "
            f"Ensure neighbors are computed before running leiden clustering."
        )

    try:
        sc.tl.leiden(
            adata,
            flavor="igraph",
            resolution=leiden_resolution,
            key_added=leiden_key,
            neighbors_key=neighbors_key_added,
        )
        log.append(
            f"\n Successfully ran Leiden clustering for WSI '{wsi_name}' with resolution {leiden_resolution} and stored in '{leiden_key}' in adata.obs."
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to run Leiden clustering for WSI '{wsi_name}' with resolution {leiden_resolution}. "
            f"Ensure the adata object is properly formatted and contains the necessary data. Original error: {e}"
        ) from e

    return adata
