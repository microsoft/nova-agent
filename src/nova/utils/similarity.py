import numpy as np
from scipy.special import softmax


def compute_similarity(
    text_embeddings: np.ndarray,
    feature_X: np.ndarray,
    metric: str = "dot",
    apply_softmax: bool = False,
    softmax_axis: int = 1,
) -> np.ndarray:
    """
    Compute similarity scores between text and image feature embeddings.

    Supports both dot product and cosine similarity, with an optional softmax normalization
    over the result for interpretability (e.g., for attention heatmaps).

    Args:
        text_embeddings (np.ndarray): Array of shape (n_text, d) with text/concept embeddings.
        feature_X (np.ndarray): Array of shape (n_patches, d) with image/patch embeddings.
        metric (str, optional): Similarity metric to use: "dot" (default) or "cosine".
        apply_softmax (bool, optional): Whether to apply softmax to similarity scores (default: False).
        softmax_axis (int, optional): Axis along which to apply softmax (default: 1).

    Returns:
        np.ndarray: Similarity matrix of shape (n_patches, n_text).

    Raises:
        ValueError: If an unknown similarity metric is requested.

    Notes:
        - For "cosine" similarity, both text and image embeddings are normalized to unit length.
        - For softmax, axis=1 yields per-patch normalized probabilities (recommended for attention).
    """
    if metric == "dot":
        sim = np.dot(feature_X, text_embeddings.T)
    elif metric == "cosine":
        # Avoid division by zero in normalization
        feature_norm = feature_X / np.clip(np.linalg.norm(feature_X, axis=1, keepdims=True), a_min=1e-8, a_max=None)
        text_norm = text_embeddings / np.clip(
            np.linalg.norm(text_embeddings, axis=1, keepdims=True), a_min=1e-8, a_max=None
        )
        sim = np.dot(feature_norm, text_norm.T)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

    if apply_softmax:
        sim = softmax(sim, axis=softmax_axis)

    return sim
