import base64
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from PIL import Image
from wsidata import open_wsi
from wsidata.io._elems import add_features

import lazyslide as zs
from nova.paths import WSI_REPORT_GEN_PROMPT_PATH
from nova.toolkits.lazyslide.slide_utils import (
    geometry_to_patch_coords,
    patch_idx_to_patch_coord,
)
from nova.toolkits.lazyslide.toolkit_base import (
    PATCH_FEATURES_KEY,
    TILES_KEY,
    TISSUES_KEY,
    LazySlideBaseToolKit,
)
from nova.tools.tiles.captioning import caption_and_summarize_set_of_histology_images_tool
from nova.utils.io import _write_wsi_to_zarr_store
from nova.utils.similarity import compute_similarity
from nova.utils.summarize import print_log, summary_function

warnings.filterwarnings("ignore")

SUPPORTED_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "o4-mini"]


class LazySlideCaptioningToolKit(LazySlideBaseToolKit):
    def visualize_text_prompt_similarity_on_wsi(
        self,
        prompt_term: str,
        text_encoder: str = "conch",
        tissue_key: str = TISSUES_KEY,
        tile_key: str = TILES_KEY,
        tissue_id: int | None = None,
        patch_features_key: str = PATCH_FEATURES_KEY,
        similarity_metric: str = "cosine",
        apply_softmax: bool = True,
        dpi: int = 200,
        cmap: str = "inferno",
        similarity_key_added: str | None = None,
    ) -> dict:
        """
        Visualize similarity between text prompt terms and WSI tile features using a vision-language encoder.

        Args:
            prompt_term (str): Text prompt to visualize as a concept.
            text_encoder (str, optional): Encoder for text embedding. Must be 'conch' or 'plip'.
            tissue_key (str, optional): Key for tissue segmentation. Defaults to "tissues".
            tile_key (str, optional): Key for tile extraction. Defaults to "tissue_tiles".
            tissue_id (Optional[int], optional): If set, only visualize for this tissue region.
            patch_features_key (str, optional): Key for patch features. Must match encoder. Defaults to "uni_tissue_tiles_1024".
            similarity_metric (str, optional): Similarity metric to use ('cosine' or 'dot'). Defaults to 'cosine'.
            apply_softmax (bool, optional): Whether to apply softmax to similarity scores (default True).
            dpi (int, optional): DPI for the saved visualization figure.
            cmap (str, optional): Matplotlib colormap for visualization. Defaults to 'inferno'.

        Returns:
            dict: Dictionary with:
                - 'similarity_map_save_path' (str): Path to the saved similarity figure.
                - 'similarity_scores_csv_save_path' (str): Path to the CSV with per-tile scores.

        """
        log = []

        similarity_map_save_path = self.wsi_job_dir / f"{self.wsi_name}_similarity_to_prompt.jpg"

        # Load WSI for visualization
        try:
            wsi, log = self._load_wsi_with_checks(log)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load WSI for visualization. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        # Prerequisites
        log = self._check_tissue_key(wsi, tissue_key, log)
        log = self._check_tile_key(wsi, tile_key, log)
        log = self._check_patch_features_key(wsi, patch_features_key, log)

        # optionally compute similarity if not provided
        scoring_results = {}
        if similarity_key_added is None:
            try:
                scoring_results = self.score_tiles_in_wsi(
                    classes=[prompt_term],
                    text_encoder=text_encoder,
                    tissue_key=tissue_key,
                    tile_key=tile_key,
                    patch_features_key=patch_features_key,
                    similarity_metric=similarity_metric,
                    apply_softmax=apply_softmax,
                    similarity_key_to_add=f"{patch_features_key}_text_similarity_custom",
                )
                log = scoring_results["log_list"]
                similarity_key_added = scoring_results["similarity_key_to_add"]
                log.append(
                    f"\n Computed similarity scores for prompt: {prompt_term} using score_tiles_in_wsi, storing results in key: {similarity_key_added}."
                )

            except Exception as e:
                raise RuntimeError(
                    f"Failed to compute similarity scores for prompt: {prompt_term}. "
                    f"Original error: {e}. Operations log: {print_log(log)}"
                ) from e

        # Create visualization
        try:
            num_rows = 1
            num_cols = 2

            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), gridspec_kw={'wspace': 0.05}, squeeze=False
            )
            axes = np.atleast_2d(axes)

            # Tissue image (left subplot)
            ax = axes[0, 0]
            zs.pl.tissue(  # type: ignore
                wsi,
                ax=ax,
                tissue_id=tissue_id,
                tissue_key=tissue_key,
                show_contours=False,
                show_id=False,
                mark_origin=True,
                scalebar=True,
                title="Tissue",
            )
            ax.set_title("Tissue", fontsize=14)
            ax.axis('off')

            # Term heatmap (right subplot)
            ax = axes[0, 1]
            zs.pl.tiles(  # type: ignore
                wsi,
                feature_key=similarity_key_added,
                tile_key=tile_key,
                color=prompt_term,
                cmap=cmap,
                show_image=True,
                tissue_id=tissue_id,
                ncols=1,
                alpha=0.7,
                figure=fig,
                ax=ax,
            )
            ax.set_title(prompt_term.capitalize(), fontsize=16)
            ax.axis('off')

            plt.tight_layout()
            fig.savefig(similarity_map_save_path, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
            plt.close(fig)
            log.append(f"\n Saved similarity map to {similarity_map_save_path}.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to visualize similarity map. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        # Close operations
        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            wsi.close()
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {
                "similarity_map_save_path": str(similarity_map_save_path),
                "similarity_scores_csv_save_path": str(scoring_results["similarity_scores_csv_save_path"]),
                "operations_log": print_log(log),
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

    def predict_wsi_label(
        self,
        classes: list[str],
        text_encoder: str = "conch",
        tissue_key: str = TISSUES_KEY,
        tile_key: str = TILES_KEY,
        patch_features_key: str = PATCH_FEATURES_KEY,
        similarity_metric: str = "cosine",
        apply_softmax: bool = True,
        method: str = "titan",
    ) -> dict:
        """
        Predict the label (class) of a Whole Slide Image (WSI) using zero-shot methods.
        """

        log = []

        assert method in {"mi-zero", "titan"}, f"method must be one of 'mi-zero', 'titan'. Got method='{method}'."

        if method == "titan":
            assert "conch_v1.5" in patch_features_key, (
                f"When method is 'titan', patch features must be extracted with 'conch_v1.5' patch encoder."
                f"Your provided patch_features_key is ({patch_features_key}), which does not have conch_v1.5 in the name."
            )

        assert text_encoder in {"conch", "plip"}, (
            f"text_encoder must be 'conch' or 'plip'. Got text_encoder='{text_encoder}'."
        )

        if self.wsi_store_path.exists():
            wsi = open_wsi(self.wsi_path, store=str(self.wsi_store_path))
            log.append(f"\n Loaded WSI successfully from source {self.wsi_store_path}.")
        else:
            raise RuntimeError(
                f'Zarr file not found at {self.wsi_store_path}. This means no processing is done on this WSI!'
            )

        # Prerequisites
        log = self._check_tissue_key(wsi, tissue_key, log)
        log = self._check_tile_key(wsi, tile_key, log)
        log = self._check_patch_features_key(wsi, patch_features_key, log)

        if method == "mi-zero":
            try:
                text_embeddings = zs.tl.text_embedding(classes, model=text_encoder)  # type: ignore
                log.append(f"\n Computed text embeddings for classes: {classes} using text encoder '{text_encoder}'.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to compute text embeddings for classes: {classes}. "
                    f"Original error: {e}. Operations log: {print_log(log)}"
                ) from e

            key_added = f"{patch_features_key}_text_similarity_custom"
            feature_X = wsi.tables[patch_features_key].X

            try:
                similarity_score = compute_similarity(
                    text_embeddings=text_embeddings.values,
                    feature_X=feature_X,
                    metric=similarity_metric,
                    apply_softmax=apply_softmax,
                    softmax_axis=1,
                )
                log.append(f"\n Computed similarity scores using metric: {similarity_metric}.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to compute similarity scores. "
                    f"Ensure that text embeddings and patch features are compatible. "
                    f"Original error: {e}. Operations log: {print_log(log)}"
                ) from e

            try:
                add_features(
                    wsi,
                    key_added,
                    tile_key,
                    similarity_score,
                    var=pd.DataFrame(index=text_embeddings.index),
                )
                log.append(f"\n Added similarity scores to WSI under key: {key_added}.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to add similarity scores to WSI under key '{key_added}' using wsidata.io._elems.add_features "
                    f"Original error: {e}. Operations log: {print_log(log)}"
                ) from e

            try:
                scores = zs.metrics.topk_score(wsi[key_added], agg_method="mean")  # type: ignore
                probs = np.array(scores) / np.sum(scores)
                prob_dict = dict(zip(classes, probs))
                log.append(f"\n Computed top-k scores for classes: {classes}.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to compute top-k scores for classes: {classes}. "
                    f"Original error: {e}. Operations log: {print_log(log)}"
                ) from e

        elif method == "titan":
            try:
                zs.tl.feature_aggregation(  # type: ignore
                    wsi,
                    feature_key=patch_features_key,
                    tile_key=tile_key,
                    encoder=method,
                )
                log.append(
                    f"\n Aggregated patch features into slide features using slide encoder model '{method}' using key: {patch_features_key}."
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to aggregate patch features for method '{method}'. "
                    f"Original error: {e}. Operations log: {print_log(log)}"
                ) from e

            try:
                df_probs = zs.tl.zero_shot_score(  # type: ignore
                    wsi,
                    prompts=classes,  # incorrect type given in lazyslide docs, should be list[str] # type: ignore[arg-type]
                    feature_key=patch_features_key,
                    model=method,
                    device="cpu",
                )
                prob_dict = df_probs.iloc[0].to_dict()
                log.append(f"\n Computed zero-shot scores for classes: {classes} using method '{method}'.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to compute zero-shot scores for classes: {classes} using method '{method}'. "
                    f"Original error: {e}. Operations log: {print_log(log)}"
                ) from e
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: 'mi-zero', 'titan', 'prism'.")

        # do closing operations
        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            wsi.close()
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {
                "probs_for_classes": prob_dict,
                "operations_log": print_log(log),
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

    def generate_wsi_report_with_prism(
        self,
        prompt: list[str],
        tissue_key: str = TISSUES_KEY,
        tile_key: str = TILES_KEY,
        patch_features_key: str = PATCH_FEATURES_KEY,
    ) -> dict[str, str]:
        """
        Generate a Prism-based caption for a Whole Slide Image (WSI).

        Args:
            prompt (list[str]): List of user prompt strings to guide caption generation.
            tissue_key (str, optional): Key for tissue segmentation. Defaults to "tissues".
            tile_key (str, optional): Key for tile extraction. Defaults to "tissue_tiles".
            patch_features_key (str, optional): Key for patch features. Defaults to "uni_tissue_tiles_1024".

        Returns:
            dict: A dictionary containing:
            - 'slide_caption' (str): The generated caption for the WSI.

        Raises:
            RuntimeError: If any prerequisite WSI processing step or caption generation fails.
        """

        log = []

        wsi, log = self._load_wsi_with_checks(log)

        # Prerequisites
        log = self._check_tissue_key(wsi, tissue_key, log)
        log = self._check_tile_key(wsi, tile_key, log)
        log = self._check_patch_features_key(wsi, patch_features_key, log)

        try:
            zs.tl.feature_aggregation(  # type: ignore
                wsi,
                feature_key=patch_features_key,
                tile_key=tile_key,
                encoder="prism",
            )
            log.append(
                f"\n Aggregated patch features into slide features using slide encoder model 'prism' using key: {patch_features_key}."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to aggregate patch features for method 'prism'. "
                f"Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        try:
            results = zs.tl.slide_caption(  # type: ignore
                wsi,
                prompt,
                feature_key=patch_features_key,
                model='prism',
                device='cpu',
            )
            caption = results["caption"][0]
            log.append("\n Generated caption for WSI using Prism model")
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate caption for WSI using Prism model. "
                f"Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        # do closing operations
        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            wsi.close()
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {
                'slide_caption': caption,
                "operations_log": print_log(log),
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict

    def caption_single_wsi(
        self,
        clustering_key: str,
        tissue_key: str = TISSUES_KEY,
        tile_key: str = TILES_KEY,
        patch_features_key: str = PATCH_FEATURES_KEY,
        user_instructions: str | None = None,
        caption_model: str = "gpt-4.1",
        summary_model: str = "o4-mini",
        summary_temperature: float = 1.0,
        n_tiles_to_select: int = 5,
    ) -> dict:
        """ """

        assert caption_model in SUPPORTED_MODELS, (
            f"caption_model must be one of {SUPPORTED_MODELS}. Got caption_model='{caption_model}'."
        )
        assert summary_model in SUPPORTED_MODELS, (
            f"summary_model must be one of {SUPPORTED_MODELS}. Got summary_model='{summary_model}'."
        )

        log = []

        wsi, log = self._load_wsi_with_checks(log)

        # Prerequisites
        log = self._check_tissue_key(wsi, tissue_key, log)
        log = self._check_tile_key(wsi, tile_key, log)
        log = self._check_patch_features_key(wsi, patch_features_key, log)

        # 1. cluster the patches feature space into clusters
        adata = wsi[patch_features_key]
        log = self._check_clustering_key(clustering_key=clustering_key, log=log, adata=adata)  # type: ignore[call-arg]

        # 3. generate captions for each cluster
        cluster_ids = [str(i) for i in range(adata.obs[clustering_key].nunique())]  # type: ignore[call-arg]
        cluster_descriptions = {}
        for cluster_id in cluster_ids:
            # get cluster patches
            cluster_df = adata.obs[adata.obs[clustering_key] == cluster_id]  # type: ignore[call-arg]
            if len(cluster_df) == 0:
                log.append(
                    f"\n No patches found for cluster {cluster_id}. Skipping caption generation for this cluster."
                )
                continue

            n = min(n_tiles_to_select, len(cluster_df))
            log.append(f"\n Selecting {n} patches for cluster {cluster_id}.")
            selected_tiles_for_cluster = cluster_df.sample(n=n, random_state=42)['tile_id'].tolist()
            log.append(f"\n Selected tiles for cluster {cluster_id}: {selected_tiles_for_cluster}")

            # convert patches to coordinates
            try:
                patch_regions = patch_idx_to_patch_coord(
                    wsi,
                    list_of_patch_ids=selected_tiles_for_cluster,
                    patch_features_key=patch_features_key,
                    tile_key=tile_key,
                )
                log.append(f"\n Converted selected patches to coordinates for cluster {cluster_id}.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert selected patches to coordinates for cluster {cluster_id}. "
                    f"Original error: {e}. Operations log: {print_log(log)}"
                ) from e

            # read regions into a list of images
            try:
                patch_images = []
                for patch_region in patch_regions:
                    top_left_x = patch_region["top_left_x"]
                    top_left_y = patch_region["top_left_y"]
                    width = patch_region["base_width"]
                    height = patch_region["base_height"]
                    level = patch_region['base_level']

                    region_arr = wsi.read_region(
                        x=top_left_x,
                        y=top_left_y,
                        width=width,
                        height=height,
                        level=level,
                    )

                    pil_img = Image.fromarray(region_arr)
                    buffer = BytesIO()
                    pil_img.save(buffer, format="PNG")
                    img_bytes = buffer.getvalue()
                    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    patch_images.append(img_base64)

                log.append(f"\n Read {len(patch_images)} images for cluster {cluster_id}.")

            except Exception as e:
                raise RuntimeError(
                    f"Failed to read or encode images for cluster {cluster_id}. "
                    f"Original error: {e}. Operations log: {print_log(log)}"
                ) from e

            try:
                results_dict_for_cluster = caption_and_summarize_set_of_histology_images_tool(
                    list_of_base64_images=patch_images,
                    dense_caption_model_name=caption_model,
                    summary=False,
                )
                # dense_description is a dictionary with keys 'morphological_features', 'similarities', 'variations' which have list[str] as values.
                cluster_descriptions[cluster_id] = results_dict_for_cluster['dense_description']  # type: ignore
                log.append(f"\n Generated caption for cluster {cluster_id}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate caption for cluster {cluster_id}. "
                    f"Original error: {e}. Operations log: {print_log(log)}"
                ) from e

        # 3. generate a summary caption for the WSI using the cluster captions
        try:
            prompt_path = WSI_REPORT_GEN_PROMPT_PATH
            with open(prompt_path, "r") as file:
                prompts = yaml.safe_load(file)

            # fill in the prompt
            n_clusters = len(cluster_descriptions)
            cluster_summaries = "\n".join([f"Cluster {i}: {desc}" for i, desc in cluster_descriptions.items()])
            user_instructions = user_instructions or ''
            system_prompt = prompts['system_prompt']
            user_prompt_filled = prompts['user_prompt'].format(
                n_clusters=n_clusters, cluster_summaries=cluster_summaries, user_query=user_instructions
            )

            report = summary_function(
                summary_model_name=summary_model,
                summary_model_temperature=1.0 if summary_model == "o4-mini" else summary_temperature,
                system_prompt=system_prompt,
                user_prompt=user_prompt_filled,
            )["summary"]
            log.append(f"\n Generated summary report for WSI with {n_clusters} clusters.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate summary report for WSI. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        wsi.close()

        return {'report': report, 'operations_log': print_log(log)}

    def score_tiles_in_wsi(
        self,
        classes: list[str],
        text_encoder: str = "conch",
        tissue_key: str = TISSUES_KEY,
        tile_key: str = TILES_KEY,
        patch_features_key: str = PATCH_FEATURES_KEY,
        similarity_metric: str = "cosine",
        apply_softmax: bool = True,
        similarity_key_to_add: str | None = None,
    ) -> dict:
        """
        Score tiles in the WSI using similarity with text prompts.

        Computes similarity scores between patch features and text class prompts,
        then saves the results with patch coordinates to a CSV file.

        Args:
            classes (list[str]): List of class names/prompts to score against.
            text_encoder (str, optional): Text encoder for embeddings. Defaults to "conch".
            tissue_key (str, optional): Key for tissue segmentation. Defaults to "tissues".
            tile_key (str, optional): Key for tile extraction. Defaults to "tissue_tiles".
            patch_features_key (str, optional): Key for patch features. Defaults to "uni_tissue_tiles_1024".
            similarity_metric (str, optional): Similarity metric ("cosine" or "dot"). Defaults to "cosine".
            apply_softmax (bool, optional): Whether to apply softmax to similarity scores. Defaults to True.
            add_features_to_wsi (bool, optional): Whether to add similarity scores to WSI as features. Defaults to False.

        Returns:
            dict: A dictionary containing:
                - 'similarity_scores_csv_save_path' (str): Path to the saved CSV file with scores.
                - 'tiles_with_scores' (pd.DataFrame): DataFrame with tile information and similarity scores.
                - 'text_embeddings' (pd.DataFrame): Text embeddings used for scoring.
                - 'similarity_scores' (np.ndarray): Raw similarity scores matrix.
                - 'key_added' (str, optional): Key under which features were added to WSI (if add_features_to_wsi=True).
                - 'operations_log' (str): Log of operations performed.

        Raises:
            RuntimeError: If any step in the scoring process fails.
        """
        log = []

        # Validate encoder and patch feature key
        assert text_encoder in {"conch", "plip"}, "text_encoder must be 'conch' or 'plip'"
        if text_encoder not in patch_features_key:
            raise RuntimeError(
                f"Provided patch_features_key '{patch_features_key}' does not contain '{text_encoder}'. "
                "Ensure that features are extracted with the selected vision-language encoder."
            )

        similarity_scores_csv_save_path = self.wsi_job_dir / f"{self.wsi_name}_similarity_with_prompt_scores.csv"

        wsi, log = self._load_wsi_with_checks(log)

        # Prerequisite: Check all required data exists
        log = self._check_tissue_key(wsi, tissue_key, log)
        log = self._check_tile_key(wsi, tile_key, log)
        log = self._check_patch_features_key(wsi, patch_features_key, log)

        # Compute text embeddings for classes
        try:
            text_embeddings = zs.tl.text_embedding(classes, model=text_encoder)  # type: ignore
            log.append(f"\n Computed text embeddings for classes: {classes} using text encoder '{text_encoder}'.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute text embeddings for classes: {classes}. "
                f"Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        # Get patch features
        try:
            feature_X = wsi.tables[patch_features_key].X
            log.append(f"\n Accessed patch features under key: {patch_features_key}.")
        except KeyError as e:
            raise RuntimeError(
                f"Failed to access patch features under key '{patch_features_key}'. "
                f"Ensure that patch features are extracted and available in the WSI. "
                f"Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        # Compute similarity scores
        try:
            similarity_score = compute_similarity(
                text_embeddings=text_embeddings.values,
                feature_X=feature_X,
                metric=similarity_metric,
                apply_softmax=apply_softmax,
                softmax_axis=1,
            )
            log.append(f"\n Computed similarity scores using metric: {similarity_metric}.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute similarity scores. "
                f"Ensure that text embeddings and patch features are compatible. "
                f"Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        # Create DataFrame with tile information and similarity scores
        try:
            # Get the tile shapes GeoDataFrame
            tiles_gdf = wsi.shapes[tile_key].copy()

            # Add similarity scores for each class as new columns
            for idx, class_name in enumerate(classes):
                tiles_gdf[class_name] = similarity_score[:, idx]

            # Convert geometries to patch coordinates
            coords_df = (
                tiles_gdf['geometry'].apply(lambda geom: geometry_to_patch_coords(wsi, tile_key, geom)).apply(pd.Series)
            )
            tiles_gdf = pd.concat([tiles_gdf, coords_df], axis=1)
            tiles_gdf.drop(columns=['geometry'], inplace=True)

            log.append(f"\n Created DataFrame with similarity scores for {len(classes)} classes.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create DataFrame with similarity scores. "
                f"Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        # Optionally add features to WSI
        similarity_key_to_add = similarity_key_to_add or f"{patch_features_key}_text_similarity_custom"
        try:
            add_features(
                wsi,
                similarity_key_to_add,
                tile_key,
                similarity_score,
                var=pd.DataFrame(index=text_embeddings.index),
            )
            log.append(f"\n Added similarity scores to WSI under key: {similarity_key_to_add}.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to add similarity scores to WSI under key '{similarity_key_to_add}' using wsidata.io._elems.add_features "
                f"Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        # Save results to CSV
        try:
            tiles_gdf.to_csv(similarity_scores_csv_save_path, index=False)
            log.append(f"\n Saved similarity scores to CSV at {similarity_scores_csv_save_path}.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to save similarity scores to CSV. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        # Close WSI and write to zarr store
        try:
            _write_wsi_to_zarr_store(wsi, self.wsi_store_path)
            wsi.close()
            log.append(f"\n WSI written to zarr store at {self.wsi_store_path}.")

            asset_dict = {
                "similarity_scores_csv_save_path": str(similarity_scores_csv_save_path),
                "similarity_key_to_add": similarity_key_to_add,
                "operations_log": print_log(log),
                'log_list': log,
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to write WSI to zarr store. Original error: {e}. Operations log: {print_log(log)}"
            ) from e

        return asset_dict
