from smolagents import tool

from nova.toolkits.trident.toolkit_dataset_processing import TridentProcessingToolkit


@tool
def dataset_of_wsi_tissue_segmentation_tool(
    job_dir: str,
    wsi_source: str,
    skip_errors: bool = False,
    search_nested: bool = False,
    holes_are_tissue: bool = True,
    batch_size: int = 64,
    segmentation_model_name: str = 'grandqc',
    tissue_seg_confidence_thresh: float = 0.5,
    overwrite: bool = False,
    skip_specific_wsi: list[str] | None = None,
    keep_only_these_wsi: list[str] | None = None,
    max_workers: int = 16,
) -> dict:
    """
    Run tissue segmentation on multiple WSIs and return the locations of output files. Optimized to process multiple WSIs, but can be used with selected WSIs as well.

    Tissue segmentation is the first step for all WSI processing pipelines such as patching, feature extraction, etc.
    The tool provides options to control whether holes should be considered tissue or not, threshold for what is tissue,
    and removing artifacts.

    Notes on using this tool:
        - If overwrite, then sets up segmentation_model and runs tissue segmentation job on all valid slides in `wsi_source`
        - When running tissue segmentation job, the tool creates the folder structure:
            - `{job_dir}/contours_geojson`: Directory containing tissue contour files in GeoJSON format (one per slide). File name format: `{wsi_name}.geojson`.
            - `{job_dir}/contours`: Directory containing tissue contours visualization files in JPG format (one per slide). File name format: `{wsi_name}.jpg`.
            - `{job_dir}/thumbnails`: Directory containing the raw WSI thumbnail images without any contour overlays in JPG format (one per slide). File name format: `{wsi_name}.jpg`.
            - `{job_dir}/_config_segmentation.json`: Configuration JSON file for the segmentation job.
            - `{job_dir}/_logs_segmentation.txt`: Log file with progress and errors of the segmentation process for each file.
        - If not overwriting, the tool checks if segmentation geojson results exist at `{job_dir}/contours_geojson/{wsi_name}.geojson` for all `wsi_name` in the `wsi_source`.
            If results exist, it skips the segmentation step and returns path to the directory with geojson contours. Raises an error if any WSI does not have results.
        - 'grandqc' segmentation_model_name does artifact filtering by default. 'hest' segmentation_model_name does not support artifact filtering.
        - Each GeoJSON in the `contours_geojson` directory is a GeoPandas GeoDataFrame with columns `tissue_id` (unique for each detected tissue) and `geometry` (Polygon object storing the geometry of the tissue)
            Note: the coordinates stored in `geometry` are with respect with level_0 magnification in the respective WSI and must be converted appropriately.
            Note: the `tissue_id` column is used to uniquely identify each tissue region across the entire WSI.
            Note: each independent contiguous tissue region is on a different row in the GeoDataFrame.

    Prerequisites to run this tool:
        - `job_dir` must exist and be writable.
        - `wsi_source` must exist and contain valid WSI files.

    The tool returns:
        dict: Dictionary containing paths to the segmentation outputs:
                - 'dir_with_geojson_contours': Directory of actual tissue contour coordinates files (.geojson per WSI).
                - 'dir_with_tissue_contours_jpg': Directory of contour images overlayed on WSIs (.jpg per WSI).
                - 'dir_with_slide_thumbnails': Directory of raw slide thumbnail images (.jpg per WSI).
                - 'tissue_segmentation_log_file': Absolute path to log file with details of the segmentation process.
                - 'tissue_segmentation_config_file': Absolute path to configuration JSON file for the segmentation job.
                - 'number_of_processed_segmentations': Number of files processed successfully processed.
                - 'operation_log': Formatted log messages from the segmentation process.

    Args:
        job_dir: Path to the job directory where intermediate and output files will be stored.
        wsi_source: Path to the directory containing input WSI files.
        skip_errors: Optional. If True, processing will skip over WSI files that cause errors. Default: False (code crashes with error at first error).
        search_nested: Optional. If True, WSI source is searched recursively. If False, only the top-level is searched. Default: False.
        holes_are_tissue: Optional. If True, holes within tissue regions are considered tissue. Default: True.
        batch_size: Optional. Batch size for WSI tile processing. Default: 64.
        segmentation_model_name: Optional. Name of the tissue segmentation model. Options: ['grandqc', 'hest']. Default: 'grandqc'.
        tissue_seg_confidence_thresh: Optional. Tissue is predicted if probability of tissue in image is greater than or equal to this threshold. Default: 0.5.
        overwrite: Optional. If True then runs the segmentation job. If False then checks if geojson results exists and returns paths to existing results. Default: False.
        skip_specific_wsi: Optional. List of specific WSI names without extension to skip during processing.
                            If provided with overwrite=True, then these will be removed manually from the list of WSIs to process. No results for these will exist.
                            If provided with overwrite=False, then these will be skipped from the check for existing segmentation results.
                            If None, no WSIs are skipped. Default: None (all files in `wsi_source` considered).
        keep_only_these_wsi: Optional. List of WSI names without extension to keep for processing. If None, all WSIs are kept after `skip_specific_wsi` filtering.
                            Default: None (no filtering).
        max_workers: Optional. Number of data loading workers. Default: 16.
    """

    # init the processor
    tridentoolkit_instance = TridentProcessingToolkit(
        job_dir=job_dir,
        wsi_source=wsi_source,
        search_nested=search_nested,
        skip_errors=skip_errors,
        max_workers=max_workers,
        skip_specific_wsi=skip_specific_wsi,
        keep_only_these_wsi=keep_only_these_wsi,
    )

    asset_dict = tridentoolkit_instance.run_tissue_segmentation_job(
        holes_are_tissue=holes_are_tissue,
        batch_size=batch_size,
        segmentation_model_name=segmentation_model_name,
        tissue_seg_confidence_thresh=tissue_seg_confidence_thresh,
        device='cuda:0',
        overwrite=overwrite,
    )

    return asset_dict


@tool
def dataset_of_wsi_patch_coordinate_extraction_tool(
    job_dir: str,
    wsi_source: str,
    skip_errors: bool = False,
    search_nested: bool = False,
    target_magnification: int = 20,
    patch_size: int = 512,
    overlap: int = 0,
    visualize: bool = True,
    min_tissue_proportion: float = 0.95,
    overwrite: bool = False,
    skip_specific_wsi: list[str] | None = None,
    saveto_folder: str | None = None,
    keep_only_these_wsi: list[str] | None = None,
    max_workers: int = 16,
) -> dict:
    """
    Extract patch coordinates from a directory of multiple WSIs after tissue segmentation. Optimized to process multiple WSIs, but can be used with selected WSIs as well.

    This tool generates patch coordinate files for downstream tasks such as patch-level feature extraction
    and slide-level feature aggregation. It does not save raw tile images; instead, the top-left coordinates,
    patch size, and magnification level are stored. Visualization images with patch overlays can optionally be saved.

    Notes on using this tool:
        - If `overwrite=True`, the patch coordinate extraction job is run for all valid slides, replacing any existing outputs.
        - If `overwrite=False`, the tool checks if results already exist at
          `{job_dir}/{saveto_folder}/patches/{slide_name}_patches.h5` for each slide. If any are missing, it raises an error.
        - Patches are only saved if they contain at least `min_tissue_proportion` tissue content.
        - If `visualize=True`, the tool creates visualization images overlaying extracted patch coordinates on WSIs thumbnails.
        - Each H5 file stores patch coordinates under the key `coords`.
        - Either `saveto_folder` or all of `target_magnification`, `patch_size`, and `overlap` must be specified.
            - if `saveto_folder` is None, then the tool constructs saveto_folder as `{target_magnification}x_{patch_size}px_{overlap}px_overlap`
            - Preference is to is to set saveto_folder as `{target_magnification}x_{patch_size}px_{overlap}px_overlap`

    Directory structure & files created:
        - `{job_dir}/{saveto_folder}/patches/`: Directory containing patch coordinate files in H5 format.
              File name format: `{slide_name}_patches.h5`.
              Each file contains the dataset `coords`, which is an array of patch top-left coordinates (x, y). Shape of 'coords' is (num_patches, 2)
        - `{job_dir}/{saveto_folder}/visualization/`: Directory containing visualization images of patches overlaid on WSIs thumbnails.
              File name format: `{slide_name}.jpg`.
              These images are generated only if `visualize=True`.
        - `{job_dir}/{saveto_folder}/_config_coords.json`: Configuration JSON file storing parameters of the patching job.
        - `{job_dir}/{saveto_folder}/_logs_coords.txt`: Log file with details of the patch coordinate extraction process
              (progress, warnings, and errors).

    Prerequisites to run this tool:
        - Tissue segmentation must already be completed for all slides in `wsi_source`. Patches are extracted only from tissue regions.
        - `job_dir` must exist and be writable.
        - `wsi_source` must exist and contain valid WSI files.

    The tool returns:
        dict: Dictionary containing paths to patch extraction outputs:
            - 'dir_with_visualization': Directory with visualization JPGs if `visualize=True`, else None.
            - 'patch_job_config_file': Absolute path to the JSON configuration file for the job.
            - 'patch_job_log_file': Absolute path to the log file for the patch coordinate extraction job.
            - 'patches_key_in_h5': Name of the dataset key in H5 files ('coords').
            - 'patches_saved_dir': Directory containing patch coordinate files in H5 format.
            - 'operation_log': Formatted log messages from the patch coordinate extraction process.

    Args:
        job_dir: Path to the job directory where intermediate and output files will be stored.
        wsi_source: Path to the directory containing input WSI files.
        skip_errors: Optional. If True, processing skips over WSI files that cause errors. Default: False (crashes on error).
        search_nested: Optional. If True, recursively searches for WSIs in `wsi_source`. Default: False (top-level only).
        target_magnification: Optional. Magnification level at which patches are extracted. Default: 20.
        patch_size: Optional. Size of extracted patches in pixels (square). Default: 512.
        overlap: Optional. Overlap between adjacent patches in pixels. Default: 0.
        visualize: Optional. If True, generates visualization images (.jpg) with patches overlaid on WSIs. Default: True.
        min_tissue_proportion: Optional. Minimum fraction (0-1) of tissue required within a patch to keep the patch. Default: 0.95.
        overwrite: Optional. If True, runs patch extraction and overwrites existing results.
                   If False, checks for existing patch files and raises an error if any are missing. Default: False.
        skip_specific_wsi: Optional. List of specific WSI names (without extension) to skip.
                           If `overwrite=True`, these WSIs are excluded from processing.
                           If `overwrite=False`, they are excluded from the results check. Default: None.
        saveto_folder: Optional. Subdirectory under `job_dir` where outputs are saved.
                       If None, the tool constructs saveto_folder as `{target_magnification}x_{patch_size}px_{overlap}px_overlap`.
        keep_only_these_wsi: Optional. List of WSI names (without extension) to keep. If None, all WSIs are kept
                             after applying `skip_specific_wsi`. Default: None (all files in `wsi_source` are processed).
        max_workers: Optional. Number of data loading workers. Default: 16.
    """

    # init the processor
    tridentoolkit_instance = TridentProcessingToolkit(
        job_dir=job_dir,
        wsi_source=wsi_source,
        search_nested=search_nested,
        skip_errors=skip_errors,
        max_workers=max_workers,
        skip_specific_wsi=skip_specific_wsi,
        keep_only_these_wsi=keep_only_these_wsi,
    )

    # run patch coordinate extraction
    asset_dict = tridentoolkit_instance.run_patch_coordinate_extraction_job(
        target_magnification=target_magnification,
        patch_size=patch_size,
        overlap=overlap,
        visualize=visualize,
        min_tissue_proportion=min_tissue_proportion,
        overwrite=overwrite,
        saveto_folder=saveto_folder,
    )

    return asset_dict


@tool
def dataset_of_wsi_patch_features_extraction_tool(
    job_dir: str,
    wsi_source: str,
    skip_errors: bool = False,
    search_nested: bool = False,
    target_magnification: int = 20,
    patch_size: int = 512,
    overlap: int = 0,
    patch_encoder_name: str = "conch_v15",
    batch_limit: int = 512,
    overwrite: bool = False,
    skip_specific_wsi: list[str] | None = None,
    saveto_folder: str | None = None,
    patch_feats_save_folder: str | None = None,
    keep_only_these_wsi: list[str] | None = None,
    max_workers: int = 16,
) -> dict:
    """
    Extract patch-level feature embeddings from multiple WSIs using a patch encoder model. Optimized to process multiple WSIs, but can be used with selected WSIs as well.

    This tool encodes patches into feature vectors, representing each WSI as a bag of patch embeddings
    with shape [num_patches, feature_embedding_dim]. The features are stored in H5 files for downstream
    analysis such as supervised learning and classification.

    Notes on using this tool:
        - Tissue segmentation and patch coordinate extraction must already be completed for all WSIs.
        - If `overwrite=True`, the patch feature extraction job is run for all valid slides, replacing any existing outputs.
        - If `overwrite=False`, the tool checks if feature files already exist at
          `{job_dir}/{saveto_folder}/{patch_feats_save_folder}/{slide_name}.h5` for each slide. If any are missing, it raises an error.
        - Patch features are stored in H5 files under the dataset key `features`.
        - Either `saveto_folder` or all of `target_magnification`, `patch_size`, and `overlap` must be specified.
            - if `saveto_folder` is None, then the tool constructs saveto_folder as `{target_magnification}x_{patch_size}px_{overlap}px_overlap`
            - saveto_folder must exist under job_dir
        - Either `patch_feats_save_folder` or `patch_encoder_name` must be specified.
            - If `patch_feats_save_folder` is None, then the tool constructs it as `features_{patch_encoder_name}`.
            - Preference is given to `features_{patch_encoder_name}`.
        - supported patch encoders are: "uni_v1", "conch_v1", "conch_v15", "resnet50"
        - it is recommended to run the patch encoders with these settings:
            - "uni_v1": target_magnification: 20, patch_size: 256, overlap: any
            - "conch_v1": target_magnification: 20, patch_size: 512, overlap: any
            - "conch_v15": target_magnification: 20, patch_size: 512, overlap: any
            - "resnet50": any combination

    Directory structure & files created:
        - `{job_dir}/{saveto_folder}/{patch_feats_save_folder}/`: Directory containing patch feature files in H5 format.
              File name format: `{slide_name}.h5`.
              Each file contains the dataset `features`, which is an array of patch embeddings with shape `(num_patches, feature_embedding_dim)`.
        - `{job_dir}/{saveto_folder}/_config_feats_{patch_encoder_name}.json`: Configuration JSON file storing parameters of the patch feature extraction job.
        - `{job_dir}/{saveto_folder}/_logs_feats_{patch_encoder_name}.txt`: Log file with details of the patch feature extraction process (progress, warnings, and errors).

    Prerequisites to run this tool:
        - Tissue segmentation and patch coordinate extraction must already be completed for all WSIs.
        - `job_dir` must exist and be writable.
        - `wsi_source` must exist and contain valid WSIs.
        - The user should have access to patch encoder model weights.

    The tool returns:
        dict: Dictionary containing paths to patch feature extraction outputs:
            - 'dir_with_patch_features': Directory containing H5 patch feature files (one per slide).
            - 'patch_features_key_in_h5': Name of the dataset key in H5 files ('features').
            - 'patch_features_job_config_file': Absolute path to the JSON configuration file for the job.
            - 'patch_features_job_log_file': Absolute path to the log file for the patch feature extraction job.
            - 'number_of_patch_features_extracted': Total number of patch feature files successfully generated.
            - 'operation_log': Formatted log messages from the patch feature extraction process.

    Args:
        job_dir: Path to the job directory where intermediate and output files will be stored.
        wsi_source: Path to the directory containing input WSIs.
        skip_errors: Optional. If True, processing skips over WSIs that cause errors. Default: False (crashes on error).
        search_nested: Optional. If True, recursively searches for WSIs in `wsi_source`. Default: False (top-level only).
        target_magnification: Optional. Magnification level at which patches were extracted. Default: 20.
        patch_size: Optional. Size of each square patch in pixels. Default: 512.
        overlap: Optional. Overlap between adjacent patches in pixels. Default: 0.
        patch_encoder_name: Optional. Name of the patch encoder used for feature extraction.
                            Common options: ["conch_v15", "uni_v1"]. Default: "conch_v15".
        batch_limit: Optional. Maximum number of patches processed per batch. Default: 512.
        overwrite: Optional. If True, runs feature extraction and overwrites existing results.
                   If False, checks for existing feature files and raises an error if any are missing. Default: False.
        skip_specific_wsi: Optional. List of specific WSI names (without extension) to skip.
                           If `overwrite=True`, these WSIs are excluded from processing.
                           If `overwrite=False`, they are excluded from the results check. Default: None.
        saveto_folder: Optional. Subdirectory under `job_dir` where outputs are saved.
                       If None, the tool constructs it as `{target_magnification}x_{patch_size}px_{overlap}px_overlap`.
        patch_feats_save_folder: Optional. Subdirectory under `job_dir/{saveto_folder}` where patch feature files are saved.
                                 If None, the tool constructs it as `features_{patch_encoder_name}`.
        keep_only_these_wsi: Optional. List of WSI names (without extension) to keep.
                             If None, all WSIs are kept after applying `skip_specific_wsi`. Default: None (kepp all files in `wsi_source`).
        max_workers: Optional. Number of data loading workers. Default: 16.
    """

    tridentoolkit_instance = TridentProcessingToolkit(
        job_dir=job_dir,
        wsi_source=wsi_source,
        search_nested=search_nested,
        skip_errors=skip_errors,
        max_workers=max_workers,
        skip_specific_wsi=skip_specific_wsi,
        keep_only_these_wsi=keep_only_these_wsi,
    )

    # run patch feature extraction
    asset_dict = tridentoolkit_instance.run_patch_features_extraction_job(
        target_magnification=target_magnification,
        patch_size=patch_size,
        overlap=overlap,
        patch_encoder_name=patch_encoder_name,
        device='cuda:0',
        batch_limit=batch_limit,
        overwrite=overwrite,
        saveto_folder=saveto_folder,
        patch_feats_save_folder=patch_feats_save_folder,
    )

    return asset_dict


@tool
def dataset_of_wsi_slide_features_extraction_tool(
    job_dir: str,
    wsi_source: str,
    skip_errors: bool = False,
    search_nested: bool = False,
    target_magnification: int = 20,
    patch_size: int = 512,
    overlap: int = 0,
    slide_encoder_name: str = "titan",
    patch_encoder_name: str = "conch_v15",
    batch_limit: int = 512,
    overwrite: bool = False,
    skip_specific_wsi: list[str] | None = None,
    saveto_folder: str | None = None,
    slide_feats_save_folder: str | None = None,
    keep_only_these_wsi: list[str] | None = None,
    max_workers: int = 16,
) -> dict:
    """
    Encode multiple WSIs using a slide encoder model into slide representations. Optimized to process multiple WSIs, but can be used with selected WSIs as well.

    This tool aggregates patch embeddings into a single slide representation of shape `(feature_embedding_dim,)`,
    stored in H5 files. Slide-level embeddings are used for downstream analysis such as classification,
    clustering, and survival modeling.

    Notes on using this tool:
        - Tissue segmentation and patch coordinate extraction must already be completed for all WSIs.
        - If patch features are not already extracted, this tool will internally run patch feature extraction
          using the appropriate patch encoder for the given slide encoder.
        - If `overwrite=True`, the slide feature extraction job is run for all valid slides, replacing any existing outputs.
        - If `overwrite=False`, the tool checks if slide feature files already exist at
          `{job_dir}/{saveto_folder}/{slide_feats_save_folder}/{slide_name}.h5`. If any are missing, it raises an error.
        - Slide features are stored in H5 files under the dataset key `features`.
        - Either `saveto_folder` or all of `target_magnification`, `patch_size`, and `overlap` must be specified.
            - If `saveto_folder` is None, then the tool constructs it as `{target_magnification}x_{patch_size}px_{overlap}px_overlap`.
            - `saveto_folder` must exist under `job_dir`.
        - Either `slide_feats_save_folder` or `slide_encoder_name` must be specified.
            - If `slide_feats_save_folder` is None, then the tool constructs it as `features_{slide_encoder_name}`.
            - Preference is given to `features_{slide_encoder_name}`.

    Directory structure & files created:
        - `{job_dir}/{saveto_folder}/{slide_feats_save_folder}/`: Directory containing slide feature files in H5 format.
              File name format: `{slide_name}.h5`.
              Each file contains the dataset `features`, which is a slide embedding with shape `(feature_embedding_dim,)`.
        - `{job_dir}/{saveto_folder}/_config_slide_features_{slide_encoder_name}.json`: Path to configuration JSON file storing parameters of the slide feature extraction job.
        - `{job_dir}/{saveto_folder}/_logs_slide_features_{slide_encoder_name}.txt`: Path to log file with details of the slide feature extraction process (progress, warnings, and errors).

    Prerequisites to run this tool:
        - Tissue segmentation must already exist for all WSIs.
        - Patch coordinates must already exist for all WSIs.
        - Patch features must already exist for all WSIs, or this tool will generate them internally.
        - `job_dir` must exist and be writable.
        - `wsi_source` must exist and contain valid WSIs.
        - User must have access to both patch and slide encoder model weights.
        - Supported slide encoders: `titan`, `prism`, `mean-{patch_encoder_name}`.
            - `titan` can only be used with patch encoder `conch_v15`.
            - `prism` can only be used with patch encoder `virchow`.
            - `mean-{patch_encoder_name}` can be used with any patch encoder (you MUST name it as `mean-{patch_encoder_name}`, ex: `mean-conch_v15`).

    The tool returns:
        dict: Dictionary containing paths to slide feature extraction outputs:
            - 'dir_with_slide_features': Directory containing H5 slide feature files (one per slide).
            - 'slide_features_key_in_h5': Name of the dataset key in H5 files (`features`).
            - 'slide_features_job_config_file': Absolute path to the JSON configuration file for the job.
            - 'slide_features_job_log_file': Absolute path to the log file for the slide feature extraction job.
            - 'number_of_slide_features_extracted': Total number of slide feature files successfully generated.
            - 'operation_log': Formatted log messages from the slide feature extraction process.

    Args:
        job_dir: Path to the job directory where intermediate and output files will be stored.
        wsi_source: Path to the directory containing input WSIs.
        skip_errors: Optional. If True, processing skips over WSIs that cause errors. Default: False (crashes on error).
        search_nested: Optional. If True, recursively searches for WSIs in `wsi_source`. Default: False (top-level only).
        target_magnification: Optional. Magnification level at which patch coordinates and features were computed. Default: 20.
        patch_size: Optional. Size of each square patch in pixels. Default: 512.
        overlap: Optional. Overlap between adjacent patches in pixels. Default: 0.
        slide_encoder_name: Optional. Name of the slide encoder used for slide-level feature extraction.
                            Supported: `titan`, `prism`, `mean-{patch_encoder_name}`. Default: `titan`.
        patch_encoder_name: Optional. Name of the patch encoder used to generate patch features.
                            If not provided, uses the required encoder for the given slide encoder. Default: `conch_v15`.
        batch_limit: Optional. Maximum number of patches processed per batch. Default: 512.
        overwrite: Optional. If True, runs feature extraction and overwrites existing results.
                   If False, checks for existing slide feature files and raises an error if any are missing. Default: False.
        skip_specific_wsi: Optional. List of specific WSI names (without extension) to skip.
                           If `overwrite=True`, these WSIs are excluded from processing.
                           If `overwrite=False`, they are excluded from the results check. Default: None.
        saveto_folder: Optional. Subdirectory under `job_dir` where outputs are saved.
                       If None, the tool constructs it as `{target_magnification}x_{patch_size}px_{overlap}px_overlap`.
        slide_feats_save_folder: Optional. Subdirectory under `job_dir/{saveto_folder}` where slide feature files are saved.
                                 If None, the tool constructs it as `features_{slide_encoder_name}`.
        keep_only_these_wsi: Optional. List of WSI names (without extension) to keep.
                             If None, all WSIs are kept after applying `skip_specific_wsi`. Default: None (process all files in `wsi_source`).
        max_workers: Optional. Number of data loading workers. Default: 16.
    """

    tridentoolkit_instance = TridentProcessingToolkit(
        job_dir=job_dir,
        wsi_source=wsi_source,
        search_nested=search_nested,
        skip_errors=skip_errors,
        max_workers=max_workers,
        skip_specific_wsi=skip_specific_wsi,
        keep_only_these_wsi=keep_only_these_wsi,
    )

    # run slide feature extraction
    asset_dict = tridentoolkit_instance.run_slide_features_extraction_job(
        target_magnification=target_magnification,
        patch_size=patch_size,
        overlap=overlap,
        slide_encoder_name=slide_encoder_name,
        patch_encoder_name=patch_encoder_name,
        device='cuda:0',
        batch_limit=batch_limit,
        overwrite=overwrite,
        saveto_folder=saveto_folder,
        slide_feats_save_folder=slide_feats_save_folder,
    )

    return asset_dict
