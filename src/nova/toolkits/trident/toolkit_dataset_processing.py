from pathlib import Path

from nova.toolkits.trident.toolkit_base import TridentProcessingBaseToolkit
from nova.utils.summarize import print_log
from trident.patch_encoder_models import encoder_factory as patch_encoder_factory
from trident.segmentation_models import segmentation_model_factory
from trident.slide_encoder_models import encoder_factory as slide_encoder_factory

__all__ = [
    'TridentProcessingToolkit',
]


class TridentProcessingToolkit(TridentProcessingBaseToolkit):
    """
    Concrete implementation of TridentProcessingBaseToolkit for comprehensive WSI processing.

    This toolkit provides methods for tissue segmentation, patch coordinate extraction,
    patch feature extraction, slide feature extraction, and visualization generation.
    """

    def get_supported_operations(self) -> list:
        """
        Return a list of operations supported by this toolkit.

        Returns:
            list: List of supported operation names.
        """
        return [
            "tissue_segmentation",
            "patch_coordinate_extraction",
            "patch_feature_extraction",
            "slide_feature_extraction",
            "visualization_generation",
        ]

    def validate_prerequisites(self, operation: str, **kwargs) -> bool:
        """
        Validate that prerequisites for the toolkit are met.

        This method checks if the job directory and WSI source directory exist,
        and if valid WSI files are present in the source directory.
        """
        return True

    def run_tissue_segmentation_job(
        self,
        holes_are_tissue: bool = True,
        batch_size: int = 64,
        segmentation_model_name: str = 'grandqc',
        tissue_seg_confidence_thresh: float = 0.5,
        device: str = 'cuda:0',
        overwrite: bool = False,
    ) -> dict:
        """
        Core function to run tissue segmentation on all slides in a Trident Processor object and return core segmentation assets.

        This function applies a tissue segmentation pipeline to all WSIs in the provided Processor, saving contours,
        thumbnails, configuration, and logs in standard subdirectories under `job_dir`. It supports optional artifact
        removal and is configurable for different models, batch sizes, and compute devices.
        """

        log = []
        SEG_MAG = 10
        log.append(f"Using harcoded seg_mag of {SEG_MAG}")

        if overwrite:
            try:
                segmentation_model = segmentation_model_factory(
                    model_name=segmentation_model_name,
                    confidence_thresh=tissue_seg_confidence_thresh,
                )
                log.append(
                    f'\n Initialized segmentation model: {segmentation_model_name} with confidence threshold: {tissue_seg_confidence_thresh}'
                )
            except Exception as e:
                raise RuntimeError(
                    f"Unable to initialize segmentation model '{segmentation_model_name}'"
                    f'Orignal error: {str(e)}'
                    f'Operation log: {print_log(log)}'
                ) from e

            artifact_remover_model = None

            try:
                _ = self.processor.run_segmentation_job(
                    segmentation_model=segmentation_model,
                    seg_mag=SEG_MAG,
                    holes_are_tissue=holes_are_tissue,
                    batch_size=batch_size,
                    artifact_remover_model=artifact_remover_model,
                    device=device,
                )
                dir_with_geojson_contours = Path(self.job_dir) / 'contours_geojson'
                log.append(f'\n Segmentation job completed. Geojson files saved to: {dir_with_geojson_contours}')
            except Exception as e:
                if self.processor.skip_errors:
                    log.append(
                        '\n Segmentation job failed for some slides but skipping due to skip_errors=True. Rerun with skip_errors=False to see details.'
                    )

                raise RuntimeError(
                    f"Segmentation job failed for some slides. Original error: {str(e)}Operation log: {print_log(log)}"
                ) from e

        else:
            # If not overwriting, simply check for existing outputs
            tissue_seg_result, log = self.validate_tissue_segmentation(log=log)
            if not tissue_seg_result:
                raise ValueError(
                    "Overwrite is false but segmentation results do not exist for all slides. Run tissue segmentation first"
                    f"Operation log: {print_log(log)}"
                )
            log.append('\n Skipping segmentation job as all results found. Using existing results.')

        dir_with_geojson_contours = Path(self.job_dir) / 'contours_geojson'
        dir_with_tissue_contours_jpg = Path(self.job_dir) / 'contours'
        dir_with_slide_thumbnails = Path(self.job_dir) / 'thumbnails'
        tissue_segmentation_log_file = Path(self.job_dir) / '_logs_segmentation.txt'
        tissue_segmentation_config_file = Path(self.job_dir) / '_config_segmentation.json'
        number_of_processed_segmentations = len(list(Path(dir_with_geojson_contours).glob("*.geojson")))

        asset_dict = {
            'dir_with_geojson_contours': str(dir_with_geojson_contours),
            'dir_with_tissue_contours_jpg': str(dir_with_tissue_contours_jpg),
            'dir_with_slide_thumbnails': str(dir_with_slide_thumbnails),
            'tissue_segmentation_log_file': str(tissue_segmentation_log_file),
            'tissue_segmentation_config_file': str(tissue_segmentation_config_file),
            'number_of_processed_segmentations': number_of_processed_segmentations,
            'operation_log': print_log(log),
        }

        return asset_dict

    def run_patch_coordinate_extraction_job(
        self,
        target_magnification: int = 20,
        patch_size: int = 512,
        overlap: int = 0,
        visualize: bool = True,
        min_tissue_proportion: float = 0.95,
        overwrite: bool = False,
        saveto_folder: str | None = None,
    ) -> dict:
        """
        Extract patch coordinates for all slides in the Processor and save results to disk.

        For each slide, patches are extracted at the specified magnification and patch size, subject to
        a minimum tissue proportion filter. Results are saved in standard subfolders under the job directory.
        """

        log = []

        # Prerequisite: make sure tissue segmentation exists
        tissue_seg_result, log = self.validate_tissue_segmentation(log=log)
        if not tissue_seg_result:
            raise RuntimeError(
                "Tissue segmentation does not exist for all slides. Please run segmentation before extracting patches."
                f"Operation log: {print_log(log)}"
            )
        log.append('\n Tissue segmentation check passed. Proceeding with patch coordinate extraction.')

        saveto_folder = saveto_folder or f'{target_magnification}x_{patch_size}px_{overlap}px_overlap'
        if overwrite:
            try:
                dir_with_patches_folder = Path(
                    self.processor.run_patching_job(
                        target_magnification=target_magnification,
                        patch_size=patch_size,
                        overlap=overlap,
                        saveto=saveto_folder,
                        visualize=visualize,
                        min_tissue_proportion=min_tissue_proportion,
                    )
                )
                log.append(
                    f'\n Patch coordinate extraction job completed. Patches saved to: {Path(dir_with_patches_folder) / "patches"}'
                )
            except Exception as e:
                if self.processor.skip_errors:
                    log.append(
                        '\n Patch coordinate extraction job failed for some slides but skipping due to skip_errors=True. Rerun with skip_errors=False to see details.'
                    )
                raise RuntimeError(
                    f"Patch coordinate extraction job failed for some slides. "
                    f"Original error: {str(e)}"
                    f"Operation log: {print_log(log)}"
                ) from e

        else:
            # If not overwriting, simply check for existing outputs
            tissue_patch_results, log = self.validate_patch_coordinates(saveto_folder=saveto_folder, log=log)
            if not tissue_patch_results:
                raise ValueError(
                    "Overwrite is false but patch coordinates do not exist for all slides. "
                    "Run patch coordinate extraction first or set overwrite=True."
                    f"Operation log: {print_log(log)}"
                )
            log.append('\n Found patches for all WSIs. Not re-extracting patches')
            dir_with_patches_folder = Path(self.job_dir) / f'{target_magnification}x_{patch_size}px_{overlap}px_overlap'

        if visualize:
            dir_with_visualization = Path(self.job_dir) / saveto_folder / 'visualization'
        else:
            dir_with_visualization = ''
        log.append(f'\n Patch visualization directory: {dir_with_visualization}')

        patch_job_config_file = dir_with_patches_folder / '_config_coords.json'
        patch_job_log_file = dir_with_patches_folder / '_logs_coords.txt'
        patches_saved_dir = str((Path(dir_with_patches_folder) / 'patches'))

        asset_dict = {
            'dir_with_visualization': str(dir_with_visualization),
            'patch_job_config_file': str(patch_job_config_file),
            'patch_job_log_file': str(patch_job_log_file),
            'patches_key_in_h5': 'coords',
            'patches_saved_dir': patches_saved_dir,
            'operation_log': print_log(log),
        }

        return asset_dict

    def run_patch_features_extraction_job(
        self,
        target_magnification: int = 20,
        patch_size: int = 512,
        overlap: int = 0,
        patch_encoder_name: str = "uni_v1",
        device: str = 'cuda:0',
        batch_limit: int = 512,
        overwrite: bool = False,
        saveto_folder: str | None = None,
        patch_feats_save_folder: str | None = None,
    ) -> dict:
        """
        Extract and save patch-level features for all slides in the Processor.

        This function loads patch coordinate files, computes features for each patch using the specified encoder,
        and saves the features to H5 files with standard naming and attribute conventions. Outputs include features,
        configuration, and logs.
        """

        log = []

        # Prerequisite: make sure tissue segmentation exists
        tissue_seg_results, log = self.validate_tissue_segmentation(log=log)
        if not tissue_seg_results:
            raise RuntimeError(
                "Tissue segmentation does not exist for all slides. Please run segmentation before extracting patches."
                f"Operation log: {print_log(log)}"
            )
        log.append('\n Tissue segmentation check passed. Proceeding with patch feature extraction.')

        # Prerequisite: make sure patch coordinates exist
        saveto_folder = saveto_folder or f'{target_magnification}x_{patch_size}px_{overlap}px_overlap'
        tissue_patch_results, log = self.validate_patch_coordinates(saveto_folder=saveto_folder, log=log)
        if not tissue_patch_results:
            raise ValueError(
                "Overwrite is false but patch coordinates do not exist for all slides. "
                "Run patch coordinate extraction first or set overwrite=True."
                f"Operation log: {print_log(log)}"
            )
        log.append('\n Patch coordinates check passed. Proceeding with patch feature extraction.')

        try:
            patch_encoder = patch_encoder_factory(patch_encoder_name)
            log.append(f'\n Initialized patch encoder: {patch_encoder_name}')
        except ValueError as e:
            raise RuntimeError(
                f"Unable to initialize patch encoder '{patch_encoder_name}'"
                f'Original error: {str(e)}'
                f'Operation log: {print_log(log)}'
            ) from e

        dir_where_patch_feats_saved = patch_feats_save_folder or f'features_{patch_encoder.enc_name}'
        saveto_path = Path(self.job_dir) / saveto_folder / dir_where_patch_feats_saved

        if overwrite:
            try:
                dir_with_patch_features = Path(
                    self.processor.run_patch_feature_extraction_job(
                        coords_dir=str(
                            saveto_folder
                        ),  # 'patches' appeneded to the dir internally!!!!!!!! Do not append here
                        patch_encoder=patch_encoder,
                        device=device,
                        saveas='h5',
                        batch_limit=batch_limit,
                        saveto=str(saveto_path),  # h5 files directly stored in this folder
                    )
                )
                log.append(f'\n Patch feature extraction job completed. Features saved to: {dir_with_patch_features}')
            except Exception as e:
                if self.processor.skip_errors:
                    log.append(
                        '\n Patch feature extraction job failed for some slides but skipping due to skip_errors=True. Rerun with skip_errors=False to see details.'
                    )
                raise RuntimeError(
                    f"Patch feature extraction job failed for some slides. "
                    f"Original error: {str(e)}"
                    f"Operation log: {print_log(log)}"
                ) from e

        else:
            # If not overwriting, simply check for existing outputs
            tissue_patch_feat_results, log = self.validate_patch_features(
                saveto_folder=saveto_folder,
                patch_feats_save_folder=dir_where_patch_feats_saved,
                log=log,
            )
            if not tissue_patch_feat_results:
                raise ValueError(
                    "Overwrite is false but patch features do not exist for all slides. "
                    "Run patch feature extraction first or set overwrite=True."
                    f"Operation log: {print_log(log)}"
                )
            log.append('\n Skipping patch feature extraction job as all results found. Using existing results.')
            dir_with_patch_features = Path(self.job_dir) / saveto_folder / f'features_{patch_encoder.enc_name}'

        patch_features_job_config_file = (
            Path(self.job_dir) / saveto_folder / f'_config_feats_{patch_encoder.enc_name}.json'
        )
        patch_features_job_log_file = Path(self.job_dir) / saveto_folder / f'_logs_feats_{patch_encoder.enc_name}.txt'
        number_of_patch_features_extracted = len(list(Path(dir_with_patch_features).glob("*.h5")))

        asset_dict = {
            'dir_with_patch_features': str(dir_with_patch_features),
            'patch_features_key_in_h5': 'features',
            'patch_features_job_config_file': str(patch_features_job_config_file),
            'patch_features_job_log_file': str(patch_features_job_log_file),
            'number_of_patch_features_extracted': number_of_patch_features_extracted,
            'operation_log': print_log(log),
        }

        return asset_dict

    def run_slide_features_extraction_job(
        self,
        target_magnification: int = 20,
        patch_size: int = 512,
        overlap: int = 0,
        slide_encoder_name: str = "titan",
        patch_encoder_name: str = 'conch_v15',
        device: str = 'cuda:0',
        batch_limit: int = 512,
        overwrite: bool = False,
        saveto_folder: str | None = None,
        slide_feats_save_folder: str | None = None,
    ) -> dict:
        """
        Extract and save slide-level features for all slides in the Processor.

        For each slide, aggregates patch-level information using the specified slide encoder and saves the
        resulting slide-level features to H5 files with standard naming and attribute conventions.
        Outputs also include job configuration and log files.
        """

        log = []

        PATCH_TO_SLIDE_ENC = {
            'threads': 'conch_v15',
            'titan': 'conch_v15',
            'prism': 'virchow',
            'chief': 'ctranspath',
            'gigapath': 'gigapath',
            'madeleine': 'conch_v1',
            'mean': None,  # Means ANY patch encoder is allowed
            'abmil': None,  # Means ANY patch encoder is allowed
        }
        log.append(f'\n Slide encoder mapping being used: {PATCH_TO_SLIDE_ENC}')

        # Validate slide_encoder_name
        if slide_encoder_name not in PATCH_TO_SLIDE_ENC:
            raise ValueError(
                f"slide_encoder_name '{slide_encoder_name}' not recognized. "
                f"Must be one of: {list(PATCH_TO_SLIDE_ENC.keys())}"
                f'Operation log: {print_log(log)}'
            )

        # Prerequisite: make sure tissue segmentation exists
        tissue_seg_results, log = self.validate_tissue_segmentation(log=log)
        if not tissue_seg_results:
            raise RuntimeError(
                "Tissue segmentation does not exist for all slides. Please run segmentation before extracting patches."
                f"Operation log: {print_log(log)}"
            )
        log.append('\n Tissue segmentation check passed. Proceeding with patch feature extraction.')

        # Prerequisite: make sure patch coordinates exist
        saveto_folder = saveto_folder or f'{target_magnification}x_{patch_size}px_{overlap}px_overlap'
        tissue_patch_results, log = self.validate_patch_coordinates(saveto_folder=saveto_folder, log=log)
        if not tissue_patch_results:
            raise ValueError(
                "Overwrite is false but patch coordinates do not exist for all slides. "
                "Run patch coordinate extraction first or set overwrite=True."
                f"Operation log: {print_log(log)}"
            )
        log.append('\n Patch coordinates check passed. Proceeding with slide feature extraction.')

        # Prerequisite: make sure patch features exist
        required_patch_encoder = PATCH_TO_SLIDE_ENC[slide_encoder_name]
        if required_patch_encoder is not None:
            # If patch_encoder_name is not set, default to required_patch_encoder
            patch_encoder_name = patch_encoder_name or required_patch_encoder
            if patch_encoder_name != required_patch_encoder:
                raise ValueError(
                    f"slide_encoder '{slide_encoder_name}' requires patch_encoder '{required_patch_encoder}', "
                    f"but got patch_encoder '{patch_encoder_name}'."
                    f"Operation log: {print_log(log)}"
                )
        else:
            # mean/abmil: accept any patch encoder (do not restrict)
            patch_encoder_name = patch_encoder_name  # can be any, or default to some global default if needed

        # Check for correct patch encoder usage
        if patch_encoder_name != required_patch_encoder:
            raise ValueError(
                f"Incompatible patch encoder '{patch_encoder_name}' for slide encoder '{slide_encoder_name}'. "
                f"Expected patch encoder: '{required_patch_encoder}'."
                f"Operation log: {print_log(log)}"
            )
        log.append(
            f'\n Correct patch encoder for slide encoder {slide_encoder_name} found. Patch encoder being used: {patch_encoder_name}'
        )

        try:
            slide_encoder = slide_encoder_factory(slide_encoder_name)
            log.append(f'\n Initialized slide encoder: {slide_encoder_name}')
        except ValueError as e:
            raise RuntimeError(
                f"Unable to initialize slide encoder '{slide_encoder_name}'"
                f'Original error: {str(e)}'
                f'Operation log: {print_log(log)}'
            ) from e

        dir_with_patches = Path(self.job_dir) / saveto_folder
        slide_feats_save_folder = slide_feats_save_folder or f'features_{slide_encoder.enc_name}'
        saveto_path = Path(self.job_dir) / saveto_folder / slide_feats_save_folder

        if overwrite:
            try:
                dir_with_slide_features = Path(
                    self.processor.run_slide_feature_extraction_job(  # type: ignore[return-value]
                        slide_encoder=slide_encoder,
                        coords_dir=str(dir_with_patches),  # 'patches' appended to the dir internally!!!!!!!!
                        device=device,
                        batch_limit=batch_limit,
                        saveas='h5',
                        saveto=str(saveto_path),  # h5 files directly saved here
                    )
                )
                log.append(f'\n Slide feature extraction job completed. Features saved to: {dir_with_slide_features}')
            except Exception as e:
                if self.processor.skip_errors:
                    log.append(
                        '\n Slide feature extraction job failed for some slides but skipping due to skip_errors=True. Rerun with skip_errors=False to see details.'
                    )
                raise RuntimeError(
                    f"Slide feature extraction job failed for some slides. "
                    f"Original error: {str(e)}"
                    f"Operation log: {print_log(log)}"
                ) from e
        else:
            # If not overwriting, simply check for existing outputs
            slide_feats_results, log = self.validate_slide_features(
                saveto_folder=saveto_folder,
                slide_feats_save_folder=slide_feats_save_folder,
                log=log,
            )
            if not slide_feats_results:
                raise ValueError(
                    "Overwrite is false but slide features do not exist for all slides. "
                    "Run slide feature extraction first or set overwrite=True."
                    f"Operation log: {print_log(log)}"
                )
            log.append('\n Slide feature found. Not going to re-extract slide features.')
            dir_with_slide_features = Path(self.job_dir) / saveto_folder / f'features_{slide_encoder.enc_name}'

        # final output paths and files
        slide_features_job_config_file = (
            Path(self.job_dir) / saveto_folder / f'_config_slide_features_{slide_encoder.enc_name}.json'
        )
        slide_features_job_log_file = (
            Path(self.job_dir) / saveto_folder / f'_logs_slide_features_{slide_encoder.enc_name}.txt'
        )
        number_of_slide_features_extracted = len(list(Path(str(dir_with_slide_features)).glob("*.h5")))

        asset_dict = {
            'dir_with_slide_features': str(dir_with_slide_features),
            'slide_features_key_in_h5': 'features',
            'slide_features_job_config_file': str(slide_features_job_config_file),
            'slide_features_job_log_file': str(slide_features_job_log_file),
            'number_of_slide_features_extracted': number_of_slide_features_extracted,
            'operation_log': print_log(log),
        }

        return asset_dict
