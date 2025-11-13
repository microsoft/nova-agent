from abc import ABC, abstractmethod
from pathlib import Path

from nova.tools.dataset_io.wsi_dataset_io_tools import (
    dataset_of_wsi_check_patch_coordinates_exist_and_schema_tool,
    dataset_of_wsi_check_patch_features_exist_and_schema_tool,
    dataset_of_wsi_check_slide_features_exist_and_schema_tool,
    dataset_of_wsi_check_tissue_segmentation_exists_tool,
)
from trident import Processor
from trident.Converter import OPENSLIDE_EXTENSIONS, PIL_EXTENSIONS


class TridentProcessingBaseToolkit(ABC):
    """
    Base class for Trident processing toolkits.

    This abstract base class defines the common interface and shared functionality
    for all Trident-based WSI processing toolkits. Subclasses should implement
    specific processing workflows while inheriting common initialization and
    utility methods.
    """

    def __init__(
        self,
        job_dir: str,
        wsi_source: str,
        search_nested: bool = False,
        skip_errors: bool = False,
        max_workers: int = 10,
        skip_specific_wsi: list[str] | None = None,
        keep_only_these_wsi: list[str] | None = None,
    ):
        """
        Initialize the base toolkit with common parameters.

        Args:
            job_dir (str): Path to the job directory where outputs will be stored.
            wsi_source (str): Path to the directory containing input WSI files.
            search_nested (bool): If True, search subdirectories for WSI files.
            skip_errors (bool): If True, skip slides that cause errors during processing.
            max_workers (int): Maximum number of parallel workers.
            keep_only_these_wsi (list[str] | None): List of WSI names to keep for processing. None means all WSIs are kept after skip_specific_wsi filtering.
            skip_specific_wsi (list[str] | None): List of WSI names to skip during processing. None means no WSIs are skipped.
        """
        self.job_dir = job_dir
        self.wsi_source = wsi_source
        self.search_nested = search_nested
        self.skip_errors = skip_errors
        self.max_workers = max_workers

        # these parameters are used to filter the WSIs that will be processed
        self.skip_specific_wsi = skip_specific_wsi if skip_specific_wsi is not None else []
        self.keep_only_these_wsi = keep_only_these_wsi if keep_only_these_wsi is not None else []
        assert set(self.skip_specific_wsi).isdisjoint(set(self.keep_only_these_wsi)), (
            "skip_specific_wsi and keep_only_these_wsi cannot have common elements."
        )

        # Initialize processor. Accounts for skip_specific_wsi and keep_only_these_wsi
        self.processor = self._init_trident_processor()
        self.wsis = [wsi.name for wsi in self.processor.wsis]

        print(f'[PROCESSOR] After filtering, the processor is initialized with {len(self.wsis)} WSIs')

    def _find_valid_extensions(self):
        """
        Find all unique valid file extensions in wsi_source directory.

        Returns:
            list: Sorted list of found valid extensions, e.g., ['.ndpi', '.png']
        """

        allowed_exts = set([e.lower() for e in OPENSLIDE_EXTENSIONS.union(PIL_EXTENSIONS)])
        if self.search_nested:
            files = Path(self.wsi_source).rglob('*')
        else:
            files = Path(self.wsi_source).glob('*')

        found_exts = set()
        for f in files:
            if f.is_file():
                ext = f.suffix.lower()
                if ext in allowed_exts:
                    found_exts.add(ext)

        return sorted(found_exts)

    def _init_trident_processor(self) -> Processor:
        """
        Initialize a Trident WSI processor object and return it.

        Returns:
            Processor: The initialized Processor instance, ready for downstream use.

        Raises:
            ValueError: If job_dir or wsi_source don't exist, or no valid WSI files found.
        """
        # IO checks
        if not Path(self.job_dir).exists():
            raise ValueError(f"Job directory {self.job_dir} does not exist.")

        if not Path(self.wsi_source).exists():
            raise ValueError(f"WSI source directory {self.wsi_source} does not exist.")

        wsi_ext = self._find_valid_extensions()
        if len(wsi_ext) == 0:
            raise ValueError(
                f"No valid WSI file extensions found in {self.wsi_source}. "
                f"Please check the directory and ensure it contains valid WSI files. "
                f"If you have nested directories, set search_nested=True to search recursively."
            )

        processor = Processor(
            job_dir=self.job_dir,
            wsi_source=self.wsi_source,
            wsi_ext=wsi_ext,
            skip_errors=self.skip_errors,
            max_workers=self.max_workers,
            search_nested=self.search_nested,
        )

        new_set_of_slides = []
        for wsi in processor.wsis:
            if wsi.name not in self.skip_specific_wsi:
                new_set_of_slides.append(wsi)

        # if empty then evaluates to False and no filtering is done.
        if self.keep_only_these_wsi:
            new_set_of_slides = [wsi for wsi in new_set_of_slides if wsi.name in self.keep_only_these_wsi]

        processor.wsis = new_set_of_slides

        return processor

    @abstractmethod
    def get_supported_operations(self) -> list:
        """
        Return a list of operations supported by this toolkit.

        Returns:
            list: List of supported operation names.
        """
        pass

    @abstractmethod
    def validate_prerequisites(self, operation: str, **kwargs) -> bool:
        """
        Validate that prerequisites for a given operation are met.

        Args:
            operation (str): Name of the operation to validate.
            **kwargs: Operation-specific parameters.

        Returns:
            bool: True if prerequisites are met.

        Raises:
            RuntimeError: If prerequisites are not met.
        """
        pass

    def validate_tissue_segmentation(
        self,
        log: list[str],
    ) -> tuple[bool, list[str]]:
        """
        Validate that tissue segmentation exists for all slides.

        Returns:
            bool: True if tissue segmentation exists for all slides.
            list[str]: Log messages indicating the status of each slide.
        """

        results, _ = dataset_of_wsi_check_tissue_segmentation_exists_tool(  # type: ignore
            job_dir=self.job_dir,
            wsis=self.wsis,
            skip_wsi_to_check=self.skip_specific_wsi,
        )
        all_results = []
        for wsi in results:
            if not results[wsi]['geojson_exists']:  # type: ignore
                log.append(
                    f"\n Tissue segmentation does not exist for slide {wsi}. Either skip processing this slide or run tissue segmentation first on this slide."
                )
                all_results.append(False)
            else:
                log.append(f"\n Tissue segmentation exists for slide {wsi}.")
                all_results.append(True)

        return all(all_results), log

    def validate_patch_coordinates(
        self,
        saveto_folder: str,
        log: list[str],
    ) -> tuple[bool, list[str]]:
        """
        Validate that patch coordinates exist for all slides.

        Returns:
            bool: True if patch coordinates exist for all slides.
            list[str]: Log messages indicating the status of each slide.
        """
        results, _ = dataset_of_wsi_check_patch_coordinates_exist_and_schema_tool(  # type: ignore
            job_dir=self.job_dir,
            wsis=self.wsis,
            saveto_folder=saveto_folder,
            skip_wsi_to_check=self.skip_specific_wsi,
        )
        all_results = []
        for wsi in results:
            if not results[wsi]['file_schema_ok']:  # type: ignore
                log.append(
                    f"\n Patch coordinates do not exist for slide {wsi}. Either skip processing this slide or run patch coordinate extraction first on this slide."
                )
                all_results.append(False)
            else:
                log.append(f"\n Patch coordinates exist for slide {wsi}.")
                all_results.append(True)

        return all(all_results), log

    def validate_patch_features(
        self,
        saveto_folder: str,
        patch_feats_save_folder: str,
        log: list[str],
    ) -> tuple[bool, list[str]]:
        """
        Validate that patch features exist for all slides.

        Returns:
            bool: True if patch features exist for all slides.
            list[str]: Log messages indicating the status of each slide.
        """
        results, _ = dataset_of_wsi_check_patch_features_exist_and_schema_tool(  # type: ignore
            job_dir=self.job_dir,
            wsis=self.wsis,
            saveto_folder=saveto_folder,
            patch_feats_save_folder=patch_feats_save_folder,
            skip_wsi_to_check=self.skip_specific_wsi,
        )
        all_results = []
        for wsi in results:
            if not results[wsi]['file_schema_ok']:  # type: ignore
                log.append(
                    f"\n Patch features do not exist for slide {wsi}. Either skip processing this slide or run patch feature extraction first on this slide."
                )
                all_results.append(False)
            else:
                log.append(f"\n Patch features exist for slide {wsi}.")
                all_results.append(True)
        return all(all_results), log

    def validate_slide_features(
        self,
        saveto_folder: str,
        slide_feats_save_folder: str,
        log: list[str],
    ) -> tuple[bool, list[str]]:
        """
        Validate that slide features exist for all slides.

        Returns:
            bool: True if slide features exist for all slides.
            list[str]: Log messages indicating the status of each slide.
        """
        results, _ = dataset_of_wsi_check_slide_features_exist_and_schema_tool(  # type: ignore
            job_dir=self.job_dir,
            wsis=self.wsis,
            skip_wsi_to_check=self.skip_specific_wsi,
            saveto_folder=saveto_folder,
            slide_feats_save_folder=slide_feats_save_folder,
        )
        all_results = []
        for wsi in results:
            if not results[wsi]['file_schema_ok']:  # type: ignore
                log.append(
                    f"\n Slide features do not exist for slide {wsi}. Either skip processing this slide or run slide feature extraction first on this slide."
                )
                all_results.append(False)
            else:
                log.append(f"\n Slide features exist for slide {wsi}.")
                all_results.append(True)
        return all(all_results), log
