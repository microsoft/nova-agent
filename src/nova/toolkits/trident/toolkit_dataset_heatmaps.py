from pathlib import Path

import h5py
import numpy as np

from nova.toolkits.trident.toolkit_base import TridentProcessingBaseToolkit
from nova.utils.summarize import print_log
from trident import OpenSlideWSI, visualize_heatmap


class TridentVisualizationToolkit(TridentProcessingBaseToolkit):
    """
    Concrete implementation of TridentProcessingBaseToolkit for WSI visualization tasks.

    This toolkit provides methods for generating heatmaps and other visualizations
    from attention scores and patch coordinates.
    """

    def get_supported_operations(self) -> list:
        """
        Return a list of operations supported by this toolkit.

        Returns:
            list: List of supported operation names.
        """
        return [
            "heatmap_generation",
        ]

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
        return True

    def generate_heatmaps(
        self,
        scores_dir: str,
        target_magnification: int,
        patch_size: int,
        overlap: int,
        output_dir: str,
        normalize: bool = True,
        num_top_patches_to_save: int = 10,
        cmap: str = 'coolwarm',
        saveto_folder: str | None = None,
        scores_key: str = 'attention_scores',
    ) -> dict:
        """
        Generate heatmaps from scores and corresponding patch coordinates for each WSI.

        Args:
            scores_dir: Directory containing .h5 score files.
            target_magnification: Magnification level used for patch extraction.
            patch_size: Size of each extracted patch, in pixels.
            overlap: Overlap between adjacent patches, in pixels.
            output_dir: Path where output heatmaps and top-k patch images will be saved.
            normalize: Whether to normalize scores before visualization.
            num_top_patches_to_save: Number of top-scoring patches to extract and save per slide.
            cmap: Colormap for heatmap overlays.
            saveto_folder: Subfolder name for saving outputs.
            scores_key: Name of the dataset within the .h5 files containing the scores.
        """

        log = []

        scores_dir_obj = Path(scores_dir)
        output_dir_obj = Path(output_dir)
        saveto_folder = saveto_folder or f'{target_magnification}x_{patch_size}px_{overlap}px_overlap'

        if not scores_dir_obj.exists():
            raise ValueError(
                f"Scores directory {scores_dir_obj} does not exist. "
                "Make sure scores are generated before running this function."
            )

        # Collect all WSIs that have scores
        wsi_names = [f.stem for f in scores_dir_obj.glob("*.h5")]
        # only keep the wsi that are in the processor (which is made after removing skip_specific_wsi and keeping wsis requested by user)
        wsi_names = [wsi for wsi in wsi_names if wsi in self.wsis]

        if not wsi_names:
            raise ValueError(f"No .h5 score files found in {scores_dir_obj}.")
        log.append(f"Found {len(wsi_names)} WSIs with scores")

        # Check for patch coordinates
        tissue_patch_results, log = self.validate_patch_coordinates(saveto_folder=saveto_folder, log=log)
        if not tissue_patch_results:
            raise ValueError(
                "Patch coordinates do not exist for all slides. "
                "Run patch coordinate extraction first."
                f"Operation log: {print_log(log)}"
            )
        log.append('\n Patch coordinates check passed. Proceeding with heatmap generation.')

        heatmaps_save_paths = {}
        print(f'Generating heatmaps with {len(wsi_names)} WSIs...')
        for wsi_name in wsi_names:
            # === Load scores ===
            scores_for_slide_path = scores_dir_obj / f"{wsi_name}.h5"

            with h5py.File(scores_for_slide_path, "r") as h5f:
                scores_dataset = h5f.get(scores_key)
                if scores_dataset is None:
                    raise KeyError(
                        f"'{scores_key}' dataset not found in {scores_for_slide_path}Operation log: {print_log(log)}"
                    )
                log.append(f"\n Loaded scores for WSI '{wsi_name}' from {scores_for_slide_path}")

                # Type check and convert
                if isinstance(scores_dataset, h5py.Dataset):
                    scores = np.array(scores_dataset[:]).squeeze()
                else:
                    raise TypeError(f"Expected h5py.Dataset, got {type(scores_dataset)}Operation log: {print_log(log)}")
            print('- Got attention scores')

            # === Load patch coordinates and attributes ===
            patch_coords_path = Path(self.job_dir) / saveto_folder / "patches" / f"{wsi_name}_patches.h5"
            if not patch_coords_path.exists():
                raise FileNotFoundError(
                    f"Patch coordinates file {patch_coords_path} does not exist. "
                    "Make sure patch coordinates are generated before running this function."
                    f'Operation log: {print_log(log)}'
                )

            with h5py.File(patch_coords_path, 'r') as h5f:
                coords_dataset = h5f.get('coords')
                if coords_dataset is None:
                    raise KeyError(f"'coords' dataset not found in {patch_coords_path}")

                # Type check and convert
                if isinstance(coords_dataset, h5py.Dataset):
                    coords = coords_dataset[:]
                    coords_attrs = dict(coords_dataset.attrs)
                    patch_size_level0_raw = coords_attrs.get('patch_size_level0')

                    if patch_size_level0_raw is None:
                        raise KeyError(
                            f"'patch_size_level0' attribute not found in {patch_coords_path}. "
                            "Ensure patch coordinates were generated correctly."
                            f'Operation log: {print_log(log)}'
                        )

                    # Ensure patch_size_level0 is an integer
                    if isinstance(patch_size_level0_raw, (int, np.integer)):
                        patch_size_level0 = int(patch_size_level0_raw)
                    else:
                        raise TypeError(f"Expected integer for patch_size_level0, got {type(patch_size_level0_raw)}")
                else:
                    raise TypeError(f"Expected h5py.Dataset, got {type(coords_dataset)}")

            log.append(f"\n Loaded patch coordinates for WSI '{wsi_name}' from {patch_coords_path}")
            print('- Got patches')

            # === Check matching shapes ===
            if scores.shape[0] != coords.shape[0]:
                raise ValueError(
                    f"Number of patches in scores ({scores.shape[0]}) does not match number of patches in coordinates ({coords.shape[0]}) for slide '{wsi_name}'. "
                    "Ensure scores are generated for the same patches as the coordinates."
                    f'Operation log: {print_log(log)}'
                )

            # === Initialize OpenSlideWSI instance (slide path must be specified here) ===
            try:
                slide_path = next(p for p in Path(self.wsi_source).iterdir() if p.is_file() and p.stem == wsi_name)
            except StopIteration:
                raise FileNotFoundError(
                    f"No WSI found for '{wsi_name}' in {self.wsi_source}Operation log: {print_log(log)}"
                )
            slide = OpenSlideWSI(slide_path=slide_path, lazy_init=False)
            log.append(f"\n Initialized OpenSlideWSI for '{wsi_name}' from {slide_path}")

            # === Create output directory for this slide ===
            output_dir_for_slide = output_dir_obj / wsi_name
            output_dir_for_slide.mkdir(parents=True, exist_ok=True)
            log.append(f"\n Created output directory for WSI '{wsi_name}' at {output_dir_for_slide}")

            # === Generate and save the heatmap ===
            heatmap_save_path = visualize_heatmap(
                wsi=slide,
                scores=scores,
                coords=coords,
                vis_level=2,
                patch_size_level0=patch_size_level0,
                normalize=normalize,
                num_top_patches_to_save=num_top_patches_to_save,
                output_dir=str(output_dir_for_slide),
                cmap=cmap,
            )
            log.append(
                f"\n Generated heatmap for WSI '{wsi_name}' and saved to {heatmap_save_path}. topk [{num_top_patches_to_save}] patches saved."
            )
            print('- Saved heatmap')

            heatmaps_save_paths[wsi_name] = {
                'heatmap': str(heatmap_save_path),
                'topk_patches_dir': str(output_dir_for_slide / 'topk_patches'),
            }
            print()
            print()

        return {'heatmaps_save_paths': heatmaps_save_paths}
