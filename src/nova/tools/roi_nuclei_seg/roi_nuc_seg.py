from smolagents import tool

from nova.paths import HOVERNET_CELL_TYPES_PATH, HOVERNET_ROI_CONFIG_PATH
from nova.toolkits.nuc_seg.toolkit_seg import ROINucleiSegmentation


@tool
def segment_and_classify_nuclei_in_histology_roi_tool(
    input_dir: str,
    output_dir: str,
    level0_mpp_of_imgs: dict[str, float],
) -> dict:
    """
    Segment and classify nuclei in histology ROIs into six classes: Background, Neoplastic, Inflammatory,
    Connective, Necrosis, and Non-Neoplastic Epithelial.

    This tool runs a HoVerNet-based nuclei segmentation and classification workflow over flat image folders
    (no subdirectories). For each image, it produces a JSON file describing every detected nucleus (geometry,
    class, and confidence) and an overlay PNG visualization.

    Notes on using this tool:
        - Supported input formats: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`.
        - The tool loads model weights and configuration, then runs HoVerNet inference followed by post-processing.
            - If paths to these assets are not provided, then they are downloaded on the fly.
        - `level0_mpp_of_imgs` provides per-image level-0 microns-per-pixel used to convert pixel units to microns.

    Directory structure & files created:
        - `{output_dir}/json/{image_name}.json`:
            Per-image JSON with nuclei results. Top-level keys:
              - `mpp` (float): level-0 microns-per-pixel for the ROI.
              - `mag` : MUST BE IGNORED
              - `nuc` (dict[str, dict]): mapping from nucleus ID (as a string) to nucleus segmentation asset:
                  * `bbox` (list[[int, int], [int, int]]): [[min_x, min_y], [max_x, max_y]] in pixel coordinates at level 0 magnification.
                  * `centroid` (list[float, float]): nuclei contour centroid as [x, y] in pixel coordinates at level 0 magnification.
                  * `contour` (list[list[int, int]]): nuclei boundary contour represented as a list of [x, y] points in pixel coordinates at level 0 magnification.
                  * `type_prob` (float): confidence in [0, 1].
                  * `type` (str): one of
                    (`Background`, `Neoplastic`, `Inflammatory`, `Connective`, `Necrosis`, `Non-Neoplastic Epithelial`).
        - `{output_dir}/overlay/{image_name}.png`:
            Overlay visualization of segmentation/classification on the input image. Centroids are shown as dots.

    Prerequisites to run this tool:
        - `input_dir` must exist, contain **only images**, and **must not** contain subdirectories.
        - `output_dir` must be writable.
        - `level0_mpp_of_imgs` must map each image (filename **without extension**) in `input_dir` to its level-0 MPP (float).

    The tool returns:
        - dict with the following keys:
            - 'path_to_default_overlays': Absolute path to `{output_dir}/overlay`.
            - 'path_to_jsons': Absolute path to `{output_dir}/json`.

    Args:
        input_dir: Directory containing the input images. Must be flat (no subdirectories) and contain only images.
        output_dir: Directory to save outputs (JSONs, overlays, and optional exports).
        level0_mpp_of_imgs: Dict mapping image basenames (no extension) to level-0 microns-per-pixel (float).
                            Every image in `input_dir` must have an entry.
    """

    nuc_seg_toolkit = ROINucleiSegmentation(
        config_path=str(HOVERNET_ROI_CONFIG_PATH),
        cell_types_path=str(HOVERNET_CELL_TYPES_PATH),
        level0_mpp_of_imgs=level0_mpp_of_imgs,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    return nuc_seg_toolkit.segment_classify_nuclei()
