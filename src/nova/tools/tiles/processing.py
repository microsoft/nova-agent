from pathlib import Path

import h5py
import torch
from PIL import Image
from smolagents import tool
from trident.patch_encoder_models import encoder_factory

from nova.utils.summarize import print_log


@tool
def encode_histology_roi_tool(
    img_path: str,
    output_dir: str,
    histology_roi_encoder_name: str = 'conch_v15',
    device: str = "cuda:0",
) -> dict:
    """
    Encode a histology ROI image using a specified encoder model.

    Operations:
        - Load a WSI foundation model using trident library.
            - Supported encoders are: [
                "conch_v1", "conch_v15", "uni_v1",
                "uni_v2", "ctranspath", "phikon",
                "phikon_v2", "resnet50", "gigapath",
                "virchow", "virchow2", "hoptimus0",
                "hoptimus1", "musk", "hibou_l",
                "kaiko-vitb8", "kaiko-vitb16", "kaiko-vits8",
                "kaiko-vits16", "kaiko-vitl14", "lunit-vits8"
                ]
        - Encode the input image using the specified encoder model. Uses the precision and eval transforms defined in the model.
        - Save the resulting embedding in a h5 file under dataset 'embedding'. Embedding is of shape [1, embedding_dim].
        - Tool generates output files at:
            - output_dir/{img_name}_{histology_roi_encoder_name}_embedding.h5

    Prerequisites:
        - The input image should be a valid image file.
            - Supported formats are: ['.png', '.jpg', '.jpeg', '.tif', '.tiff'].
        - The specified encoder model should be available in the trident library and user should have access to it.

    Returns:
        - dict: A dictionary with keys:
            - 'embedding_path': Path to the saved h5 files with embedding.
            - 'embedding_dim': Dimension of the embedding.
            - 'operation_log' : A string describing the operation performed.

    Args:
        img_path (str): Path to the input image file.
        output_dir (str): Directory where the output embedding will be saved.
        histology_roi_encoder_name (str): Name of the encoder model to use. Defaults to 'conch_v15'.
        device (str): Device to run the model on, e.g., "cuda:0" or "cpu". Defaults to "cuda:0" if available. Defaults to 'cuda:0'.
    """

    log = []

    # open image
    try:
        image = Image.open(img_path)
    except Exception as e:
        raise ValueError(f"Failed to open image {img_path}Original error: {e}Operations log: {print_log(log)}")
    log.append(f'\n Loaded image {img_path} successfully.')

    # prepare model
    try:
        device = device or "cuda:0" if torch.cuda.is_available() else "cpu"
        model = encoder_factory(histology_roi_encoder_name).eval().to(device)
        eval_transforms = model.eval_transforms
        precision = model.precision
    except Exception as e:
        raise ValueError(
            f"Failed to load model {histology_roi_encoder_name}Original error: {e}Operations log: {print_log(log)}"
        )
    log.append(f'\n Loaded model {histology_roi_encoder_name} successfully.')

    # Preprocess image
    try:
        img_tensor = eval_transforms(image).unsqueeze(0).to(device)
    except Exception as e:
        raise ValueError(
            f"Failed to preprocess image {img_path} with model {histology_roi_encoder_name}"
            f"Original error: {e}"
            f'Operations log: {print_log(log)}'
        )
    log.append(f'\n Preprocessed image {img_path} successfully.')

    # Inference with correct precision
    try:
        with torch.no_grad():
            if precision in [torch.float16, torch.bfloat16]:
                # Use autocast for mixed precision if available
                with torch.autocast(device, dtype=precision):
                    embedding = model(img_tensor)
            else:
                embedding = model(img_tensor)
    except Exception as e:
        raise ValueError(
            f"Failed to encode image {img_path} with model {histology_roi_encoder_name}"
            f"Original error: {e}"
            f'Operations log: {print_log(log)}'
        )
    log.append(f'\n Encoded image {img_path} successfully with model {histology_roi_encoder_name}.')

    # If the model returns a tuple, select the first element
    if isinstance(embedding, (tuple, list)):
        embedding = embedding[0]

    # Move to CPU and convert to numpy if desired
    embedding_np = embedding.cpu().numpy()

    # save the embedding in a h5 file
    output_file_path = Path(output_dir) / f"{Path(img_path).stem}_{histology_roi_encoder_name}_embedding.h5"
    try:
        with h5py.File(output_file_path, "w") as f:
            f.create_dataset("embedding", data=embedding_np)
    except Exception as e:
        raise ValueError(
            f"Failed to save embedding for image {img_path} with model {histology_roi_encoder_name}"
            f"Original error: {e}"
            f'Operations log: {print_log(log)}'
        )
    log.append(f'\n Saved embedding for image {img_path} at {output_file_path}.')

    asset_dict = {
        "embedding_path": str(output_file_path),
        "embedding_dim": embedding_np.shape[1],
        "operations_log": print_log(log),
    }

    return asset_dict
