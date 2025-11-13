import json
import shutil
import subprocess
import sys
import warnings
from os import environ
from pathlib import Path

import torch

from nova.constants import HOVERNET_ID
from nova.paths import HOVERNET_PATH, HOVERNET_ROI_WEIGHTS_PATH

sys.path.append(str(HOVERNET_PATH))
from infer.tile import InferManager  # type: ignore[import]


class ROINucleiSegmentation:
    def __init__(
        self,
        config_path: str,
        cell_types_path: str,
        level0_mpp_of_imgs: dict[str, float],
        input_dir: str,
        output_dir: str,
        model_weights_path: str | None = None,
        batch_size: int = 128,
        nr_inference_workers: int = 8,
        nr_post_proc_workers: int = 16,
        gpu_list: str = '0',
        mem_usage: float = 0.2,
        draw_dot: bool = True,
        save_qupath: bool = False,
        save_raw_map: bool = False,
    ):
        # basic setup
        environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        nr_gpus = torch.cuda.device_count()
        print(f'Detect #GPUS: {nr_gpus}')

        # if model weights path is not provided, download to a default path
        if model_weights_path is None:
            HOVERNET_ROI_WEIGHTS_PATH.parent.mkdir(exist_ok=True, parents=True)
            warnings.warn(f"Model weight path is not provided. Downloading weights to {HOVERNET_ROI_WEIGHTS_PATH}")
            cmd_hovernet_download = f"gdown {HOVERNET_ID} --output {HOVERNET_ROI_WEIGHTS_PATH}"
            try:
                subprocess.run(cmd_hovernet_download, shell=True, capture_output=True, text=True)
            except Exception as e:
                raise RuntimeError(f"Failed to download HoverNet ROI weights: {e}")
            self.model_weights_path = HOVERNET_ROI_WEIGHTS_PATH

        # if model weights path is provided, check if it exists If not, download to the provided path
        elif not Path(model_weights_path).exists():
            warnings.warn(
                f"Provided model weights path does not exist: {model_weights_path}. Downloading weights to {model_weights_path}"
            )
            cmd_hovernet_download = f"gdown {HOVERNET_ID} --output {model_weights_path}"
            try:
                subprocess.run(cmd_hovernet_download, shell=True, capture_output=True, text=True)
            except Exception as e:
                raise RuntimeError(f"Failed to download HoverNet ROI weights: {e}")
            self.model_weights_path = model_weights_path

        else:
            print('Located weights at:', model_weights_path)
            self.model_weights_path = model_weights_path

        self.config = self._load_json(config_path)
        self.cell_types_path = cell_types_path
        self.level0_mpp_of_imgs = level0_mpp_of_imgs

        # iterate over all files in input_dir and check if they exist in level0_mpp_of_imgs (without extension)
        self.input_dir = Path(input_dir)
        assert self.input_dir.exists(), f"Input directory does not exist: {self.input_dir}"
        self.input_files = list(self.input_dir.glob('*'))
        self.input_files = [f.stem for f in self.input_files]
        if len(self.input_files) > len(self.level0_mpp_of_imgs):
            print(f"Folder contains more images than specified in {self.level0_mpp_of_imgs=}")
            self.input_files = [f for f in self.input_files if f in self.level0_mpp_of_imgs]
            tmp_folder = Path(output_dir) / 'tmp_data'
            tmp_folder.mkdir(exist_ok=True, parents=True)
            print(f"Copying only specified images to a temporary folder for processing {tmp_folder=}")
            for img_name in self.input_files:
                src = self.input_dir / f"{img_name}.png"
                dst = tmp_folder / f"{img_name}.png"
                _ = shutil.copy(src, dst)
            self.input_dir = tmp_folder
            print(f"Using input dir {self.input_dir} with {len(self.input_files)} images for processing")
        for img_name in self.input_files:
            assert img_name in self.level0_mpp_of_imgs, (
                f"Image {img_name} not found in level0_mpp_of_imgs. Provide level0 mpp of this image"
            )

        # method args (fixed for all runs, except model path)
        self.method_args = self.config['method_args']
        self.method_args['method']['model_path'] = self.model_weights_path
        self.method_args['type_info_path'] = self.cell_types_path

        # run args (defaults, can be overwritten per run)
        self.run_args = self.config['run_args']
        self.run_args['input_dir'] = str(self.input_dir)
        self.run_args['output_dir'] = output_dir
        self.run_args['batch_size'] = min(batch_size * nr_gpus, len(self.input_files))
        self.run_args['nr_inference_workers'] = min(nr_inference_workers, len(self.input_files))
        self.run_args['nr_post_proc_workers'] = min(nr_post_proc_workers, len(self.input_files))
        self.run_args['mem_usage'] = mem_usage
        self.run_args['draw_dot'] = draw_dot
        self.run_args['save_qupath'] = save_qupath
        self.run_args['save_raw_map'] = save_raw_map

        # harcoded cell type map
        self.cell_type_map = {
            0: 'Background',
            1: 'Neoplastic',
            2: 'Inflammatory',
            3: 'Connective',
            4: 'Necrosis',
            5: 'Non-Neoplastic Epithelial',
        }

    def _load_json(self, path: str | Path) -> dict:
        json_path = Path(path)
        assert json_path.exists(), f"Config file not found: {json_path}"
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        assert isinstance(data, dict), "Loaded config is not a dictionary"
        return data

    def segment_classify_nuclei(
        self,
    ):
        # Run segmentation and classification
        infer = InferManager(**self.method_args)
        infer.process_file_list(self.run_args)

        path_to_overlays = Path(self.run_args['output_dir']) / 'overlay'
        path_to_jsons = Path(self.run_args['output_dir']) / 'json'

        # iterate over each json to update cell_type and add magnification
        for json_file in path_to_jsons.glob('*.json'):
            data = self._load_json(json_file)

            # add magnification to the json
            data["mpp"] = self.level0_mpp_of_imgs[json_file.stem]

            # update cell type
            for nid, nuc in data['nuc'].items():
                data['nuc'][nid]['type'] = self.cell_type_map.get(nuc['type'], 'Unknown')

            # write back the updated json
            with json_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

        return {
            'path_to_default_overlays': str(path_to_overlays),
            'path_to_jsons': str(path_to_jsons),
        }
