from pathlib import Path


CHECKPOINTS_PATH = Path("outputs") / "checkpoints"

# Define the root directory of the repository
REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"

# configs paths
CONFIGS_DIR = REPO_ROOT / "configs"
LLM_CONFIGS_DIR = CONFIGS_DIR / "llms"

# Prompts root paths
PROMPTS_DIR = REPO_ROOT / "prompts"
CAPTIONING_PROMPT_DIR = PROMPTS_DIR / "captioning"
GENERAL_SYSTEM_PROMPTS_DIR = PROMPTS_DIR / "general_system_prompts"

# Prompt paths for captioning
HIST_IMG_CLUSTER_CAPTIONS_PROMPT_PATH = CAPTIONING_PROMPT_DIR / "histology_image_cluster_captions.yaml"
SINGLE_HIST_IMG_CAPTIONS_PROMPT_PATH = CAPTIONING_PROMPT_DIR / "single_histology_image_captions.yaml"
SUMMARY_PROMPT_PATH = CAPTIONING_PROMPT_DIR / "summarize_histology_captions.yaml"
WSI_REPORT_GEN_PROMPT_PATH = CAPTIONING_PROMPT_DIR / "whole_slide_image_caption.yaml"

# API Paths for RAG
APIS_DIR = REPO_ROOT / "apis"

# nuclei segmentation
HOVERNET_PATH = REPO_ROOT / 'src/nova/toolkits/nuc_seg/hover_net'
HOVERNET_ROI_CONFIG_PATH = REPO_ROOT / "configs" / "segmentation" / "hovernet_roi" / "pannuke_config.json"
HOVERNET_CELL_TYPES_PATH = REPO_ROOT / "configs" / "segmentation" / "hovernet_roi" / "pannuke_cell_types.json"
HOVERNET_ROI_WEIGHTS_PATH = CHECKPOINTS_PATH / "hovernet" / "hovernet_fast_pannuke_type_tf2pytorch.tar"

# benchmark paths
BENCHMARK_DIR = REPO_ROOT / "benchmark"
DATA_QA_DIR = BENCHMARK_DIR / "data_qa"
PATCH_QA_DIR = BENCHMARK_DIR / "patch_qa"
CELLULAR_QA_DIR = BENCHMARK_DIR / "cellular_qa"
SLIDE_QA_DIR = BENCHMARK_DIR / "slide_qa"
