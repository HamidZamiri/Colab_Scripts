
# ================================================================
# TRIPOSPLAT + COMFYUI COMPLETE INSTALLER
#
# Platform: Google Colab
# GPU: NVIDIA T4 (16 GB VRAM)
# Storage: Temporary /content storage
#
# Installs:
#   1. Required Linux software (Git if missing)
#   2. Official TripoSplat GitHub repository
#   3. Required Python inference dependencies
#   4. Five official pretrained model files
#   5. Correct ComfyUI model directories
#   6. TripoSplat standalone ckpts directory link
#   7. Official ComfyUI workflow
#   8. Installation validation
#
# Preserves:
#   - Existing PyTorch
#   - Existing Torchvision
#   - Existing CUDA runtime
#   - Existing ComfyUI installation
#
# ================================================================

import os
import sys
import json
import shutil
import subprocess
import urllib.request
import importlib.util
from pathlib import Path

# ================================================================
# 1. CONFIGURATION
# ================================================================

COMFY_DIR = Path("/content/ComfyUI")

SOURCE_DIR = Path("/content/TripoSplat")

MODEL_DIR = COMFY_DIR / "models"

CHECKPOINT_LINK = SOURCE_DIR / "ckpts"

WORKFLOW_URL = (
    "https://raw.githubusercontent.com/Comfy-Org/"
    "workflow_templates/main/templates/"
    "3d_triposplat_image_to_gaussian_splat.json"
)

WORKFLOW_DIR = (
    COMFY_DIR / "user" / "default" / "workflows"
)

WORKFLOW_FILE = (
    WORKFLOW_DIR /
    "3d_triposplat_image_to_gaussian_splat.json"
)

GITHUB_URL = (
    "https://github.com/VAST-AI-Research/TripoSplat.git"
)

# Official model repositories and exact paths.
# Sizes are approximate decimal GB, used for disk preflight.

MODELS = [
    {
        "repo": "Comfy-Org/BiRefNet",
        "file": "background_removal/birefnet.safetensors",
        "size_gb": 0.424
    },
    {
        "repo": "VAST-AI/TripoSplat",
        "file": "diffusion_models/triposplat_fp16.safetensors",
        "size_gb": 0.707
    },
    {
        "repo": "VAST-AI/TripoSplat",
        "file": "clip_vision/dino_v3_vit_h.safetensors",
        "size_gb": 1.570
    },
    {
        "repo": "VAST-AI/TripoSplat",
        "file": "vae/flux2-vae.safetensors",
        "size_gb": 0.550
    },
    {
        "repo": "VAST-AI/TripoSplat",
        "file": "vae/triposplat_vae_decoder_fp16.safetensors",
        "size_gb": 0.321
    }
]

FAILED = []
SUCCESSFUL = []

# ================================================================
# 2. HELPER FUNCTIONS
# ================================================================

def run_command(command, description=None):
    """Run a system command and stop if it fails."""

    if description:
        print("\n" + description)

    subprocess.check_call(command)


def validate_safetensors(filepath):
    """Check SafeTensors header and tensor keys without loading
    model weights into GPU memory."""

    filepath = Path(filepath)

    if not filepath.is_file():
        return False

    if filepath.stat().st_size < 1024:
        return False

    try:
        from safetensors import safe_open

        with safe_open(
            str(filepath),
            framework="pt",
            device="cpu"
        ) as model:

            return len(model.keys()) > 0

    except Exception:
        return False


def download_workflow(url, destination):
    """Download and structurally validate official workflow."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(
        request,
        timeout=90
    ) as response:

        content = response.read()

    data = json.loads(content.decode("utf-8"))

    if not isinstance(data, dict):
        raise ValueError("Workflow is not a JSON object.")

    if not isinstance(data.get("nodes"), list):
        raise ValueError("Workflow nodes are missing.")

    definitions = data.get("definitions", {})

    if not isinstance(definitions.get("subgraphs"), list):
        raise ValueError("Workflow subgraph definitions are missing.")

    temporary = destination.with_suffix(".json.tmp")

    with open(temporary, "wb") as file:
        file.write(content)

    temporary.replace(destination)

    return data


# ================================================================
# 3. CHECK EXISTING ENVIRONMENT
# ================================================================

print("\n" + "=" * 65)
print("TRIPOSPLAT COMPLETE INSTALLER")
print("=" * 65)

print("\n[1/8] CHECKING COLAB ENVIRONMENT")

if not COMFY_DIR.is_dir():
    raise FileNotFoundError(
        f"ComfyUI not found at {COMFY_DIR}"
    )

# Import existing GPU environment BEFORE installing anything.

import torch
import torchvision

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Enable GPU acceleration in Colab."
    )

GPU_NAME = torch.cuda.get_device_name(0)

GPU_MEMORY = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print("GPU:", GPU_NAME)
print("VRAM:", round(GPU_MEMORY, 2), "GB")
print("CUDA runtime:", torch.version.cuda)

if "T4" not in GPU_NAME:
    print("WARNING: This runtime is not an NVIDIA T4.")

# Check free temporary storage.

DISK = shutil.disk_usage("/content")

FREE_GB = DISK.free / 1024**3

print("Available temporary storage:", round(FREE_GB, 2), "GB")

if FREE_GB < 1:
    raise RuntimeError("Insufficient temporary storage.")

# ================================================================
# 4. INSTALL REQUIRED LINUX SOFTWARE
# ================================================================

print("\n[2/8] CHECKING LINUX SOFTWARE")

if not shutil.which("git"):

    print("Git is missing. Installing Git...")

    run_command(
        ["apt-get", "update", "-qq"],
        "Updating package index"
    )

    run_command(
        ["apt-get", "install", "-y", "-qq", "git"],
        "Installing Git"
    )

print("Git:", shutil.which("git"))

# Colab provides the NVIDIA driver and CUDA runtime.
# Do NOT install another CUDA toolkit or replace GPU drivers.

# ================================================================
# 5. INSTALL PYTHON DEPENDENCIES
# ================================================================

print("\n[3/8] INSTALLING PYTHON DEPENDENCIES")

# Official TripoSplat inference requirements:
#
# numpy
# safetensors
# pillow
# tqdm
#
# PyTorch and Torchvision are already supplied by Colab/ComfyUI.
# HuggingFace Hub is an additional installer requirement.

DEPENDENCIES = {
    "numpy": "numpy",
    "safetensors": "safetensors",
    "Pillow": "PIL",
    "tqdm": "tqdm",
    "huggingface_hub": "huggingface_hub"
}

MISSING = []

for package, module in DEPENDENCIES.items():

    try:
        if importlib.util.find_spec(module) is None:
            MISSING.append(package)

    except (ImportError, ValueError):
        MISSING.append(package)

if MISSING:

    print("Missing packages:", MISSING)

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            *MISSING
        ],
        "Installing missing dependencies"
    )

else:

    print("All required packages already installed.")

# Import verification.

import numpy
import safetensors
import PIL
import tqdm

from huggingface_hub import hf_hub_download
from safetensors import safe_open

print("NumPy:", numpy.__version__)
print("SafeTensors:", safetensors.__version__)
print("Pillow:", PIL.__version__)
print("tqdm: OK")
print("HuggingFace Hub: OK")

# Confirm that installation did not replace GPU libraries.

assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == GPU_NAME

print("Existing PyTorch/CUDA environment preserved.")

# ================================================================
# 6. DOWNLOAD OFFICIAL GITHUB SOURCE
# ================================================================

print("\n[4/8] DOWNLOADING OFFICIAL TRIPOSPLAT SOURCE")

if SOURCE_DIR.exists():

    if not (SOURCE_DIR / ".git").is_dir():

        raise RuntimeError(
            f"{SOURCE_DIR} exists but is not a Git repository. "
            "Rename or remove it before running again."
        )

    remote = subprocess.check_output(
        [
            "git",
            "-C",
            str(SOURCE_DIR),
            "remote",
            "get-url",
            "origin"
        ],
        text=True
    ).strip()

    if remote.rstrip("/").removesuffix(".git") != (
        GITHUB_URL.rstrip("/").removesuffix(".git")
    ):
        raise RuntimeError(
            "Existing source directory points to a different "
            "GitHub repository."
        )

    print("Official repository already cloned. Reusing it.")

else:

    run_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            GITHUB_URL,
            str(SOURCE_DIR)
        ],
        "Cloning official TripoSplat repository"
    )

# Verify important source files.

REQUIRED_SOURCE_FILES = [
    SOURCE_DIR / "triposplat.py",
    SOURCE_DIR / "model.py"
]

for filepath in REQUIRED_SOURCE_FILES:

    if not filepath.is_file():
        raise RuntimeError(
            f"Official source file missing: {filepath}"
        )

    print("[OK]", filepath.name)

COMMIT = subprocess.check_output(
    [
        "git",
        "-C",
        str(SOURCE_DIR),
        "rev-parse",
        "--short",
        "HEAD"
    ],
    text=True
).strip()

print("Official source commit:", COMMIT)
print("Source directory:", SOURCE_DIR)

# ================================================================
# 7. DOWNLOAD AND INSTALL MODELS
# ================================================================

print("\n[5/8] DOWNLOADING PRETRAINED MODELS")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Calculate required space only for models that are missing
# or invalid. Include a safety reserve for temporary files.

missing_size = 0.0

for item in MODELS:

    destination = MODEL_DIR / item["file"]

    if not validate_safetensors(destination):
        missing_size += item["size_gb"]

required_space = missing_size + 0.75

free_gb = shutil.disk_usage("/content").free / 1024**3

print("Models requiring download:", round(missing_size, 2), "GB")
print("Required with safety reserve:", round(required_space, 2), "GB")
print("Available:", round(free_gb, 2), "GB")

if free_gb < required_space:

    raise RuntimeError(
        "Insufficient disk space for remaining model downloads."
    )

for index, item in enumerate(MODELS, 1):

    filename = item["file"]
    repo = item["repo"]

    destination = MODEL_DIR / filename

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "-" * 60)
    print(f"MODEL {index}/{len(MODELS)}")
    print("Repository:", repo)
    print("File:", filename)

    # Skip existing valid model.

    if validate_safetensors(destination):

        print("Already installed and valid. Skipping.")
        SUCCESSFUL.append(filename)
        continue

    # Remove invalid destination.

    if destination.exists():
        print("Removing invalid existing file.")
        destination.unlink()

    try:

        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(MODEL_DIR)
        )

        downloaded = Path(downloaded)

        if not validate_safetensors(downloaded):

            raise RuntimeError(
                "Downloaded file failed SafeTensors validation."
            )

        # Confirm final destination.

        if downloaded.resolve() != destination.resolve():

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(
                downloaded,
                destination
            )

        if not validate_safetensors(destination):

            raise RuntimeError(
                "Final model destination failed validation."
            )

        print("[SUCCESS] Model installed.")
        print(
            "Size:",
            round(destination.stat().st_size / 1024**3, 3),
            "GB"
        )

        SUCCESSFUL.append(filename)

    except Exception as error:

        print("[FAILED]", str(error))
        FAILED.append(filename)

# ================================================================
# 8. CONNECT STANDALONE CHECKPOINT DIRECTORY
# ================================================================

print("\n[6/8] CONNECTING MODEL DIRECTORIES")

# Official standalone inference expects:
#
# TripoSplat/ckpts/
#   background_removal/
#   diffusion_models/
#   clip_vision/
#   vae/
#
# ComfyUI expects the same folder structure inside models/.
#
# A symbolic link avoids downloading or storing duplicate models.

if CHECKPOINT_LINK.is_symlink():

    current_target = CHECKPOINT_LINK.resolve()

    if current_target != MODEL_DIR.resolve():

        CHECKPOINT_LINK.unlink()

elif CHECKPOINT_LINK.exists():

    if CHECKPOINT_LINK.is_dir() and not any(
        CHECKPOINT_LINK.iterdir()
    ):

        CHECKPOINT_LINK.rmdir()

    else:

        raise RuntimeError(
            "A non-empty ckpts directory already exists. "
            "Refusing to overwrite it."
        )

if not CHECKPOINT_LINK.exists():

    CHECKPOINT_LINK.symlink_to(
        MODEL_DIR,
        target_is_directory=True
    )

if CHECKPOINT_LINK.resolve() != MODEL_DIR.resolve():

    raise RuntimeError("Checkpoint symbolic link failed.")

print("Checkpoint link:", CHECKPOINT_LINK)
print("Linked to:", MODEL_DIR)

# ================================================================
# 9. DOWNLOAD OFFICIAL COMFYUI WORKFLOW
# ================================================================

print("\n[7/8] DOWNLOADING OFFICIAL COMFYUI WORKFLOW")

WORKFLOW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

try:

    workflow_data = download_workflow(
        WORKFLOW_URL,
        WORKFLOW_FILE
    )

    print("[SUCCESS] Workflow downloaded.")
    print("Nodes:", len(workflow_data["nodes"]))
    print(
        "Subgraphs:",
        len(workflow_data["definitions"]["subgraphs"])
    )

    print("Location:", WORKFLOW_FILE)

except Exception as error:

    print("[FAILED] Workflow:", str(error))
    FAILED.append("Official ComfyUI workflow")

# ================================================================
# 10. FINAL VALIDATION
# ================================================================

print("\n[8/8] FINAL INSTALLATION VALIDATION")

print("\nGITHUB SOURCE")

SOURCE_OK = all(
    path.is_file()
    for path in REQUIRED_SOURCE_FILES
)

print("Source:", "PASS" if SOURCE_OK else "FAIL")
print("Commit:", COMMIT)

print("\nMODEL VALIDATION")

MODEL_RESULTS = {}

for item in MODELS:

    filepath = MODEL_DIR / item["file"]

    valid = validate_safetensors(filepath)

    MODEL_RESULTS[item["file"]] = valid

    print(
        "[PASS]" if valid else "[FAIL]",
        item["file"]
    )

MODELS_OK = all(MODEL_RESULTS.values())

print("\nCHECKPOINT LINK")

LINK_OK = (
    CHECKPOINT_LINK.is_symlink()
    and CHECKPOINT_LINK.resolve() == MODEL_DIR.resolve()
)

print("Checkpoint link:", "PASS" if LINK_OK else "FAIL")

print("\nWORKFLOW")

WORKFLOW_OK = False

if WORKFLOW_FILE.is_file():

    try:

        with open(WORKFLOW_FILE, "r") as file:

            check = json.load(file)

        WORKFLOW_OK = (
            isinstance(check.get("nodes"), list)
            and isinstance(
                check.get("definitions", {}).get("subgraphs"),
                list
            )
        )

    except Exception:

        WORKFLOW_OK = False

print("Workflow:", "PASS" if WORKFLOW_OK else "FAIL")

print("\nGPU ENVIRONMENT")

GPU_OK = torch.cuda.is_available()

print("GPU:", GPU_NAME)
print("CUDA:", "PASS" if GPU_OK else "FAIL")
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)

print("\nSTORAGE")

print(
    "Remaining:",
    round(
        shutil.disk_usage("/content").free / 1024**3,
        2
    ),
    "GB"
)

# ================================================================
# FINAL REPORT
# ================================================================

ALL_OK = (
    SOURCE_OK
    and MODELS_OK
    and LINK_OK
    and WORKFLOW_OK
    and GPU_OK
)

print("\n" + "=" * 65)

if ALL_OK:

    print("TRIPOSPLAT INSTALLATION SUCCESSFUL")

    print("\nInstalled components:")
    print("1. Official TripoSplat GitHub source")
    print("2. Required Python dependencies")
    print("3. Five validated pretrained models")
    print("4. ComfyUI model directories")
    print("5. Standalone checkpoint directory link")
    print("6. Official ComfyUI workflow")

    print("\nNEXT STEP:")
    print("Restart ComfyUI before loading the workflow.")

else:

    print("INSTALLATION INCOMPLETE")

    print("\nReview the failed checks above.")

    if FAILED:
        print("Failed downloads:", FAILED)

print("=" * 65)
