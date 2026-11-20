
# ============================================================
# TRELLIS.2 + PIXAL3D — VERIFIED MODEL DOWNLOADER
# Google Colab | ComfyUI | Tesla T4
# ============================================================

import os
import sys
import hashlib
import subprocess
from pathlib import Path

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "huggingface_hub", "safetensors"
])

from huggingface_hub import hf_hub_download
from safetensors import safe_open

COMFYUI = Path("/content/ComfyUI")
MODELS = COMFYUI / "models"

if not COMFYUI.exists():
    raise FileNotFoundError("ComfyUI installation not found.")

# Official SHA256 checksums where published.
# None means validate the SafeTensors structure only.
model_list = [
    {
        "name": "TRELLIS.2 Diffusion",
        "repo": "Comfy-Org/TRELLIS.2",
        "file": "diffusion_models/trellis_2_int8_convrot.safetensors",
        "sha256": "d01952ad137213f6a868f86b6b877026276f84af5eec23069217475a0bad3a31"
    },
    {
        "name": "Pixal3D Diffusion",
        "repo": "Comfy-Org/Pixal3D",
        "file": "diffusion_models/pixal3d_int8_convrot.safetensors",
        "sha256": None
    },
    {
        "name": "TRELLIS.2 Shape VAE",
        "repo": "Comfy-Org/TRELLIS.2",
        "file": "vae/trellis_2_shape_vae_bf16.safetensors",
        "sha256": "de0cb4949a76c59ee5c091a995a69bcc8c51d5aeda939f0c641a50d2a72341f4"
    },
    {
        "name": "TRELLIS.2 Texture VAE",
        "repo": "Comfy-Org/TRELLIS.2",
        "file": "vae/trellis_2_texture_vae_bf16.safetensors",
        "sha256": "714e5ebf094a610e12a8e3b5175c18a62f37f6ea4218acb6073644456b73ab0e"
    },
    {
        "name": "DINOv3 Vision Encoder",
        "repo": "Comfy-Org/Pixal3D",
        "file": "clip_vision/dino_v3_L_naf_fp32.safetensors",
        "sha256": "4ad2ec4e0879a5b5b04cd97325cc37da954a7b6edca5170b86510f17f2b2290f"
    },
    {
        "name": "BiRefNet",
        "repo": "Comfy-Org/BiRefNet",
        "file": "background_removal/birefnet.safetensors",
        "sha256": "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"
    },
    {
        "name": "MoGe 2",
        "repo": "Comfy-Org/MoGe",
        "file": "geometry_estimation/moge_2_vitl_normal_fp16.safetensors",
        "sha256": "cb1a692d03235671e959e81360d7b4d9f44aefadb1f852d6ca6aa17799d5e31f"
    }
]

def validate_file(path, expected_hash=None):
    if not path.exists() or path.stat().st_size < 100:
        return False

    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            next(iter(f.keys()), None)

        if expected_hash:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(
                    lambda: f.read(8 * 1024 * 1024), b""
                ):
                    h.update(chunk)

            if h.hexdigest() != expected_hash:
                return False

        return True

    except Exception:
        return False


successful = []
failed = []

print("=" * 65)
print("TRELLIS.2 + PIXAL3D MODEL DOWNLOADER")
print("=" * 65)

for i, model in enumerate(model_list, 1):

    relative_path = Path(model["file"])
    destination = MODELS / relative_path

    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[{i}/7] {model['name']}")
    print(f"Repository: {model['repo']}")
    print(f"File: {model['file']}")

    # Reuse an existing valid model
    if validate_file(destination, model["sha256"]):
        print("VALID MODEL FOUND — SKIPPING DOWNLOAD")
        successful.append(model["name"])
        continue

    # Remove invalid existing file
    if destination.exists():
        print("Invalid file found. Removing it.")
        destination.unlink()

    try:
        downloaded = hf_hub_download(
            repo_id=model["repo"],
            filename=model["file"],
            local_dir=str(MODELS)
        )

        downloaded = Path(downloaded)

        if downloaded != destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(destination)

        if not validate_file(destination, model["sha256"]):
            raise ValueError(
                "File validation failed: check SafeTensors and SHA256."
            )

        size = destination.stat().st_size / (1024 ** 3)

        print(f"SUCCESS — {size:.2f} GB")
        successful.append(model["name"])

    except Exception as e:
        print(f"FAILED: {e}")
        failed.append((model["name"], str(e)))

# Final report
print("\n" + "=" * 65)
print("INSTALLATION REPORT")
print("=" * 65)

print(f"Successful: {len(successful)}/7")

for name in successful:
    print("[OK]", name)

for name, error in failed:
    print("[FAILED]", name, error)

print("\nFINAL MODEL LOCATIONS:")

for model in model_list:
    path = MODELS / model["file"]

    status = (
        "OK" if validate_file(path, model["sha256"])
        else "MISSING / INVALID"
    )

    print(f"{status}: {path}")

print("\n" + "=" * 65)

if not failed:
    print("ALL SEVEN MODELS VALIDATED SUCCESSFULLY.")
else:
    print("CHECK FAILED MODELS ABOVE AND RETRY.")
