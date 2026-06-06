"""Safe dependency helper for Dots-TTS-ComfyUI.

ComfyUI owns the Torch/CUDA stack. This installer installs only lightweight
runtime helpers and does not auto-upgrade torch, torchaudio, torchvision,
transformers, or pydantic.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys


PREFIX = "[Dots-TTS-ComfyUI]"

PROTECTED = {
    "torch": "ComfyUI manages PyTorch and CUDA/ROCm wheels.",
    "torchaudio": "Often tied exactly to the PyTorch build; Dots has fallback paths where possible.",
    "torchvision": "Must match the local PyTorch build when installed.",
    "transformers": "Upgrading can break other custom nodes; Dots upstream recommends >=4.57.0.",
    "pydantic": "Upgrading can break other custom nodes; Dots needs pydantic v2 semantics.",
}

RUNTIME_PACKAGES = [
    ("huggingface-hub>=0.26.0", "huggingface_hub", "huggingface-hub", "0.26.0"),
    ("safetensors", "safetensors", "safetensors", None),
    ("soundfile>=0.13.1", "soundfile", "soundfile", "0.13.1"),
    ("numpy", "numpy", "numpy", None),
    ("scipy", "scipy", "scipy", None),
    ("einops", "einops", "einops", None),
    ("loguru", "loguru", "loguru", None),
    ("PyYAML", "yaml", "PyYAML", None),
    ("torchdiffeq", "torchdiffeq", "torchdiffeq", None),
]

OPTIONAL_TEXT_PACKAGES = [
    "langcodes[data]",
    "lingua-language-detector",
    "WeTextProcessing",
]


def _run(cmd: list[str], timeout: int = 900) -> bool:
    print(f"{PREFIX} Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        print(f"{PREFIX} Install command failed: {exc}")
        return False
    if result.returncode == 0:
        return True
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return False


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in version.replace("-", ".").split("."):
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts)


def _version_at_least(dist_name: str, minimum: str) -> bool:
    try:
        installed = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return _version_tuple(installed) >= _version_tuple(minimum)


def _pip_install(requirement: str, *, no_deps: bool = True) -> bool:
    flags = ["--no-deps"] if no_deps else []
    uv_cmd = [sys.executable, "-m", "uv", "pip", "install", requirement] + flags
    if _run(uv_cmd):
        return True
    pip_cmd = [sys.executable, "-m", "pip", "install", requirement] + flags
    return _run(pip_cmd)


def _ensure(requirement: str, module_name: str, dist_name: str, minimum: str | None = None) -> None:
    installed = _has_module(module_name)
    if installed and minimum is not None:
        installed = _version_at_least(dist_name, minimum)
    if installed:
        print(f"{PREFIX} {requirement} already available.")
        return
    print(f"{PREFIX} Installing {requirement} with --no-deps.")
    if not _pip_install(requirement, no_deps=True):
        print(f"{PREFIX} WARNING: Failed to install {requirement}. Try manually:")
        print(f"  {sys.executable} -m pip install {requirement} --no-deps")


def _protected_status() -> None:
    print(f"{PREFIX} Protected package check:")
    for module_name, reason in PROTECTED.items():
        if _has_module(module_name):
            try:
                version = importlib.metadata.version(module_name)
            except Exception:
                version = "installed"
            print(f"{PREFIX}   {module_name}: {version} - {reason}")
        else:
            print(f"{PREFIX}   {module_name}: missing - {reason}")
    try:
        transformers_version = importlib.metadata.version("transformers")
        if _version_tuple(transformers_version) < (4, 57, 0):
            print(f"{PREFIX} WARNING: transformers {transformers_version} is older than upstream's recommended >=4.57.0.")
            print(f"{PREFIX}          Not upgrading automatically. If Dots import fails, upgrade transformers intentionally.")
    except Exception:
        print(f"{PREFIX} WARNING: transformers is not installed. Install a Comfy-compatible transformers build if Dots cannot load.")
    try:
        pydantic_version = importlib.metadata.version("pydantic")
        if _version_tuple(pydantic_version) < (2, 0, 0):
            print(f"{PREFIX} WARNING: pydantic {pydantic_version} is older than Dots' v2 config API.")
            print(f"{PREFIX}          Not upgrading automatically because pydantic can affect other nodes.")
    except Exception:
        print(f"{PREFIX} WARNING: pydantic is not installed.")


def main() -> None:
    print("=" * 72)
    print(f"{PREFIX} Safe dependency install")
    print("=" * 72)
    _protected_status()

    if not _has_module("torch"):
        print(f"{PREFIX} ERROR: torch is missing. Repair the ComfyUI Python environment first.")
        return

    for requirement, module_name, dist_name, minimum in RUNTIME_PACKAGES:
        _ensure(requirement, module_name, dist_name, minimum)

    print(f"{PREFIX} Optional text normalization packages are not auto-installed:")
    for pkg in OPTIONAL_TEXT_PACKAGES:
        print(f"{PREFIX}   {pkg}")
    print(f"{PREFIX} Dots TTS will still run without them; normalize_text/auto language detection may be limited.")
    print(f"{PREFIX} Install check complete. Restart ComfyUI.")


if __name__ == "__main__":
    main()

