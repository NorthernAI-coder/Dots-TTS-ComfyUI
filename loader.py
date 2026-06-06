"""Dots TTS model loading and ComfyUI memory registration."""

from __future__ import annotations

import gc
import importlib.util
import logging
import math
import os
import shutil
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .dots_tts.runtime import DotsTtsRuntime

logger = logging.getLogger("Dots-TTS-ComfyUI")

MODEL_FOLDER_NAME = "dotstts"
HF_ENDPOINT = "https://huggingface.co"
COMMON_HEAVY_REPO_ID = "drbaph/dots.tts-common"

DTYPE_OPTIONS = ["auto", "bf16", "fp16", "fp32"]
ATTENTION_OPTIONS = ["auto", "sdpa", "flash_attention"]
DEVICE_OPTIONS = ["auto", "cuda", "cpu", "xpu"]
DEFAULT_MAX_AUDIO_PATCHES = 500

SMALL_ASSET_FILES = [
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "latent_stats.pt",
    "llm_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
HEAVY_COMMON_FILES = [
    "speaker_encoder.safetensors",
    "vocoder.safetensors",
]
REQUIRED_RUNTIME_FILES = [
    "config.json",
    "latent_stats.pt",
    "llm_config.json",
    "model.safetensors",
    "speaker_encoder.safetensors",
    "vocoder.safetensors",
]

MODEL_CATALOG: dict[str, dict[str, str]] = {
    "dots.tts Base FP32 (auto-download)": {
        "repo_id": "rednote-hilab/dots.tts-base",
        "heavy_repo_id": COMMON_HEAVY_REPO_ID,
        "heavy_fallback_repo_id": "rednote-hilab/dots.tts-base",
        "slug": "dots.tts-base",
        "model_precision": "float32",
        "description": "Pretrained dots.tts checkpoint.",
    },
    "dots.tts SOAR FP32 (auto-download)": {
        "repo_id": "rednote-hilab/dots.tts-soar",
        "heavy_repo_id": COMMON_HEAVY_REPO_ID,
        "heavy_fallback_repo_id": "rednote-hilab/dots.tts-soar",
        "slug": "dots.tts-soar",
        "model_precision": "float32",
        "description": "Self-corrective-aligned dots.tts checkpoint.",
    },
    "dots.tts MF FP32 (auto-download)": {
        "repo_id": "rednote-hilab/dots.tts-mf",
        "heavy_repo_id": COMMON_HEAVY_REPO_ID,
        "heavy_fallback_repo_id": "rednote-hilab/dots.tts-mf",
        "slug": "dots.tts-mf",
        "model_precision": "float32",
        "description": "MeanFlow-distilled dots.tts checkpoint.",
    },
    "dots.tts Base BF16 (auto-download)": {
        "repo_id": "drbaph/dots.tts-base-bf16",
        "assets_repo_id": "rednote-hilab/dots.tts-base",
        "assets_slug": "dots.tts-base",
        "heavy_repo_id": COMMON_HEAVY_REPO_ID,
        "heavy_fallback_repo_id": "rednote-hilab/dots.tts-base",
        "slug": "dots.tts-base-bf16",
        "model_filename": "dots.tts-base-bf16.safetensors",
        "local_model_file": "dots.tts-base-bf16.safetensors",
        "model_precision": "bfloat16",
        "description": "BF16 Dots TTS checkpoint converted by drbaph.",
    },
    "dots.tts SOAR BF16 (auto-download)": {
        "repo_id": "drbaph/dots.tts-soar-bf16",
        "assets_repo_id": "rednote-hilab/dots.tts-soar",
        "assets_slug": "dots.tts-soar",
        "heavy_repo_id": COMMON_HEAVY_REPO_ID,
        "heavy_fallback_repo_id": "rednote-hilab/dots.tts-soar",
        "slug": "dots.tts-soar-bf16",
        "model_filename": "dots.tts-soar-bf16.safetensors",
        "local_model_file": "dots.tts-soar-bf16.safetensors",
        "model_precision": "bfloat16",
        "description": "BF16 Dots TTS checkpoint converted by drbaph.",
    },
    "dots.tts MF BF16 (auto-download)": {
        "repo_id": "drbaph/dots.tts-mf-bf16",
        "assets_repo_id": "rednote-hilab/dots.tts-mf",
        "assets_slug": "dots.tts-mf",
        "heavy_repo_id": COMMON_HEAVY_REPO_ID,
        "heavy_fallback_repo_id": "rednote-hilab/dots.tts-mf",
        "slug": "dots.tts-mf-bf16",
        "model_filename": "dots.tts-mf-bf16.safetensors",
        "local_model_file": "dots.tts-mf-bf16.safetensors",
        "model_precision": "bfloat16",
        "description": "BF16 Dots TTS checkpoint converted by drbaph.",
    },
}
MODEL_ALIASES = {
    "dots.tts Base - rednote-hilab/dots.tts-base": "dots.tts Base FP32 (auto-download)",
    "dots.tts SOAR - rednote-hilab/dots.tts-soar": "dots.tts SOAR FP32 (auto-download)",
    "dots.tts MF - rednote-hilab/dots.tts-mf": "dots.tts MF FP32 (auto-download)",
    "dots.tts Base BF16 - drbaph/dots.tts-base-bf16": "dots.tts Base BF16 (auto-download)",
    "dots.tts SOAR BF16 - drbaph/dots.tts-soar-bf16": "dots.tts SOAR BF16 (auto-download)",
    "dots.tts MF BF16 - drbaph/dots.tts-mf-bf16": "dots.tts MF BF16 (auto-download)",
}
DEFAULT_MODEL = "dots.tts SOAR BF16 (auto-download)"

_ACTIVE_BUNDLE: "DotsTTSBundle | None" = None
_ACTIVE_LOAD_KEY: tuple[Any, ...] | None = None
_UNLOAD_CALLBACKS: list[Any] = []


@dataclass
class DotsTTSBundle:
    runtime: DotsTtsRuntime | None
    patchers: list[Any]
    model_dir: Path
    asset_dir: Path
    repo_id: str
    device: torch.device
    dtype_name: str
    precision: str
    attention: str
    attn_implementation: str | None
    model_choice: str = ""
    dtype_choice: str = "auto"
    device_choice: str = "auto"
    download_if_missing: bool = True


def _same_device(a: torch.device, b: torch.device) -> bool:
    a = torch.device(a)
    b = torch.device(b)
    return a.type == b.type and (a.index or 0) == (b.index or 0)


def _module_unique_tensors(module: torch.nn.Module) -> list[torch.Tensor]:
    seen: set[int] = set()
    tensors: list[torch.Tensor] = []
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        ident = id(tensor)
        if ident in seen:
            continue
        seen.add(ident)
        tensors.append(tensor)
    return tensors


def _non_meta_tensor_bytes(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    total = 0
    for tensor in _module_unique_tensors(module):
        if tensor.device.type == "meta":
            continue
        total += tensor.nelement() * tensor.element_size()
    return total


class DotsTTSVBar:
    """Page-level tensor residency view for ComfyUI-MemoryVisualization."""

    page_size: int = 32 * 1024 * 1024

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = torch.device(device)
        self.tensors: list[torch.Tensor] = []
        self.total_size = 0
        self.total_pages = 1
        self.watermark = 0
        self._refresh_tensors()

    @property
    def offset(self) -> int:
        return self.total_size

    def _refresh_tensors(self) -> None:
        self.tensors = _module_unique_tensors(self.model)
        self.total_size = sum(
            tensor.nelement() * tensor.element_size()
            for tensor in self.tensors
            if tensor.device.type != "meta"
        )
        self.total_pages = max(1, math.ceil(self.total_size / self.page_size)) if self.total_size > 0 else 0

    def loaded_size(self) -> int:
        self._refresh_tensors()
        return sum(
            tensor.nelement() * tensor.element_size()
            for tensor in self.tensors
            if tensor.device.type != "meta" and _same_device(tensor.device, self.device)
        )

    def get_residency(self) -> list[int]:
        self._refresh_tensors()
        if self.total_size <= 0:
            return []
        residency = [0 for _ in range(self.total_pages)]
        cursor = 0
        for tensor in self.tensors:
            if tensor.device.type == "meta":
                continue
            size = tensor.nelement() * tensor.element_size()
            if size <= 0:
                continue
            if _same_device(tensor.device, self.device):
                start_page = cursor // self.page_size
                end_page = min(self.total_pages - 1, (cursor + size - 1) // self.page_size)
                for page in range(start_page, end_page + 1):
                    residency[page] |= 1
            cursor += size
        return residency

    def get_watermark(self) -> int:
        self.watermark = max(self.watermark, self.loaded_size())
        return self.watermark

    def prioritize(self) -> None:
        self.watermark = self.loaded_size()


try:
    import comfy.model_patcher as _model_patcher

    class DotsTTSPatcher(_model_patcher.ModelPatcher):
        def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
            super().__init__(model, load_device, offload_device, size, weight_inplace_update)
            self._ensure_dynamic_state(load_device)

        def is_dynamic(self):
            return True

        def _ensure_dynamic_state(self, device):
            device = torch.device(device)
            if not hasattr(self.model, "dynamic_vbars"):
                self.model.dynamic_vbars = {}
            if not hasattr(self.model, "dynamic_pins"):
                self.model.dynamic_pins = {}
            if device not in self.model.dynamic_pins:
                try:
                    import comfy_aimdo.host_buffer

                    empty_hostbuf = comfy_aimdo.host_buffer.HostBuffer(0, 0, 0)
                except Exception:
                    empty_hostbuf = None
                self.model.dynamic_pins[device] = {
                    "weights": (empty_hostbuf, [], [-1], [0], [0], {}),
                    "patches": (empty_hostbuf, [], [-1], [0], [0], {}),
                    "hostbufs_initialized": False,
                    "failed": False,
                    "active": False,
                }

        def _vbar_get(self):
            vbars = getattr(self.model, "dynamic_vbars", {})
            if vbars:
                return next(iter(vbars.values()))
            return None

        def loaded_size(self):
            vbar = self._vbar_get()
            if vbar is not None:
                return vbar.loaded_size()
            return getattr(self.model, "model_loaded_weight_memory", 0)

        def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
            device_to = torch.device(device_to)
            self._ensure_dynamic_state(device_to)
            before = self.loaded_size()
            self.model.to(device_to)
            self.model.model_loaded_weight_memory = (
                self.model_size() if self._vbar_get() is None else 0
            )
            return max(0, self.loaded_size() - before)

        def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
            before = self.loaded_size()
            self.detach()
            return before

        def detach(self, unpatch_all=True):
            try:
                self.model.to(self.offload_device)
                self.model.model_loaded_weight_memory = 0
                if hasattr(self.model, "dynamic_vbars"):
                    self.model.dynamic_vbars.clear()
            except Exception:
                pass
            empty_cache = globals().get("_empty_accelerator_cache")
            if callable(empty_cache):
                empty_cache()
            return self.model

        def current_loaded_device(self):
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                return self.offload_device

        def loaded_ram_size(self):
            return 0

        def pinned_memory_size(self):
            return 0

        def unregister_inactive_pins(self, ram_to_unload, subsets=["weights", "patches"]):
            return 0

        def partially_unload_ram(self, ram_to_unload, subsets=["weights", "patches"]):
            return 0

    del _model_patcher
except Exception:
    DotsTTSPatcher = None


def _empty_accelerator_cache(*, trim_process: bool = False) -> None:
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    if trim_process:
        _trim_process_memory()


def _trim_process_memory() -> None:
    try:
        import ctypes
        import os
        import platform

        if platform.system() == "Windows":
            try:
                ctypes.CDLL("msvcrt")._heapmin()
            except Exception:
                pass
            try:
                kernel32 = ctypes.windll.kernel32
                kernel32.GetCurrentProcess.restype = ctypes.c_void_p
                kernel32.SetProcessWorkingSetSize.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                ]
                kernel32.SetProcessWorkingSetSize(
                    kernel32.GetCurrentProcess(),
                    ctypes.c_size_t(-1),
                    ctypes.c_size_t(-1),
                )
            except Exception:
                pass
        elif os.name == "posix":
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
    except Exception:
        pass


def node_dir() -> Path:
    return Path(__file__).resolve().parent


def assets_root() -> Path:
    path = node_dir() / "assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_root() -> Path:
    path = node_dir() / "runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir() -> Path:
    try:
        import folder_paths

        base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
    except Exception:
        base = node_dir() / "models" / MODEL_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def register_model_folder() -> None:
    try:
        import folder_paths

        base = str(model_dir())
        if MODEL_FOLDER_NAME not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, base)
        logger.info("Dots TTS model folder registered: %s", base)
    except Exception:
        pass


def get_model_choices() -> list[str]:
    return list(MODEL_CATALOG)


def resolve_model_choice(model_choice: str) -> tuple[str, dict[str, str]]:
    canonical = MODEL_ALIASES.get(model_choice, model_choice)
    spec = MODEL_CATALOG.get(canonical)
    if spec is None:
        raise ValueError(f"Unsupported Dots TTS model choice: {model_choice}")
    return canonical, spec


def _weight_dir_for(spec: dict[str, str]) -> Path:
    return model_dir() / spec["slug"]


def _runtime_dir_for(spec: dict[str, str]) -> Path:
    return runtime_root() / spec["slug"]


def _assets_model_dir(spec: dict[str, str]) -> Path:
    return assets_root() / spec.get("assets_slug", spec["slug"])


def _heavy_common_dir() -> Path:
    path = model_dir() / "common"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _has_files(base: Path, files: list[str]) -> bool:
    return all((base / name).is_file() for name in files)


def _download_asset_snapshot(repo_id: str, target: Path, allow_patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Dots TTS runtime assets from %s to %s", repo_id, target)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        allow_patterns=allow_patterns,
        ignore_patterns=["model.safetensors"],
        endpoint=HF_ENDPOINT,
    )


def ensure_node_assets(spec: dict[str, str], download_if_missing: bool) -> Path:
    specific = _assets_model_dir(spec)
    assets_repo_id = spec.get("assets_repo_id", spec["repo_id"])
    if not _has_files(specific, SMALL_ASSET_FILES):
        if not download_if_missing:
            missing = [name for name in SMALL_ASSET_FILES if not (specific / name).is_file()]
            raise FileNotFoundError(
                f"Missing Dots TTS small runtime assets in {specific}: {missing}. "
                "Enable download_if_missing or pre-populate this node's assets folder."
            )
        _download_asset_snapshot(assets_repo_id, specific, SMALL_ASSET_FILES)
    return specific


def ensure_heavy_common_assets(spec: dict[str, str], download_if_missing: bool) -> Path:
    heavy = _heavy_common_dir()
    missing = [name for name in HEAVY_COMMON_FILES if not (heavy / name).is_file()]
    if not missing:
        return heavy
    if not download_if_missing:
        raise FileNotFoundError(
            f"Missing Dots TTS shared weight assets in {heavy}: {missing}. Enable download_if_missing."
        )
    from huggingface_hub import hf_hub_download

    heavy_repo_id = spec.get("heavy_repo_id", spec["repo_id"])
    fallback_repo_id = spec.get("heavy_fallback_repo_id")
    for filename in missing:
        repos = [heavy_repo_id]
        if fallback_repo_id and fallback_repo_id not in repos:
            repos.append(fallback_repo_id)
        last_error: Exception | None = None
        for repo_id in repos:
            try:
                logger.info("Downloading Dots TTS shared asset %s from %s to %s", filename, repo_id, heavy)
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(heavy),
                    endpoint=HF_ENDPOINT,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if repo_id == heavy_repo_id and fallback_repo_id:
                    logger.warning(
                        "Could not download %s from %s; trying fallback %s: %s",
                        filename,
                        heavy_repo_id,
                        fallback_repo_id,
                        exc,
                    )
        if last_error is not None:
            raise last_error
    return heavy


def _files_match(src: Path, dst: Path) -> bool:
    if not src.is_file() or not dst.is_file():
        return False
    try:
        if os.path.samefile(src, dst):
            return True
    except Exception:
        pass
    try:
        src_stat = src.stat()
        dst_stat = dst.stat()
        return src_stat.st_size == dst_stat.st_size and src_stat.st_mtime_ns == dst_stat.st_mtime_ns
    except Exception:
        return False


def _link_or_copy_file(src: Path, dst: Path, *, replace: bool = False) -> None:
    if dst.is_file():
        if not replace or _files_match(src, dst):
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _assemble_runtime_dir(
    spec: dict[str, str],
    model_file: Path,
    specific: Path,
    heavy: Path,
) -> Path:
    runtime_dir = _runtime_dir_for(spec)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for rel in SMALL_ASSET_FILES:
        src = specific / rel
        if not src.is_file():
            continue
        dst = runtime_dir / rel
        _link_or_copy_file(src, dst, replace=True)
    for rel in HEAVY_COMMON_FILES:
        src = heavy / rel
        if not src.is_file():
            continue
        _link_or_copy_file(src, runtime_dir / rel, replace=True)
    _link_or_copy_file(model_file, runtime_dir / "model.safetensors", replace=True)
    return runtime_dir


def _download_model_file(spec: dict[str, str], weight_dir: Path, download_if_missing: bool) -> Path:
    model_filename = spec.get("model_filename", "model.safetensors")
    model_path = weight_dir / model_filename
    if model_path.is_file():
        return model_path

    legacy_model_path = weight_dir / "model.safetensors"
    if model_filename == "model.safetensors" and legacy_model_path.is_file():
        return legacy_model_path

    local_model_file = spec.get("local_model_file")
    if local_model_file:
        local_model_path = weight_dir / local_model_file
        if local_model_path.is_file():
            return local_model_path
        if legacy_model_path.is_file():
            _link_or_copy_file(legacy_model_path, model_path)
            return model_path
        if spec.get("local_only") == "true":
            raise FileNotFoundError(
                f"Missing local test checkpoint at {local_model_path}. "
                "This temporary BF16 entry is not downloaded from the upstream fp32 repo."
            )
    if not download_if_missing:
        raise FileNotFoundError(
            f"Missing Dots TTS model weights at {model_path}. Enable download_if_missing."
        )
    from huggingface_hub import hf_hub_download

    repo_id = spec["repo_id"]
    logger.info("Downloading Dots TTS %s from %s to %s", model_filename, repo_id, weight_dir)
    weight_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        local_dir=str(weight_dir),
        endpoint=HF_ENDPOINT,
    )
    downloaded_path = weight_dir / model_filename
    if not downloaded_path.is_file():
        raise FileNotFoundError(f"Dots TTS model download did not produce {downloaded_path}")
    return downloaded_path


def resolve_runtime_dir(model_choice: str, download_if_missing: bool) -> tuple[Path, Path, str, Path]:
    _, spec = resolve_model_choice(model_choice)

    weight_dir = _weight_dir_for(spec)
    specific = ensure_node_assets(spec, download_if_missing)
    heavy = ensure_heavy_common_assets(spec, download_if_missing)
    model_file = _download_model_file(spec, weight_dir, download_if_missing)
    runtime_dir = _assemble_runtime_dir(spec, model_file, specific, heavy)

    missing = [name for name in REQUIRED_RUNTIME_FILES if not (runtime_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Dots TTS runtime directory {runtime_dir} is missing required files: {missing}")
    return runtime_dir, _assets_model_dir(spec), spec["repo_id"], model_file


def _auto_device() -> torch.device:
    try:
        import comfy.model_management as mm

        return torch.device(mm.get_torch_device())
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def resolve_device(device_name: str = "auto") -> torch.device:
    choice = (device_name or "auto").strip().lower()
    if choice == "auto":
        return _auto_device()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA was selected for Dots TTS, but CUDA is not available. Falling back to auto device.")
        return _auto_device()
    if choice == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        logger.warning("XPU was selected for Dots TTS, but torch.xpu is not available. Falling back to auto device.")
        return _auto_device()
    raise ValueError(f"Unsupported device: {device_name}")


def _guard_precision_for_device(precision: str, device: torch.device, *, source: str) -> str:
    if device.type == "cpu":
        if precision != "float32":
            logger.warning("Dots TTS CPU device selected; forcing fp32 because bf16/fp16 CPU inference is unsafe/slow.")
        return "float32"
    if precision == "bfloat16" and device.type == "cuda":
        try:
            if not torch.cuda.is_bf16_supported():
                logger.warning(
                    "%s requested bf16, but this CUDA device does not report bf16 support. Falling back to fp16.",
                    source,
                )
                return "float16"
        except Exception:
            pass
    if precision == "float16" and device.type == "xpu":
        logger.warning("fp16 on XPU may be unsupported or unstable. bf16 is usually the safer XPU choice.")
    return precision


def resolve_precision(dtype_name: str, device: torch.device, model_precision: str | None = None) -> str:
    if device.type == "cpu":
        if dtype_name not in {"auto", "fp32"}:
            logger.warning("Dots TTS CPU device selected; forcing fp32 because bf16/fp16 CPU inference is unsafe/slow.")
        return "float32"
    if dtype_name == "auto":
        if model_precision:
            precision = _guard_precision_for_device(model_precision, device, source="Selected Dots TTS checkpoint")
            logger.info("Dots TTS dtype auto resolved to %s from selected checkpoint.", precision)
            return precision
        if device.type == "cuda":
            try:
                precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            except Exception:
                precision = "float16"
            logger.info("Dots TTS dtype auto resolved to %s from CUDA capability.", precision)
            return precision
        if device.type == "xpu":
            logger.info("Dots TTS dtype auto resolved to bfloat16 for XPU.")
            return "bfloat16"
        logger.info("Dots TTS dtype auto resolved to float32.")
        return "float32"
    if dtype_name == "bf16":
        return _guard_precision_for_device("bfloat16", device, source="Dots TTS dtype setting")
    if dtype_name == "fp16":
        return _guard_precision_for_device("float16", device, source="Dots TTS dtype setting")
    if dtype_name == "fp32":
        return "float32"
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_attention(attention: str, device: torch.device | None = None) -> tuple[str, str | None]:
    flash_available = importlib.util.find_spec("flash_attn") is not None
    device = torch.device(device or "cpu")
    if attention == "auto":
        if device.type == "cuda" and flash_available:
            logger.info("Dots TTS attention auto resolved to flash_attention.")
            return "flash_attention", "flash_attention_2"
        reason = "flash_attn is not installed" if not flash_available else f"device is {device.type}"
        logger.info("Dots TTS attention auto resolved to sdpa (%s).", reason)
        return "sdpa", "sdpa"
    if attention == "sdpa":
        logger.info("Dots TTS attention resolved to sdpa.")
        return "sdpa", "sdpa"
    if attention == "flash_attention":
        if device.type != "cuda":
            raise RuntimeError("flash_attention was selected, but the selected Dots TTS device is not CUDA.")
        if not flash_available:
            raise ImportError("flash_attention was selected, but flash_attn is not installed.")
        logger.info("Dots TTS attention resolved to flash_attention.")
        return "flash_attention", "flash_attention_2"
    raise ValueError(f"Unsupported attention mode: {attention}")


def _register_with_comfy(patcher: Any) -> None:
    if patcher is None:
        return
    try:
        import comfy.model_management as mm

        if patcher.load_device.type == "cpu":
            return
        if any(loaded.model is patcher for loaded in mm.current_loaded_models):
            return
        raw = patcher.model
        if hasattr(patcher, "_ensure_dynamic_state"):
            patcher._ensure_dynamic_state(patcher.load_device)
        raw.dynamic_vbars = {patcher.load_device: DotsTTSVBar(raw, patcher.load_device)}
        raw.model_loaded_weight_memory = 0
        loaded = mm.LoadedModel(patcher)
        loaded.real_model = weakref.ref(raw)
        loaded.model_finalizer = weakref.finalize(raw, mm.cleanup_models)
        loaded.model_finalizer.atexit = False
        loaded.currently_used = True
        mm.current_loaded_models.insert(0, loaded)
        logger.info(
            "Registered %s with ComfyUI/AIMDO (%.1f MB).",
            raw.__class__.__name__,
            patcher.model_size() / (1024 * 1024),
        )
    except Exception as exc:
        logger.warning("Could not register Dots TTS with ComfyUI memory tracking: %s", exc)


def _unregister_from_comfy(patcher: Any) -> None:
    try:
        import comfy.model_management as mm

        survivors = []
        for loaded in mm.current_loaded_models:
            if loaded.model is patcher:
                try:
                    loaded.model_unload(memory_to_free=1e32, unpatch_weights=True)
                except Exception:
                    pass
                try:
                    if loaded.model_finalizer is not None:
                        loaded.model_finalizer.detach()
                    loaded.model_finalizer = None
                    loaded.real_model = None
                except Exception:
                    pass
                continue
            survivors.append(loaded)
        mm.current_loaded_models[:] = survivors
    except Exception:
        pass


def _empty_dynamic_pin_state() -> dict[str, Any]:
    try:
        import comfy_aimdo.host_buffer

        empty_hostbuf = comfy_aimdo.host_buffer.HostBuffer(0, 0, 0)
    except Exception:
        empty_hostbuf = None
    return {
        "weights": (empty_hostbuf, [], [-1], [0], [0], {}),
        "patches": (empty_hostbuf, [], [-1], [0], [0], {}),
        "hostbufs_initialized": False,
        "failed": False,
        "active": False,
    }


def _reset_dynamic_state(module: torch.nn.Module, device: torch.device | None = None) -> None:
    try:
        module.model_loaded_weight_memory = 0
    except Exception:
        pass
    try:
        module.current_weight_patches_uuid = None
    except Exception:
        pass
    try:
        if hasattr(module, "dynamic_vbars"):
            module.dynamic_vbars.clear()
    except Exception:
        pass
    try:
        if device is not None:
            module.dynamic_pins = {torch.device(device): _empty_dynamic_pin_state()}
        elif hasattr(module, "dynamic_pins"):
            module.dynamic_pins.clear()
    except Exception:
        pass


def _clear_lowvram_runtime_attrs(module: torch.nn.Module) -> None:
    for child in module.modules():
        if getattr(child, "_pin_registered", False) and hasattr(child, "_pin"):
            try:
                pin = child._pin
                size = pin.numel() * pin.element_size()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.cudart().cudaHostUnregister(pin.data_ptr())
                    except Exception:
                        pass
                try:
                    import comfy.model_management as mm

                    mm.TOTAL_PINNED_MEMORY = max(0, getattr(mm, "TOTAL_PINNED_MEMORY", 0) - size)
                except Exception:
                    pass
            except Exception:
                pass
        for attr in list(vars(child)):
            if (
                attr in {"_pin", "_pin_balancer_entry", "_v"}
                or attr.endswith("_lowvram_function")
                or attr.endswith("_function")
            ):
                try:
                    delattr(child, attr)
                except Exception:
                    try:
                        setattr(child, attr, None)
                    except Exception:
                        pass
        for attr in ("_pin_registered", "comfy_cast_weights"):
            if hasattr(child, attr):
                try:
                    setattr(child, attr, False)
                except Exception:
                    pass


def _clear_patcher_refs(patcher: Any) -> None:
    for attr in (
        "backup",
        "backup_buffers",
        "patches",
        "object_patches",
        "object_patches_backup",
        "weight_wrapper_patches",
        "callbacks",
        "wrappers",
        "current_hooks",
        "forced_hooks",
        "hook_patches",
        "cached_hook_patches",
    ):
        value = getattr(patcher, attr, None)
        if hasattr(value, "clear"):
            try:
                value.clear()
            except Exception:
                pass
    for attr in ("non_dynamic_delegate_model", "parent"):
        if hasattr(patcher, attr):
            try:
                setattr(patcher, attr, None)
            except Exception:
                pass


def _hard_release_module(module: torch.nn.Module | None, device: torch.device | None = None) -> None:
    if module is None:
        return
    try:
        _clear_lowvram_runtime_attrs(module)
    except Exception:
        pass
    _reset_dynamic_state(module, device)
    try:
        module.to_empty(device=torch.device("meta"))
    except Exception:
        try:
            module.to(torch.device("cpu"))
        except Exception:
            pass
    remaining = _non_meta_tensor_bytes(module)
    if remaining:
        logger.warning(
            "Dots TTS hard-release left %.2f MB of real tensor storage on %s.",
            remaining / (1024 * 1024),
            module.__class__.__name__,
        )


def register_runtime_module(module: torch.nn.Module, device: torch.device) -> Any:
    device = torch.device(device)
    if DotsTTSPatcher is None or device.type == "cpu":
        module.to(device)
        return None
    patcher = DotsTTSPatcher(
        module,
        load_device=device,
        offload_device=torch.device("cpu"),
    )
    module.model_loaded_weight_memory = 0
    return patcher


def unload_runtime_module(patcher: Any, *, hard: bool = True) -> None:
    if patcher is None:
        return
    module = getattr(patcher, "model", None)
    load_device = getattr(patcher, "load_device", None)
    _unregister_from_comfy(patcher)
    if hard:
        try:
            patcher.unpin_all_weights()
        except Exception:
            pass
        try:
            patcher.partially_unload_ram(1e32)
        except Exception:
            pass
        try:
            patcher.partially_unload(getattr(patcher, "offload_device", torch.device("cpu")), 1e32)
        except Exception:
            pass
        try:
            patcher.detach(unpatch_all=True)
        except Exception:
            pass
        _clear_patcher_refs(patcher)
        _hard_release_module(module, torch.device(load_device) if load_device is not None else None)
        return
    try:
        patcher.detach()
    except Exception:
        pass


def resume_runtime_module(patcher: Any, device: torch.device) -> None:
    if patcher is None:
        return
    patcher.partially_load(torch.device(device))
    _register_with_comfy(patcher)


def resume_bundle_to_device(bundle: DotsTTSBundle) -> None:
    if bundle.patchers:
        for patcher in list(bundle.patchers):
            resume_runtime_module(patcher, bundle.device)
    elif bundle.runtime is not None:
        bundle.runtime.model.to(bundle.device)
    if bundle.runtime is not None:
        bundle.runtime.device = bundle.device


def unload_dotstts_bundle(bundle: DotsTTSBundle | None, reason: str = "manual unload", hard: bool = True) -> None:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY
    if bundle is None:
        return
    logger.info("Unloading Dots TTS bundle (%s).", reason)
    runtime = bundle.runtime
    model = runtime.model if runtime is not None else None
    if bundle.patchers:
        for patcher in list(bundle.patchers):
            unload_runtime_module(patcher, hard=hard)
        bundle.patchers.clear()
    elif runtime is not None:
        try:
            runtime.model.to("cpu")
        except Exception:
            pass
    try:
        if model is not None:
            for attr in ("core", "vocoder", "xvector_extractor"):
                _hard_release_module(getattr(model, attr, None), bundle.device)
            _hard_release_module(model, bundle.device)
            for attr in ("core", "vocoder", "xvector_extractor", "tokenizer"):
                try:
                    setattr(model, attr, None)
                except Exception:
                    pass
            try:
                compiled = getattr(model, "_compiled_models", None)
                if hasattr(compiled, "clear"):
                    compiled.clear()
            except Exception:
                pass
    except Exception:
        pass
    if hard:
        try:
            if runtime is not None:
                runtime.model = None
        except Exception:
            pass
        try:
            bundle.runtime = None
        except Exception:
            pass
    gc.collect()
    _empty_accelerator_cache(trim_process=hard)
    if _ACTIVE_BUNDLE is bundle:
        _ACTIVE_BUNDLE = None
        _ACTIVE_LOAD_KEY = None


def register_unload_callback(callback: Any) -> None:
    if callback not in _UNLOAD_CALLBACKS:
        _UNLOAD_CALLBACKS.append(callback)


def unload_all_dotstts_state(reason: str = "ComfyUI manual unload") -> None:
    unload_dotstts_bundle(_ACTIVE_BUNDLE, reason=reason, hard=True)
    for callback in list(_UNLOAD_CALLBACKS):
        try:
            callback(reason)
        except Exception as exc:
            logger.warning("Dots TTS unload callback failed: %s", exc)
    gc.collect()
    _empty_accelerator_cache(trim_process=True)


def _active_bundle_owns(obj: Any) -> bool:
    if _ACTIVE_BUNDLE is None or obj is None:
        return False
    owned: list[Any] = list(_ACTIVE_BUNDLE.patchers)
    runtime = _ACTIVE_BUNDLE.runtime
    model = runtime.model if runtime is not None else None
    if model is not None:
        owned.extend(
            [
                model,
                getattr(model, "core", None),
                getattr(model, "vocoder", None),
                getattr(model, "xvector_extractor", None),
            ]
        )
    if any(existing is obj for existing in owned if existing is not None):
        return True
    nested = getattr(obj, "model", None)
    if nested is not None and any(existing is nested for existing in owned if existing is not None):
        return True
    try:
        if callable(nested):
            resolved = nested()
            return any(existing is resolved for existing in owned if existing is not None)
    except Exception:
        pass
    return False


def install_comfy_unload_hook() -> None:
    try:
        import comfy.model_management as mm
    except Exception:
        return

    if getattr(mm, "_dotstts_unload_hook_installed", False):
        return

    original_unload_all_models = mm.unload_all_models

    def unload_all_models_with_dotstts(*args, **kwargs):
        try:
            return original_unload_all_models(*args, **kwargs)
        finally:
            unload_all_dotstts_state("ComfyUI unload_all_models")

    mm.unload_all_models = unload_all_models_with_dotstts

    original_unload_model_and_clones = getattr(mm, "unload_model_and_clones", None)
    if original_unload_model_and_clones is not None:
        def unload_model_and_clones_with_dotstts(model, *args, **kwargs):
            try:
                return original_unload_model_and_clones(model, *args, **kwargs)
            finally:
                if _active_bundle_owns(model):
                    unload_all_dotstts_state("ComfyUI unload_model_and_clones")

        mm.unload_model_and_clones = unload_model_and_clones_with_dotstts

    mm._dotstts_unload_hook_installed = True
    logger.info("Installed Dots TTS hard-unload hook for ComfyUI manual unload.")


def load_dotstts_bundle(
    model_choice: str,
    dtype_name: str,
    device_name: str,
    attention: str,
    download_if_missing: bool,
) -> DotsTTSBundle:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY

    register_model_folder()
    canonical_model_choice, spec = resolve_model_choice(model_choice)
    runtime_dir, asset_dir, repo_id, weight_file = resolve_runtime_dir(model_choice, bool(download_if_missing))
    device = resolve_device(device_name)
    precision = resolve_precision(dtype_name, device, spec.get("model_precision"))
    runtime_attention, attn_impl = resolve_attention(attention, device)
    load_key = (
        str(runtime_dir.resolve()),
        str(weight_file.resolve()),
        weight_file.stat().st_mtime_ns,
        str(device),
        precision,
        runtime_attention,
        device_name,
    )
    if _ACTIVE_BUNDLE is not None and _ACTIVE_LOAD_KEY == load_key:
        resume_bundle_to_device(_ACTIVE_BUNDLE)
        return _ACTIVE_BUNDLE
    if _ACTIVE_BUNDLE is not None:
        unload_dotstts_bundle(_ACTIVE_BUNDLE, reason="load settings changed")

    logger.info(
        "Loading Dots TTS from %s on %s with precision=%s attention=%s.",
        runtime_dir,
        device,
        precision,
        runtime_attention,
    )
    initial_device = torch.device("cpu") if device.type != "cpu" else device
    runtime = DotsTtsRuntime.from_pretrained(
        str(runtime_dir),
        precision=precision,
        device=initial_device,
        attn_implementation=attn_impl,
        optimize=False,
    )
    runtime.device = device
    patchers: list[Any] = []
    for module in (runtime.model.core, runtime.model.vocoder, runtime.model.xvector_extractor):
        patcher = register_runtime_module(module, device)
        if patcher is not None:
            patchers.append(patcher)
    bundle = DotsTTSBundle(
        runtime=runtime,
        patchers=patchers,
        model_dir=runtime_dir,
        asset_dir=asset_dir,
        repo_id=repo_id,
        device=device,
        dtype_name=dtype_name,
        precision=precision,
        attention=runtime_attention,
        attn_implementation=attn_impl,
        model_choice=canonical_model_choice,
        dtype_choice=dtype_name,
        device_choice=device_name,
        download_if_missing=bool(download_if_missing),
    )
    _ACTIVE_BUNDLE = bundle
    _ACTIVE_LOAD_KEY = load_key
    resume_bundle_to_device(bundle)
    _empty_accelerator_cache()
    logger.info("Dots TTS loaded successfully.")
    return bundle
