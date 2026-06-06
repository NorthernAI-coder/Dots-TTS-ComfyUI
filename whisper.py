"""Optional Whisper transcription node for Dots TTS reference text."""

from __future__ import annotations

import logging
import warnings
import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .loader import (
    _empty_accelerator_cache,
    _hard_release_module,
    register_unload_callback,
)

logger = logging.getLogger("Dots-TTS-ComfyUI")

WHISPER_DTYPE_OPTIONS = ["auto", "bf16", "fp32"]
WHISPER_TASK_OPTIONS = ["transcribe", "translate"]
WHISPER_LANGUAGE_OPTIONS = [
    "auto",
    "english",
    "chinese",
    "japanese",
    "korean",
    "french",
    "german",
    "spanish",
    "portuguese",
    "russian",
    "italian",
    "hindi",
    "arabic",
]

POPULAR_WHISPER_MODELS = {
    "whisper-large-v3-turbo (auto-download)": "openai/whisper-large-v3-turbo",
    "whisper-large-v3 (auto-download)": "openai/whisper-large-v3",
    "whisper-medium (auto-download)": "openai/whisper-medium",
    "whisper-small (auto-download)": "openai/whisper-small",
    "whisper-tiny (auto-download)": "openai/whisper-tiny",
}

_PIPELINE_CACHE: dict[tuple[str, str, str], Any] = {}
_ACTIVE_PIPELINE_KEY: tuple[str, str, str] | None = None
HF_ENDPOINT = "https://huggingface.co"


def _safe_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")


def audio_encoders_dir() -> Path:
    try:
        import folder_paths

        base = Path(folder_paths.models_dir) / "audio_encoders"
    except Exception:
        base = Path(__file__).resolve().parent / "models" / "audio_encoders"
    base.mkdir(parents=True, exist_ok=True)
    return base


def register_audio_encoders_folder() -> None:
    try:
        import folder_paths

        base = str(audio_encoders_dir())
        if "audio_encoders" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("audio_encoders", base)
    except Exception:
        pass


def _has_whisper_files(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    try:
        return any(item.is_file() and item.suffix in {".safetensors", ".bin", ".pt", ".pth"} for item in path.iterdir())
    except OSError:
        return False


def whisper_model_choices() -> list[str]:
    choices = list(POPULAR_WHISPER_MODELS)
    known = {_safe_repo_name(repo_id) for repo_id in POPULAR_WHISPER_MODELS.values()}
    try:
        for entry in sorted(audio_encoders_dir().iterdir()):
            if entry.is_dir() and entry.name not in known and _has_whisper_files(entry):
                choices.append(entry.name)
    except OSError:
        pass
    return choices


def _download_whisper(repo_id: str, download_if_missing: bool) -> Path:
    dest = audio_encoders_dir() / _safe_repo_name(repo_id)
    if _has_whisper_files(dest):
        return dest
    if not download_if_missing:
        raise FileNotFoundError(f"Whisper model is missing at {dest}. Enable download_if_missing.")

    from huggingface_hub import snapshot_download

    logger.info("Downloading Whisper model %s to %s", repo_id, dest)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            ignore_patterns=["*.msgpack", "*.h5", "tf_model*", "flax_model*"],
            endpoint=HF_ENDPOINT,
        )
    if not _has_whisper_files(dest):
        raise RuntimeError(f"Whisper download finished, but usable files were not found at {dest}.")
    return dest


def _resolve_whisper_path(model_name: str, download_if_missing: bool) -> Path:
    if model_name in POPULAR_WHISPER_MODELS:
        return _download_whisper(POPULAR_WHISPER_MODELS[model_name], download_if_missing)
    path = audio_encoders_dir() / model_name
    if _has_whisper_files(path):
        return path
    repo_id = model_name.replace("_", "/", 1)
    if "/" in repo_id:
        return _download_whisper(repo_id, download_if_missing)
    raise FileNotFoundError(f"Whisper model not found under {audio_encoders_dir()}: {model_name}")


def _resolve_device() -> str:
    try:
        import comfy.model_management as mm

        device = torch.device(mm.get_torch_device())
    except Exception:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")
    if device.type == "cuda":
        return f"cuda:{device.index or 0}"
    if device.type == "xpu":
        return f"xpu:{device.index or 0}"
    return "cpu"


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        if device.startswith("xpu"):
            return torch.bfloat16
        return torch.float32
    if device == "cpu" and dtype != "fp32":
        logger.warning("Whisper CPU device selected; forcing fp32 because bf16 CPU transcription is unsafe/slow.")
        return torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported Whisper dtype: {dtype}")


def _set_pipeline_device(pipe: Any, device: str) -> None:
    torch_device = torch.device(device)
    for attr in ("device", "_device"):
        try:
            setattr(pipe, attr, torch_device)
            return
        except Exception:
            pass


def _unload_whisper_pipeline(pipe: Any) -> None:
    try:
        if hasattr(pipe, "model") and pipe.model is not None:
            model = pipe.model
            _hard_release_module(model, None)
            pipe.model = None
    except Exception:
        pass
    for attr in (
        "tokenizer",
        "feature_extractor",
        "processor",
        "assistant_model",
        "generation_config",
    ):
        try:
            if hasattr(pipe, attr):
                setattr(pipe, attr, None)
        except Exception:
            pass


def unload_whisper_cache(reason: str = "manual unload") -> None:
    global _ACTIVE_PIPELINE_KEY
    if _PIPELINE_CACHE:
        logger.info("Unloading Whisper ASR cache (%s).", reason)
    for pipe in list(_PIPELINE_CACHE.values()):
        _unload_whisper_pipeline(pipe)
    _PIPELINE_CACHE.clear()
    _ACTIVE_PIPELINE_KEY = None
    gc.collect()
    _empty_accelerator_cache(trim_process=True)


register_unload_callback(unload_whisper_cache)


def get_whisper_pipeline(model_name: str, dtype: str, download_if_missing: bool):
    global _ACTIVE_PIPELINE_KEY

    register_audio_encoders_folder()
    device = _resolve_device()
    key = (model_name, dtype, device)
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        if hasattr(cached, "model") and cached.model is not None:
            cached.model.to(torch.device(device))
            _set_pipeline_device(cached, device)
        _ACTIVE_PIPELINE_KEY = key
        return cached

    if _ACTIVE_PIPELINE_KEY is not None and _ACTIVE_PIPELINE_KEY != key:
        logger.info("Unloading Whisper ASR pipeline before loading changed settings.")
        for pipe in list(_PIPELINE_CACHE.values()):
            _unload_whisper_pipeline(pipe)
        _PIPELINE_CACHE.clear()
        _ACTIVE_PIPELINE_KEY = None
        gc.collect()
        try:
            import comfy.model_management as mm

            mm.soft_empty_cache()
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()

    from transformers import pipeline as hf_pipeline

    model_path = _resolve_whisper_path(model_name, download_if_missing)
    torch_dtype = _resolve_dtype(dtype, device)
    initial_device = "cpu" if torch.device(device).type != "cpu" else device
    logger.info("Loading Whisper ASR from %s on %s with %s", model_path, device, torch_dtype)
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=str(model_path),
        torch_dtype=torch_dtype,
        device=initial_device,
    )
    try:
        pipe.model.to(torch.device(device))
        _set_pipeline_device(pipe, device)
        logger.debug("Whisper ASR ready on %s.", device)
    except Exception as exc:
        logger.warning("Could not move Whisper ASR to %s; falling back to CPU: %s", device, exc)
        pipe.model.to(torch.device("cpu"))
        _set_pipeline_device(pipe, "cpu")
    _PIPELINE_CACHE[key] = pipe
    _ACTIVE_PIPELINE_KEY = key
    return pipe


def comfy_audio_to_numpy(audio: dict) -> tuple[np.ndarray, int]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    wav = waveform[0].detach().float().cpu()
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    return wav.numpy().astype(np.float32, copy=False), sample_rate


def transcribe_audio(
    audio: dict,
    model_name: str,
    dtype: str,
    language: str,
    task: str,
    chunk_length_s: int,
    download_if_missing: bool,
) -> str:
    pipe = get_whisper_pipeline(model_name, dtype, download_if_missing)
    audio_np, sample_rate = comfy_audio_to_numpy(audio)
    generate_kwargs: dict[str, str] = {"task": task}
    if language != "auto":
        generate_kwargs["language"] = language
    kwargs: dict[str, Any] = {"generate_kwargs": generate_kwargs}
    if chunk_length_s > 0:
        kwargs["chunk_length_s"] = int(chunk_length_s)
    result = pipe({"array": audio_np, "sampling_rate": sample_rate}, **kwargs)
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


class DotsTTSWhisperTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Reference audio to transcribe for Dots TTS reference_text."}),
                "model": (
                    whisper_model_choices(),
                    {
                        "default": "whisper-large-v3-turbo (auto-download)",
                        "tooltip": "Whisper ASR model. Turbo is fast and usually accurate enough for reference transcripts.",
                    },
                ),
                "dtype": (
                    WHISPER_DTYPE_OPTIONS,
                    {"default": "auto", "tooltip": "Whisper precision. auto uses bf16 on supported CUDA/XPU and fp32 otherwise."},
                ),
                "language": (
                    WHISPER_LANGUAGE_OPTIONS,
                    {"default": "auto", "tooltip": "Reference audio language. auto detects it; setting it can improve transcript accuracy."},
                ),
                "task": (
                    WHISPER_TASK_OPTIONS,
                    {"default": "transcribe", "tooltip": "transcribe keeps the original language; translate outputs English."},
                ),
                "chunk_length_s": (
                    "INT",
                    {
                        "default": 30,
                        "min": 0,
                        "max": 120,
                        "step": 1,
                        "tooltip": "Whisper chunk length for longer reference clips. 0 lets Transformers choose.",
                    },
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Download the selected Whisper model into ComfyUI/models/audio_encoders if it is missing."},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcript",)
    FUNCTION = "transcribe"
    CATEGORY = "Dots TTS"
    DESCRIPTION = "Transcribe reference AUDIO with Whisper for Dots TTS voice cloning."

    def transcribe(
        self,
        audio: dict,
        model: str,
        dtype: str,
        language: str,
        task: str,
        chunk_length_s: int,
        download_if_missing: bool,
    ) -> tuple[str]:
        text = transcribe_audio(audio, model, dtype, language, task, int(chunk_length_s), bool(download_if_missing))
        logger.debug("Whisper transcript: %s", text)
        return (text,)
