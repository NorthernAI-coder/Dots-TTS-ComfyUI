"""ComfyUI node definitions for Dots TTS."""

from __future__ import annotations

import logging

from .languages import LANGUAGE_OPTIONS
from .loader import (
    ATTENTION_OPTIONS,
    DEFAULT_MAX_AUDIO_PATCHES,
    DEFAULT_MODEL,
    DEVICE_OPTIONS,
    DTYPE_OPTIONS,
    get_model_choices,
    load_dotstts_bundle,
)
from .runtime_adapter import generate_dotstts
from .whisper import DotsTTSWhisperTranscribe

logger = logging.getLogger("Dots-TTS-ComfyUI")

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


def _text_input(default: str) -> tuple:
    return (
        "STRING",
        {
            "multiline": True,
            "default": default,
            "tooltip": "Text to synthesize.",
        },
    )


def _reference_text_input() -> tuple:
    return (
        "STRING",
        {
            "multiline": True,
            "default": "",
            "tooltip": "Exact transcript of the reference audio. Recommended for continuation voice cloning.",
        },
    )


def _generation_controls(*, voice_clone: bool = False) -> dict:
    max_audio_tooltip = (
        "Maximum audio budget for this voice-clone generation. One patch is about "
        "0.32 seconds, so 500 is about 160 seconds. The model may stop earlier at "
        "EOS. Prompt audio paired with reference_text also consumes part of this budget."
        if voice_clone
        else
        "Maximum audio budget for this generation. One patch is about 0.32 seconds, "
        "so 500 is about 160 seconds. The model may stop earlier at EOS; increase this "
        "for long text that would otherwise reach the cap."
    )
    return {
        "steps": (
            "INT",
            {
                "default": 10,
                "min": 1,
                "max": 100,
                "step": 1,
                "tooltip": "Flow-matching sampling steps. Upstream default is 10. Higher values can improve quality but slow generation.",
            },
        ),
        "CFG": (
            "FLOAT",
            {
                "default": 1.2,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1,
                "tooltip": "Classifier-free guidance scale. Upstream Dots default is 1.2.",
            },
        ),
        "seed": (
            "INT",
            {
                "default": 42,
                "min": 0,
                "max": 2**31 - 1,
                "tooltip": "Random seed. 0 uses the current random state; positive values make generation repeatable.",
            },
        ),
        "language": (
            LANGUAGE_OPTIONS,
            {
                "default": "auto",
                "tooltip": "Language tag. auto uses Dots language detection; none disables language tags; listed codes match upstream's 24-language multilingual evaluation set.",
            },
        ),
        "normalize_text": (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": "Normalize text before inference when optional normalizer packages are installed. Leave off for exact text testing.",
            },
        ),
        "max_audio_patches": (
            "INT",
            {
                "default": DEFAULT_MAX_AUDIO_PATCHES,
                "min": 1,
                "max": 4096,
                "step": 1,
                "tooltip": max_audio_tooltip,
            },
        ),
    }


class DotsTTSLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    get_model_choices(),
                    {
                        "default": DEFAULT_MODEL,
                        "tooltip": "Cataloged Dots TTS checkpoint. Main weights download to ComfyUI/models/dotstts/<model>; small source-model assets are bundled in the node; shared vocoder/speaker weights download to ComfyUI/models/dotstts/common.",
                    },
                ),
                "device": (
                    DEVICE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Device for Dots TTS. auto uses ComfyUI's torch device; cuda uses NVIDIA GPU; xpu uses Intel XPU if available; cpu forces fp32.",
                    },
                ),
                "dtype": (
                    DTYPE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Weight precision. auto uses the selected checkpoint's native dtype, then applies device guards. CPU always forces fp32.",
                    },
                ),
                "attention": (
                    ATTENTION_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Qwen2 attention implementation. auto uses flash_attention on CUDA when flash_attn is installed, otherwise SDPA. Manual flash_attention requires CUDA and flash_attn.",
                    },
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Download missing Dots assets from Hugging Face. BF16 entries use the drbaph BF16 repos; fp32 entries use the original rednote-hilab repos.",
                    },
                ),
                "compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Experimental CUDA acceleration using native torch.compile with PyTorch Inductor and Triton. Compatible with SDPA, Flash Attention, and ComfyUI's cudaMallocAsync allocator; incompatible CUDA Graph Trees are disabled automatically. The first generation for each max_audio_patches bucket is slower while it compiles; later generations reuse that graph. Changing this setting fully unloads the active model. Requires working Triton, uses extra memory/cache space, and supports up to 1024 audio patches.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("DOTSTTS_MODEL",)
    RETURN_NAMES = ("dotstts_model",)
    FUNCTION = "load"
    CATEGORY = "Dots TTS"
    DESCRIPTION = "Load Dots TTS with ComfyUI/AIMDO memory tracking."

    def load(
        self,
        model: str,
        device: str,
        dtype: str,
        attention: str,
        download_if_missing: bool,
        compile: bool = False,
    ) -> tuple[object]:
        bundle = load_dotstts_bundle(
            model_choice=model,
            dtype_name=dtype,
            device_name=device,
            attention=attention,
            download_if_missing=bool(download_if_missing),
            compile=bool(compile),
        )
        return (bundle,)


class DotsTTSGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "dotstts_model": ("DOTSTTS_MODEL", {"tooltip": "Loaded model from Dots TTS Load Model."}),
            "text": _text_input("Hello! This is Dots TTS running inside ComfyUI."),
        }
        required.update(_generation_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Dots TTS"
    DESCRIPTION = "Generate speech from text with Dots TTS."

    def generate(
        self,
        dotstts_model,
        text: str,
        steps: int,
        CFG: float,
        seed: int,
        language: str,
        normalize_text: bool,
        max_audio_patches: int = DEFAULT_MAX_AUDIO_PATCHES,
    ) -> tuple[dict]:
        pbar = ProgressBar(1) if ProgressBar is not None else None

        def update_progress(current: int, total: int) -> None:
            if pbar is not None:
                pbar.update_absolute(current, total)

        audio = generate_dotstts(
            dotstts_model,
            text=text,
            reference_audio=None,
            reference_text="",
            steps=int(steps),
            cfg=float(CFG),
            seed=int(seed),
            language=language,
            normalize_text=bool(normalize_text),
            max_audio_patches=int(max_audio_patches),
            progress_callback=update_progress,
        )
        return (audio,)


class DotsTTSVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "dotstts_model": ("DOTSTTS_MODEL", {"tooltip": "Loaded model from Dots TTS Load Model."}),
            "reference_audio": (
                "AUDIO",
                {"tooltip": "Reference speaker audio. A clean 3-15 second clip is usually enough."},
            ),
            "text": _text_input("Hello! This is a zero-shot Dots TTS voice clone."),
            "reference_text": _reference_text_input(),
        }
        required.update(_generation_controls(voice_clone=True))
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "Dots TTS"
    DESCRIPTION = "Generate speech with Dots TTS continuation voice cloning."

    def clone(
        self,
        dotstts_model,
        reference_audio: dict,
        text: str,
        reference_text: str,
        steps: int,
        CFG: float,
        seed: int,
        language: str,
        normalize_text: bool,
        max_audio_patches: int = DEFAULT_MAX_AUDIO_PATCHES,
    ) -> tuple[dict]:
        pbar = ProgressBar(1) if ProgressBar is not None else None

        def update_progress(current: int, total: int) -> None:
            if pbar is not None:
                pbar.update_absolute(current, total)

        audio = generate_dotstts(
            dotstts_model,
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            steps=int(steps),
            cfg=float(CFG),
            seed=int(seed),
            language=language,
            normalize_text=bool(normalize_text),
            max_audio_patches=int(max_audio_patches),
            progress_callback=update_progress,
        )
        return (audio,)


NODE_CLASS_MAPPINGS = {
    "DotsTTSLoadModel": DotsTTSLoadModel,
    "DotsTTSGenerate": DotsTTSGenerate,
    "DotsTTSVoiceClone": DotsTTSVoiceClone,
    "DotsTTSWhisperTranscribe": DotsTTSWhisperTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DotsTTSLoadModel": "Dots TTS Load Model",
    "DotsTTSGenerate": "Dots TTS Generate",
    "DotsTTSVoiceClone": "Dots TTS Voice Clone",
    "DotsTTSWhisperTranscribe": "Dots TTS Whisper Transcribe",
}
