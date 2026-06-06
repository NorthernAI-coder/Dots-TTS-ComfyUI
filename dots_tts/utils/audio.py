"""Audio helpers used by the retained train/infer pipeline."""

from __future__ import annotations

import torch

try:
    import torchaudio.compliance.kaldi as Kaldi
    import torchaudio.functional as AF
except Exception:
    Kaldi = None
    AF = None


def high_quality_resample(x, orig_sr, target_sr):
    if int(orig_sr) == int(target_sr):
        return x
    if AF is not None:
        return AF.resample(
            x,
            orig_freq=orig_sr,
            new_freq=target_sr,
            lowpass_filter_width=64,
            rolloff=0.95,
            resampling_method="sinc_interp_kaiser",
        )
    try:
        from scipy import signal

        gcd = torch.gcd(torch.tensor(int(orig_sr)), torch.tensor(int(target_sr))).item()
        up = int(target_sr) // int(gcd)
        down = int(orig_sr) // int(gcd)
        y = signal.resample_poly(x.detach().cpu().numpy(), up, down, axis=-1)
        return torch.from_numpy(y).to(device=x.device, dtype=x.dtype)
    except Exception:
        new_length = max(1, round(x.shape[-1] * int(target_sr) / int(orig_sr)))
        was_1d = x.ndim == 1
        interp_input = x.view(1, 1, -1) if was_1d else x.unsqueeze(0)
        y = torch.nn.functional.interpolate(
            interp_input,
            size=new_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
        return y.squeeze(0) if was_1d else y


def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, device: torch.device) -> torch.Tensor:
    min_mel = _hz_to_mel(torch.tensor(20.0, device=device))
    max_mel = _hz_to_mel(torch.tensor(float(sample_rate) / 2.0, device=device))
    mel_points = torch.linspace(min_mel, max_mel, n_mels + 2, device=device)
    hz_points = _mel_to_hz(mel_points)
    bins = torch.floor((n_fft + 1) * hz_points / float(sample_rate)).long()
    filters = torch.zeros(n_mels, n_fft // 2 + 1, device=device)
    for i in range(n_mels):
        left = int(bins[i].item())
        center = int(bins[i + 1].item())
        right = int(bins[i + 2].item())
        if center > left:
            filters[i, left:center] = torch.linspace(0, 1, center - left, device=device)
        if right > center:
            filters[i, center:right] = torch.linspace(1, 0, right - center, device=device)
    return filters


def _fallback_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    n_mels: int,
    mean_norm: bool,
) -> torch.Tensor:
    if waveform.ndim == 2:
        waveform = waveform[0]
    waveform = waveform.float()
    frame_length = max(16, round(sample_rate * 0.025))
    hop_length = max(1, round(sample_rate * 0.010))
    if waveform.numel() < frame_length:
        waveform = torch.nn.functional.pad(waveform, (0, frame_length - waveform.numel()))
    n_fft = 1
    while n_fft < frame_length:
        n_fft *= 2
    window = torch.hann_window(frame_length, device=waveform.device, dtype=waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        window=window,
        center=False,
        return_complex=True,
    )
    power = spec.abs().pow(2).transpose(0, 1)
    filters = _mel_filterbank(sample_rate, n_fft, n_mels, waveform.device).to(power.dtype)
    features = torch.matmul(power, filters.t()).clamp_min(1.0e-10).log()
    if mean_norm:
        features = features - features.mean(dim=0, keepdim=True)
    return features


def extract_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    n_mels: int,
    dither: float = 0.0,
    mean_norm: bool = False,
) -> torch.Tensor:
    if Kaldi is None:
        return _fallback_fbank(
            waveform,
            sample_rate=sample_rate,
            n_mels=n_mels,
            mean_norm=mean_norm,
        )
    if waveform.ndim == 1:
        feature_input = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        feature_input = waveform if waveform.size(0) == 1 else waveform[0:1, :]
    else:
        raise ValueError(
            f"FBank expects a 1D or 2D waveform, got shape {tuple(waveform.shape)}."
        )
    features = Kaldi.fbank(
        feature_input,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
        dither=dither,
    )
    if mean_norm:
        features = features - features.mean(dim=0, keepdim=True)
    return features
