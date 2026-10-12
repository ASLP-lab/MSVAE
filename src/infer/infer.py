"""Reconstruct 48 kHz stereo audio from a 24 kHz input."""

import argparse
import glob
import math
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import yaml

from src.model import DAC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
STRIP_CKPT_PREFIXES = (
    "module.", "model.", "generator.", "ema_generator.", "ema_model.", "online_model.",
)


def strip_ckpt_prefixes(key):
    changed = True
    while changed:
        changed = False
        for prefix in STRIP_CKPT_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def resolve_checkpoint(ckpt_path):
    """Prefer a local checkpoint, otherwise use the cached or downloaded release."""
    path = Path(ckpt_path).expanduser()
    if path.is_file():
        return str(path)
    if path.exists():
        raise ValueError(f"Checkpoint path is not a file: {path}")

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    repo_id, filename = "ASLP-lab/MSVAE", "model.pt"
    print(f"Checkpoint not found: {path}. Looking for {repo_id}/{filename} in the Hugging Face cache.")
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
    except LocalEntryNotFoundError:
        pass
    print(f"Downloading MSVAE weights from {repo_id}/{filename}...")
    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_checkpoint(model, ckpt_path, device):
    ckpt_path = resolve_checkpoint(ckpt_path)
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(ckpt_path, device="cpu")
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    state_dict = {
        strip_ckpt_prefixes(k): v for k, v in state_dict.items() if torch.is_tensor(v)
    }
    # Distillation heads are only needed during training.
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith(("z_projector_1.", "z_projector_2."))
    }
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    missing_muq = [k for k in missing_keys if k.startswith("encoder.")]
    required_missing = [k for k in missing_keys if not k.startswith("encoder.")]
    if required_missing or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint incompatible: missing={required_missing}, unexpected={unexpected_keys}"
        )
    if missing_muq:
        if any(k.startswith("encoder.") for k in state_dict):
            raise RuntimeError("Checkpoint contains only part of the MuQ encoder weights.")
        print("MuQ weights: using the configured pretrained model.")
    print(f"Loaded checkpoint: {ckpt_path} ({len(state_dict)} tensors)")
    return model.to(device)


def preprocess(audio_data, encoder_rates, min_length=0):
    hop_length = int(np.prod(encoder_rates))
    length = audio_data.shape[-1]
    padded_length = max(math.ceil(length / hop_length) * hop_length, min_length)
    return F.pad(audio_data, (0, padded_length - length))


def encode_audio(audio, vae_model, chunked=False, overlap=32, chunk_size=128, latent_mode="sample"):
    """Encode into [B, D, frames]."""
    samples_per_latent = int(vae_model.hop_length)  # 960 samples at 24 kHz
    total_frames = audio.shape[-1] // samples_per_latent
    if not chunked or total_frames <= chunk_size:
        return vae_model.encode_latents(audio, latent_mode).transpose(1, 2)

    hop_size = chunk_size - overlap
    starts = list(range(0, total_frames - chunk_size + 1, hop_size))
    if starts[-1] != total_frames - chunk_size:
        starts.append(total_frames - chunk_size)

    latents = audio.new_zeros(audio.shape[0], vae_model.vae_dim, total_frames)
    print(f"Encoding in {len(starts)} chunks...")
    for i, start in enumerate(starts):
        chunk = audio[:, :, start * samples_per_latent:(start + chunk_size) * samples_per_latent]
        z = vae_model.encode_latents(chunk, latent_mode).transpose(1, 2)
        # Trim half the overlap at each interior edge, as in the original script.
        left = 0 if i == 0 else overlap // 2
        right = chunk_size if i == len(starts) - 1 else chunk_size - overlap // 2
        if z.shape[-1] != chunk_size:
            raise RuntimeError(f"Expected {chunk_size} latent frames, got {z.shape[-1]}")
        latents[:, :, start + left:start + right] = z[:, :, left:right]
    return latents


def decode_audio(latents, vae_model, chunked=False, overlap=32, chunk_size=128):
    """Decode [B, D, frames] into [B, C, samples]."""
    samples_per_latent = int(np.prod(vae_model.decoder.h.upsample_rates))  # 1920 samples at 48 kHz
    total_frames = latents.shape[-1]
    if not chunked or total_frames <= chunk_size:
        return vae_model.decode(latents.transpose(1, 2))

    hop_size = chunk_size - overlap
    starts = list(range(0, total_frames - chunk_size + 1, hop_size))
    if starts[-1] != total_frames - chunk_size:
        starts.append(total_frames - chunk_size)

    audio = latents.new_zeros(latents.shape[0], vae_model.out_channels, total_frames * samples_per_latent)
    print(f"Decoding in {len(starts)} chunks...")
    for i, start in enumerate(starts):
        chunk = latents[:, :, start:start + chunk_size]
        decoded = vae_model.decode(chunk.transpose(1, 2))
        left = (0 if i == 0 else overlap // 2) * samples_per_latent
        right = (chunk_size if i == len(starts) - 1 else chunk_size - overlap // 2) * samples_per_latent
        offset = start * samples_per_latent
        audio[:, :, offset + left:offset + right] = decoded[:, :, left:right]
    return audio


@torch.inference_mode()
def inference(model, files, output_dir, input_sample_rate, output_sample_rate, device,
              chunked=True, overlap=32, chunk_size=128, latent_mode="sample"):
    for file_path in tqdm(files):
        filename = Path(file_path).stem

        audio, _ = librosa.load(file_path, sr=input_sample_rate, mono=False)
        audio = torch.from_numpy(audio).float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).repeat(2, 1)
        elif audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        if audio.shape[0] != model.in_channels or audio.shape[-1] == 0:
            raise ValueError(f"Expected non-empty mono/stereo audio: {file_path}")

        original_length = audio.shape[-1]
        # Padding also lets short audio use the same encoder window as long audio.
        audio = preprocess(audio, model.encoder_rates, min_length=chunk_size * int(model.hop_length))
        audio = audio.unsqueeze(0).to(device)
        output_length = round(original_length * output_sample_rate / input_sample_rate)
        latent = encode_audio(audio, model, chunked, overlap, chunk_size, latent_mode)
        output_path = os.path.join(output_dir, f"{filename}.wav")
        if Path(output_path).resolve() == Path(file_path).resolve():
            raise ValueError(f"Output would overwrite input: {file_path}")
        output = decode_audio(latent, model, chunked, overlap, chunk_size)
        if output.shape[-1] < output_length:
            raise RuntimeError(f"Reconstructed audio is too short: {file_path}")
        output = output[0, :, :output_length].cpu()
        os.makedirs(output_dir, exist_ok=True)
        torchaudio.save(output_path, output, output_sample_rate)
        print(f"Saved {output_path} ({output_sample_rate} Hz, {output_length / output_sample_rate:.3f}s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "src/config/vae/config.yaml"))
    parser.add_argument(
        "--ckpt-path", default=str(PROJECT_ROOT / "model.pt"),
        help="Local checkpoint path; if missing, use model.pt from ASLP-lab/MSVAE on Hugging Face",
    )
    parser.add_argument("--input", required=True, help="Audio file, directory, or text list of audio paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--muq-model", help="Pretrained MuQ repository or local snapshot directory")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--no-chunked", action="store_true", help="Encode/decode the entire audio at once")
    parser.add_argument("--latent-mode", choices=("sample", "mean"), default="sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.chunk_size <= 0 or not 0 <= args.overlap < args.chunk_size or args.overlap % 2:
        parser.error("--overlap must be even and in [0, --chunk-size)")

    input_path = Path(args.input).expanduser().resolve()
    if input_path.is_dir():
        files = sorted(
            path for path in glob.glob(str(input_path / "*"))
            if Path(path).is_file() and Path(path).suffix.lower() in AUDIO_EXTENSIONS
        )
    elif input_path.suffix.lower() in AUDIO_EXTENSIONS:
        files = [str(input_path)]
    else:
        files = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                files.append(str((input_path.parent / Path(line).expanduser()).resolve()))
    if not files:
        raise ValueError(f"No input audio files: {input_path}")
    if len({Path(path).stem for path in files}) != len(files):
        raise ValueError("Input filenames would produce duplicate output WAV paths.")
    for path in files:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    with open(args.model_config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("model_type", "DAC") != "DAC":
        raise ValueError("Only model_type: DAC is supported.")
    dac_config = {key[4:]: value for key, value in config.items() if key.startswith("DAC.")}
    dac_config["bigvgan_conf"] = str(PROJECT_ROOT / Path(dac_config["bigvgan_conf"]).expanduser())
    if args.muq_model:
        dac_config["muq_model_name"] = args.muq_model
    # Use existing cached MuQ weights without requesting Hub metadata.
    muq_name = dac_config.get("muq_model_name", "OpenMuQ/MuQ-large-msd-iter")
    if not Path(muq_name).is_dir():
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            dac_config["muq_model_name"] = snapshot_download(muq_name, local_files_only=True)
        except LocalEntryNotFoundError:
            pass
    input_sample_rate = int(dac_config.get("input_sample_rate", dac_config.get("sample_rate", 24000)))
    output_sample_rate = int(config.get("output_sample_rate", dac_config.get("sample_rate", input_sample_rate)))
    # Also accept the older config with DAC.sample_rate as input and output_sample_rate separate.
    dac_config["input_sample_rate"] = input_sample_rate
    dac_config["sample_rate"] = output_sample_rate
    with open(dac_config["bigvgan_conf"], encoding="utf-8") as handle:
        decoder_config = yaml.safe_load(handle)
    encoder_hop = int(np.prod(dac_config["encoder_rates"]))
    decoder_hop = int(np.prod(decoder_config["upsample_rates"]))
    if input_sample_rate <= 0 or output_sample_rate <= 0 or encoder_hop * output_sample_rate != decoder_hop * input_sample_rate:
        raise ValueError("Sample rates and encoder/decoder hop lengths must preserve duration.")

    torch.manual_seed(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    model = DAC(**dac_config).float()
    model = load_checkpoint(model, args.ckpt_path, device)
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Input: {input_sample_rate} Hz; output: {output_sample_rate} Hz")
    print(f"Latent: {args.latent_mode}")
    inference(
        model, files, args.output_dir, input_sample_rate, output_sample_rate, device,
        chunked=not args.no_chunked, overlap=args.overlap, chunk_size=args.chunk_size,
        latent_mode=args.latent_mode,
    )


if __name__ == "__main__":
    main()
