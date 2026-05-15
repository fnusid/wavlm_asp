import glob
import torch
import numpy as np
import soundfile as sf
import torchaudio
import torch.nn.functional as F
import onnxruntime as ort
from tqdm import tqdm

from model import SpeakerEncoderDualWrapper


def load_audio(path, target_sr=16000, seconds=3):
    wav, sr = sf.read(path)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    wav = torch.tensor(wav, dtype=torch.float32)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    num_samples = target_sr * seconds

    if wav.numel() < num_samples:
        wav = F.pad(wav, (0, num_samples - wav.numel()))
    else:
        wav = wav[:num_samples]

    return wav.unsqueeze(0)


def extract_feature_model():
    model = SpeakerEncoderDualWrapper(emb_dim=256)
    model.eval()
    return model


def run_onnx(sess, features):
    return sess.run(
        ["embeddings"],
        {"features": features.astype(np.float32)},
    )[0]


def compute_slot_and_perm_cosines(y_fp32_np, y_int8_np):
    """
    y_fp32_np: [B, 2, D]
    y_int8_np: [B, 2, D]
    """

    y_fp32 = torch.from_numpy(y_fp32_np).float()
    y_int8 = torch.from_numpy(y_int8_np).float()

    y_fp32 = F.normalize(y_fp32, dim=-1)
    y_int8 = F.normalize(y_int8, dim=-1)

    # Same-order slot comparison
    c00 = F.cosine_similarity(y_fp32[:, 0, :], y_int8[:, 0, :], dim=-1)
    c11 = F.cosine_similarity(y_fp32[:, 1, :], y_int8[:, 1, :], dim=-1)

    same_mean = 0.5 * (c00 + c11)
    same_min = torch.minimum(c00, c11)

    # Swapped-order comparison
    c01 = F.cosine_similarity(y_fp32[:, 0, :], y_int8[:, 1, :], dim=-1)
    c10 = F.cosine_similarity(y_fp32[:, 1, :], y_int8[:, 0, :], dim=-1)

    swap_mean = 0.5 * (c01 + c10)
    swap_min = torch.minimum(c01, c10)

    use_swap = swap_mean > same_mean

    best_mean = torch.where(use_swap, swap_mean, same_mean)
    best_min = torch.where(use_swap, swap_min, same_min)

    return {
        "slot0": c00.detach().cpu().numpy(),
        "slot1": c11.detach().cpu().numpy(),
        "same_mean": same_mean.detach().cpu().numpy(),
        "same_min": same_min.detach().cpu().numpy(),
        "swap_mean": swap_mean.detach().cpu().numpy(),
        "swap_min": swap_min.detach().cpu().numpy(),
        "perm_mean": best_mean.detach().cpu().numpy(),
        "perm_min": best_min.detach().cpu().numpy(),
        "used_swap": use_swap.detach().cpu().numpy(),
    }


def main():
    wav_paths = sorted(
        glob.glob("/home/sidcs/datasets/LibriMix/scripts/calibration_2spk/mix_both/*.wav")
    )

    feature_model = extract_feature_model()

    fp32_sess = ort.InferenceSession(
        "ecapa_feat_fp32.onnx",
        providers=["CPUExecutionProvider"],
    )

    int8_sess = ort.InferenceSession(
        # "ecapa_feat_int8_static_linear_only.onnx",
        # "ecapa_feat_int8_dynamic_linear_only.onnx",
        "ecapa_feat_int8_static_conv_only.onnx",
        providers=["CPUExecutionProvider"],
    )

    slot0_cosines = []
    slot1_cosines = []
    same_mean_cosines = []
    same_min_cosines = []
    perm_mean_cosines = []
    perm_min_cosines = []
    swap_flags = []

    pbar = tqdm(wav_paths, desc="Comparing ONNX outputs")

    for path in pbar:
        audio = load_audio(path, target_sr=16000, seconds=3)

        with torch.no_grad():
            feat = feature_model.encoder.extract_logmel(audio, aug=False)

        feat_np = feat.cpu().numpy().astype(np.float32)

        y_fp32 = run_onnx(fp32_sess, feat_np)  # [1, 2, 256]
        y_int8 = run_onnx(int8_sess, feat_np)  # [1, 2, 256]

        stats = compute_slot_and_perm_cosines(y_fp32, y_int8)

        slot0_cosines.extend(stats["slot0"].tolist())
        slot1_cosines.extend(stats["slot1"].tolist())
        same_mean_cosines.extend(stats["same_mean"].tolist())
        same_min_cosines.extend(stats["same_min"].tolist())
        perm_mean_cosines.extend(stats["perm_mean"].tolist())
        perm_min_cosines.extend(stats["perm_min"].tolist())
        swap_flags.extend(stats["used_swap"].tolist())

    print()
    print("===== Same-order slot-wise comparison =====")
    print("Slot 0 mean cosine:", np.mean(slot0_cosines))
    print("Slot 0 min cosine: ", np.min(slot0_cosines))
    print("Slot 1 mean cosine:", np.mean(slot1_cosines))
    print("Slot 1 min cosine: ", np.min(slot1_cosines))

    print()
    print("===== Same-order pair comparison =====")
    print("Same-order mean cosine:", np.mean(same_mean_cosines))
    print("Same-order min cosine: ", np.min(same_min_cosines))

    print()
    print("===== Permutation-aware pair comparison =====")
    print("Perm-aware mean cosine:", np.mean(perm_mean_cosines))
    print("Perm-aware min cosine: ", np.min(perm_min_cosines))
    print("Swap rate:", np.mean(swap_flags))


if __name__ == "__main__":
    main()