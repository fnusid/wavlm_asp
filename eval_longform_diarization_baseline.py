import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm


EPS = 1e-8

MODEL_SOURCES = {
    "ecapa": "speechbrain/spkrec-ecapa-voxceleb",
    "xvector": "speechbrain/spkrec-xvect-voxceleb",
}


def l2_normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x) + EPS)


def cluster_accuracy(pred_labels, true_labels):
    pred = np.asarray(pred_labels)
    true = np.asarray(true_labels)

    total = 0
    for cluster_id in np.unique(pred):
        mask = pred == cluster_id
        subset = true[mask]
        if len(subset) == 0:
            continue
        total += Counter(subset).most_common(1)[0][1]

    return total / max(1, len(true))


def compute_id_switches(true_labels, pred_ids, segment_ids):
    by_spk = defaultdict(list)
    for true_spk, pred_id, seg_id in zip(true_labels, pred_ids, segment_ids):
        by_spk[true_spk].append((seg_id, pred_id))

    switches = 0
    transitions = 0
    for _, items in by_spk.items():
        items = sorted(items, key=lambda x: x[0])
        prev = None
        for _, pred_id in items:
            if prev is not None:
                transitions += 1
                if pred_id != prev:
                    switches += 1
            prev = pred_id

    return switches, transitions, switches / max(1, transitions)


def compute_reid_accuracy(true_labels, pred_ids):
    by_spk = defaultdict(list)
    for true_spk, pred_id in zip(true_labels, pred_ids):
        by_spk[true_spk].append(pred_id)

    correct = 0
    total = 0
    for _, assigned_ids in by_spk.items():
        if len(assigned_ids) < 2:
            continue
        majority_id = Counter(assigned_ids).most_common(1)[0][0]
        for pred_id in assigned_ids:
            total += 1
            if pred_id == majority_id:
                correct += 1

    return correct / max(1, total)


def load_audio(path, target_sr=16000):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        raise ValueError(f"{path} has sample rate {sr}, expected {target_sr}")
    return wav


def extract_segments(wav, timeline_df, scope, sr=16000):
    records = []
    chunks = []

    for _, row in timeline_df.iterrows():
        num_active = int(row["num_active"])
        active_speakers = str(row["active_speakers"]).split("|")

        if scope == "single" and num_active != 1:
            continue
        if scope == "single":
            per_row_speakers = [active_speakers[0]]
        else:
            # A single diarization embedding from a mixed overlap segment is ambiguous.
            # We keep this mode disabled at the CLI for now.
            raise ValueError("scope='all' is not supported for this baseline.")

        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        start = int(round(start_sec * sr))
        end = int(round(end_sec * sr))
        chunk = wav[start:end]
        if chunk.size == 0:
            continue

        for true_speaker in per_row_speakers:
            records.append(
                {
                    "segment_id": int(row["segment_id"]),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "num_active": num_active,
                    "true_speaker": str(true_speaker),
                }
            )
            chunks.append(chunk)

    return records, chunks


def load_encoder(model_name, device):
    source = MODEL_SOURCES[model_name]
    savedir = f"/tmp/{source.split('/')[-1]}"
    classifier = EncoderClassifier.from_hparams(
        source=source,
        savedir=savedir,
        run_opts={"device": device},
    )
    return classifier


def embed_chunks(classifier, chunks, batch_size, device):
    all_embs = []
    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch = torch.stack([torch.from_numpy(chunk) for chunk in batch_chunks], dim=0)
        batch = batch.to(device)
        with torch.no_grad():
            emb = classifier.encode_batch(batch).squeeze(1)
            emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def cluster_embeddings(embs, n_clusters, backend):
    if backend == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        return model.fit_predict(embs)

    if backend == "spectral":
        affinity = np.clip(np.matmul(embs, embs.T), -1.0, 1.0)
        affinity = (affinity + 1.0) / 2.0
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=0,
            assign_labels="kmeans",
        )
        return model.fit_predict(affinity)

    raise ValueError(f"Unknown clustering backend: {backend}")


def evaluate_one_meeting(meeting_row, classifier, args):
    meeting_id = meeting_row["meeting_id"]
    wav = load_audio(meeting_row["mix_path"], target_sr=args.sample_rate)
    timeline_df = pd.read_csv(meeting_row["timeline_csv"])

    records, chunks = extract_segments(
        wav=wav,
        timeline_df=timeline_df,
        scope=args.scope,
        sr=args.sample_rate,
    )
    if not records:
        return None, []

    embs = embed_chunks(
        classifier=classifier,
        chunks=chunks,
        batch_size=args.batch_size,
        device=args.device,
    )
    embs = np.asarray([l2_normalize_np(x) for x in embs], dtype=np.float32)

    true_labels = np.asarray([rec["true_speaker"] for rec in records])
    if args.oracle_num_speakers:
        n_clusters = int(meeting_row["num_global_speakers"])
    else:
        n_clusters = len(np.unique(true_labels))
    n_clusters = min(n_clusters, len(records))

    pred_ids = cluster_embeddings(embs, n_clusters=n_clusters, backend=args.cluster_backend)

    for rec, pred_id in zip(records, pred_ids):
        rec["meeting_id"] = meeting_id
        rec["pred_cluster_id"] = int(pred_id)
        rec["embedding_backend"] = args.embedding_backend
        rec["cluster_backend"] = args.cluster_backend
        rec["eval_scope"] = args.scope

    segment_ids = np.asarray([rec["segment_id"] for rec in records])
    switches, transitions, switch_rate = compute_id_switches(true_labels, pred_ids, segment_ids)
    metrics = {
        "meeting_id": meeting_id,
        "embedding_backend": args.embedding_backend,
        "cluster_backend": args.cluster_backend,
        "eval_scope": args.scope,
        "num_records": len(records),
        "num_true_speakers_eval": int(len(np.unique(true_labels))),
        "num_global_speakers_manifest": int(meeting_row["num_global_speakers"]),
        "num_pred_clusters": int(len(np.unique(pred_ids))),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_ids)),
        "ari": float(adjusted_rand_score(true_labels, pred_ids)),
        "cluster_acc": float(cluster_accuracy(pred_ids, true_labels)),
        "id_switches": int(switches),
        "id_transitions": int(transitions),
        "id_switch_rate": float(switch_rate),
        "reid_accuracy": float(compute_reid_accuracy(true_labels, pred_ids)),
    }
    return metrics, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default="/home/sidcs/datasets/LibriMix/long_meetings_test/manifest.csv",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--meeting_id", type=str, default=None)
    parser.add_argument(
        "--embedding_backend",
        type=str,
        choices=sorted(MODEL_SOURCES.keys()),
        default="ecapa",
    )
    parser.add_argument(
        "--cluster_backend",
        type=str,
        choices=["agglomerative", "spectral"],
        default="agglomerative",
    )
    parser.add_argument(
        "--scope",
        type=str,
        choices=["single"],
        default="single",
        help="Evaluate only single-speaker oracle segments.",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--oracle_num_speakers",
        action="store_true",
        help="Cluster using the manifest's known number of global speakers.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(args.manifest_csv)
    if args.meeting_id is not None:
        manifest_df = manifest_df[manifest_df["meeting_id"].astype(str) == str(args.meeting_id)].copy()

    classifier = load_encoder(args.embedding_backend, args.device)

    metrics_rows = []
    assignment_rows = []
    for _, meeting_row in tqdm(manifest_df.iterrows(), total=len(manifest_df)):
        metrics, records = evaluate_one_meeting(meeting_row, classifier, args)
        if metrics is None:
            continue
        metrics_rows.append(metrics)
        assignment_rows.extend(records)

    metrics_df = pd.DataFrame(metrics_rows)
    assignments_df = pd.DataFrame(assignment_rows)

    base = f"{args.embedding_backend}_{args.cluster_backend}_{args.scope}"
    metrics_path = out_dir / f"{base}_metrics.csv"
    assignments_path = out_dir / f"{base}_assignments.csv"
    metrics_df.to_csv(metrics_path, index=False)
    assignments_df.to_csv(assignments_path, index=False)

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved assignments to: {assignments_path}")
    if len(metrics_df) > 0:
        print(metrics_df.mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
