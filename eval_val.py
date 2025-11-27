import torch
from train import MySpEmb
from dataset import LibriMixDataModule
from metrics import EmbeddingMetrics

CKPT_OLD = "/mnt/disks/data/model_ckpts/librispeech_asp_wavlm_ft_dualemb_queryorthogonality_mhqa_lib2mix_tr360_trfs_valwham/best-epoch=50-val_separation=0.000.ckpt"

DATA_ROOT = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix"
SPEAKER_MAP = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libriuni_05_08/Libri2Mix_ovl50to80/wav16k/min/metadata/train360_mapping.json"

# 1) Rebuild datamodule exactly like now
dm = LibriMixDataModule(
    data_root=DATA_ROOT,
    speaker_map_path=SPEAKER_MAP,
    batch_size=10,
    num_workers=20,
    num_speakers=2,
)
dm.setup("validate")
val_loader = dm.val_dataloader()

# 2) Rebuild model skeleton (same hyperparams as old run)
model = MySpEmb(
    lr=1e-4,
    finetune_encoder=False,
    emb_dim=256,
    speaker_map_path=SPEAKER_MAP,
)

ckpt = torch.load(CKPT_OLD, map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()
model.cuda()

metrics = EmbeddingMetrics(device="cuda")

all_embs = []
all_labels = []

with torch.no_grad():
    for mix, source, labels in val_loader:
        mix = mix.cuda()
        emb, Q = model(mix, return_queries=True)
        all_embs.append(emb.cpu())
        all_labels.append(labels.cpu())

import torch
embs_flat = torch.cat(all_embs, dim=0)   # [B,2,D]
labels_flat = torch.cat(all_labels, dim=0)

results = metrics.compute_from_tensors(embs_flat, labels_flat)
print(results)