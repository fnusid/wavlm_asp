import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class EmbeddingMetrics:
    def __init__(self, device="cuda"):
        self.device = device

    def _flatten_embs_labels(self, embs, labels):
        """
        embs:   [B, D] or [B, S, D]
        labels: [B]    or [B, S]
        -> return [N, D], [N]
        """
        if embs.dim() == 3:
            B, S, D = embs.shape
            embs = embs.reshape(B * S, D)
        if labels.dim() > 1:
            labels = labels.reshape(-1)
        return embs, labels

    def extract_embeddings(self, model, val_loader):
        model.eval()
        all_embeddings, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                wavs, labels = batch          # labels: [B] or [B,2]
                wavs = wavs.to(self.device)

                embs = model(wavs)            # [B,D] or [B,2,D]
                embs = F.normalize(embs, p=2, dim=-1)

                # flatten speaker axis if present
                embs, labels_flat = self._flatten_embs_labels(embs, labels)

                all_embeddings.append(embs.cpu())
                all_labels.append(labels_flat.cpu().numpy())

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()  # [N,D]
        all_labels = np.concatenate(all_labels, axis=0)             # [N]
        return all_embeddings, all_labels
    

    def compute_intra_inter(self, all_embs, all_labels):
        speakers = np.unique(all_labels)
        centroids = {}

        for spk in speakers:
            centroids[spk] = np.mean(all_embs[all_labels == spk], axis=0) #[D]
        
        intra_list = []
        for i in range(all_embs.shape[0]):
            spk = all_labels[i]
            e = all_embs[i]

            intra_list.append(1-np.dot(e, centroids[spk]))

        intra = np.mean(intra_list)



        #Interspeaker cosine distance
        inter_list = []
        spk_list = list(speakers)
        for i in range(len(spk_list)):
            for j in range(i+1, len(spk_list)):
                c1 = centroids[spk_list[i]]
                c2 = centroids[spk_list[j]]

                inter_list.append(1 - np.dot(c1, c2))
            
            inter = np.mean(inter_list)

        separation = inter - intra

        return intra, inter, separation

    
    def compute_clustering(self, all_embs, all_labels):
        K = len(np.unique(all_labels))

        kmeans = KMeans(n_clusters=K, random_state=0)

        cluster_ids = kmeans.fit_predict(all_embs)

        nmi = normalized_mutual_info_score(all_labels, cluster_ids)
        ari = adjusted_rand_score(all_labels, cluster_ids)

        try:
            sil = silhouette_score(all_embs, all_labels)
        except:
            sil = -1.0

        return nmi, ari, sil

    
    # def plot_tsne(self, all_embs, all_labels, num_speakers = 4):

    #     speakers = np.unique(all_labels)
    #     chosen = np.random.choice(speakers, min(num_speakers, len(speakers)), replace=False)

    #     mask = np.isin(all_labels, chosen)
    #     X = all_embs[mask]
    #     Y = all_labels[mask]

    #     tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    #     X2 = tsne.fit_transform(X)

    #     fig = plt.figure(figsize=(10, 7))
    #     for spk in chosen:
    #         pts = X2[Y == spk]
    #         plt.scatter(pts[:, 0], pts[:, 1], label=f"Spk {spk}", s=12)

    #     plt.legend()
    #     plt.title("t-SNE of Speaker Embeddings")
    #     plt.tight_layout()

    def compute_from_tensors(self, embs, labels, num_clusters=None):
        # embs: [N,D] or [N,S,D]  (tensor)
        # labels: [N] or [N,S]    (tensor)

        if isinstance(embs, torch.Tensor):
            if embs.dim() == 3:
                N, S, D = embs.shape
                embs = embs.reshape(N * S, D)
            else:
                embs = embs.reshape(-1, embs.shape[-1])
            embs = embs.cpu().numpy()
        else:
            embs = np.asarray(embs)

        if isinstance(labels, torch.Tensor):
            labels = labels.reshape(-1).cpu().numpy()
        else:
            labels = np.asarray(labels).reshape(-1)

        N = len(labels)

        # normalize embeddings
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)

        # 1. separation
        same, diff = [], []
        for i in range(N):
            for j in range(i + 1, N):
                cos = float(np.dot(embs[i], embs[j]))
                if labels[i] == labels[j]:
                    same.append(cos)
                else:
                    diff.append(cos)
        separation = np.mean(same) - np.mean(diff)

        # 2. clustering
        if num_clusters is None:
            num_clusters = len(np.unique(labels))

        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        pred = kmeans.fit_predict(embs)

        cluster_acc = self._cluster_accuracy(pred, labels)
        nmi = normalized_mutual_info_score(labels, pred)
        ari = adjusted_rand_score(labels, pred)

        try:
            silhouette = silhouette_score(embs, labels)
        except Exception:
            silhouette = float("nan")

        return {
            "separation": separation,
            "cluster_acc": cluster_acc,
            "nmi": nmi,
            "ari": ari,
            "silhouette": silhouette,
        }

    # -------------------------------------------------------
    # Helper: majority-vote cluster accuracy
    # -------------------------------------------------------
    def _cluster_accuracy(self,pred_labels, true_labels):
        """Compute clustering accuracy by assigning each cluster
        to the most common true label."""
        from collections import Counter

        pred = np.array(pred_labels)
        true = np.array(true_labels)
        clusters = np.unique(pred)

        total = 0
        for c in clusters:
            mask = (pred == c)
            true_subset = true[mask]
            if len(true_subset) == 0:
                continue
            most_common = Counter(true_subset).most_common(1)[0][1]
            total += most_common
        return total / len(true)

    
    def validate(self, model, val_loader, log_tsne=False):
        all_embs, all_labels = self.extract_embeddings(model, val_loader)

        intra, inter, sep = self.compute_intra_inter(all_embs, all_labels)
        nmi, ari, sil = self.compute_clustering(all_embs, all_labels)
        # fig=None
        # if log_tsne:
        #     fig = self.plot_tsne(all_embs, all_labels)
        
        metrics = {
            "intra": intra,
            "inter": inter,
            "separation": sep,
            "NMI": nmi,
            "ARI": ari,
            "Silhouette": sil,
            # 'tsne_figure': fig
        }


        return metrics



