import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class EmbeddingMetrics:
    def __init__(self, device='cuda'):
        self.device=device

    def extract_embeddings(self, model, val_loader):
        model.eval()
        all_embeddings, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                wavs, labels = batch
                wavs = wavs.to(self.device)
                embeddings = model(wavs)  # [B, D]
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.numpy())

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        all_labels = np.concatenate(all_labels, axis=0)
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
        """
        Compute all embedding metrics using pre-collected tensors.
        This mirrors the full 'validate()' behavior but without dataloaders.

        Args:
            embs   : [N, D] float tensor (CPU)
            labels : [N] int tensor (CPU)
            num_clusters : override cluster count if needed

        Returns:
            dict with:
                - separation
                - cluster_acc
                - nmi
                - ari
                - silhouette
                - tsne_fig
        """

        # convert
        embs = embs.cpu().numpy()
        labels = labels.cpu().numpy()
        N = len(labels)

        # normalize embeddings
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)

        # ====================================================
        # 1. Separation (same vs different cosine)
        # ====================================================
        same, diff = [], []
        for i in range(N):
            for j in range(i+1, N):
                cos = float(np.dot(embs[i], embs[j]))
                if labels[i] == labels[j]:
                    same.append(cos)
                else:
                    diff.append(cos)

        separation = np.mean(same) - np.mean(diff)

        # ====================================================
        # 2. Clustering (KMeans)
        # ====================================================
        if num_clusters is None:
            num_clusters = len(np.unique(labels))

        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        pred = kmeans.fit_predict(embs)

        # majority voting cluster accuracy
        cluster_acc = self._cluster_accuracy(pred, labels)

        # NMI
        nmi = normalized_mutual_info_score(labels, pred)

        # ARI
        ari = adjusted_rand_score(labels, pred)

        # ====================================================
        # 3. Silhouette
        # ====================================================
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



