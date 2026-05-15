import torch
import numpy as np
import onnxruntime as ort
import torch.nn.functional as F
from model import SpeakerEncoderDualWrapper
from train import strip_dual_model_weights
from model_onnx_wrapper import FeatureToEmbeddingONNXWrapper

def load_model():
    ckpt_path =  "/home/sidcs/model_ckpts/ECAPA_UNMIX_3072_teacher_ECAPA/best-epoch=87-val_separation=0.000.ckpt"
    model = SpeakerEncoderDualWrapper(emb_dim=256)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = strip_dual_model_weights(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Student CKPT] loaded. missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return FeatureToEmbeddingONNXWrapper(model).eval()


def main():
    torch_model = load_model()
    x = torch.randn(1, 80, 301, dtype=torch.float32)
    with torch.no_grad():
        torch_out = torch_model(x)
    
    sess = ort.InferenceSession("ecapa_feat_fp32.onnx", providers=["CPUExecutionProvider"])

    onnx_out = sess.run(['embeddings'], {'features': x.numpy().astype(np.float32)})[0]
    onnx_out = torch.from_numpy(onnx_out)

    torch_flat = torch_out.reshape(1, -1)
    onnx_flat = onnx_out.reshape(1, -1)

    cos_sim = F.cosine_similarity(torch_flat, onnx_flat, dim=-1)

    print("Torch output shape:", torch_out.shape)
    print("ONNX output shape:", onnx_out.shape)
    print("Cosine similarity:", cos_sim.item())

    print("Max absolute difference:", torch.abs(torch_flat - onnx_flat).max().item())


if __name__ == "__main__":
    main()