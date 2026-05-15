import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SpeakerEncoderDualWrapper 
from train import strip_dual_model_weights




class FeatureToEmbeddingONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feats):
        '''
        feats : [B, 80, T_frames]
        '''
        return self.model.forward_from_features(feats)
    


def main():
    ckpt_path =  "/home/sidcs/model_ckpts/ECAPA_UNMIX_3072_teacher_ECAPA/best-epoch=87-val_separation=0.000.ckpt"
    model = SpeakerEncoderDualWrapper(emb_dim=256)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = strip_dual_model_weights(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Student CKPT] loaded. missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    export_model = FeatureToEmbeddingONNXWrapper(model)
    export_model.eval()

    #3s audio at 16Khz with hop 160 gives 301 feamews
    dummy_input = torch.randn(1, 80, 301, dtype=torch.float32)

    with torch.no_grad():
        out = export_model(dummy_input)
        print(f"Output shape: {out.shape}")

    torch.onnx.export(export_model, dummy_input, "ecapa_feat_fp32.onnx", 
                      input_names=["features"], output_names=["embeddings"], opset_version=17, 
                      do_constant_folding=True, dynamic_axes={"features":{0: "batch"}, "embeddings":{0: "batch"}})
    print("ONNX export successful.")

if __name__ == "__main__":
    main()
    
    