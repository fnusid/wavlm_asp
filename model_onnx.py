import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SpeakerEncoderDualWrapper 
JOINT_TRAINED_CKPT = "/home/sidcs/model_ckpts/pDCCRN_2sp_dpccn_joint_training_freezewavlm_indloss/best-epoch=19-val_separation=0.000.ckpt"

def strip_dual_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if not k.startswith("model."):
            continue

        k2 = k[len("model."):]

        if k2.startswith("single_sp_model.") or k2.startswith("arcface_loss."):
            continue

        new_state[k2] = v

    return new_state


def joint_trained_model_weights(state):
    new_state = {}
    for k, v in state.items():
        if k.startswith("dual_emb_model."):
            new_state[k[len("dual_emb_model."):]] = v
    return new_state


def load_dual(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    sd = strip_dual_model_weights(sd)
    model.load_state_dict(sd, strict=True)

    joint_sd = torch.load(JOINT_TRAINED_CKPT, map_location="cpu")["state_dict"]
    joint_sd = joint_trained_model_weights(joint_sd)
    model.load_state_dict(joint_sd, strict=True)

    model.eval()
    return model




class FeatureToEmbeddingONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio):
        
        '''audio : [B, T]'''
        return self.model.forward(audio)
    


def main():
    ckpt_path = "/home/sidcs/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"
    model = SpeakerEncoderDualWrapper(emb_dim=256)
    model = load_dual(model, ckpt_path)

    export_model = FeatureToEmbeddingONNXWrapper(model)
    export_model.eval()

    #10s
    dummy_input = torch.randn(1, 10*16000)  # [B, T]
    

    with torch.no_grad():
        out = export_model(dummy_input)
        print(f"Output shape: {out.shape}")

    torch.onnx.export(export_model, dummy_input, "wavlm_feat_joint_trained_fp32.onnx", 
                      input_names=["features"], output_names=["embeddings"], opset_version=17, 
                      do_constant_folding=True, dynamic_axes={"features":{0: "batch"}, "embeddings":{0: "batch"}})
    print("ONNX export successful.")

if __name__ == "__main__":
    main()