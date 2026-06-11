import glob
import torch
import numpy as np
import soundfile as sf
import torchaudio
import torch.nn.functional as F

from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)

from model import SpeakerEncoderDualWrapper


class FeatureCalibrationReader(CalibrationDataReader):
    def __init__(self, feature_list, input_name="features"):
        self.input_name = input_name
        self.data = []

        for feat in feature_list:
            #feat should be [80, T_frames]
            feat_data = feat.astype(np.float32)
            feat = feat[None, :, :] # add batch dim
            self.data.append({self.input_name: feat})
        
        self.iterator = iter(self.data)


    def get_next(self):
        return next(self.iterator, None)
    

def load_audio(path, target_sr=16000, seconds=3):
    wav, sr = sf.read(path)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    wav = torch.tensor(wav, dtype=torch.float32)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    
    num_samples = target_sr * seconds

    if wav.numel() < num_samples:
        wav = F.pad(wav, (0, num_samples - wav.numel()), mode='constant')
    else:
        wav = wav[:num_samples]

    return wav.unsqueeze(0)  # [1, T]


def build_calibration_audios(wav_paths, max_files=200):
    model = SpeakerEncoderDualWrapper(emb_dim=256)
    model.eval()

    audios = []

    for path in wav_paths[:max_files]:
        audio = load_audio(path, target_sr=16000, seconds=3)
        audios.append(audio)

    return audios

def main():
    wav_paths = sorted(glob.glob("/home/sidcs/datasets/LibriMix/scripts/calibration_2spk/mix_both/*.wav"))
    print("Calibration wavs:", len(wav_paths))
    calib_audios = build_calibration_audios(wav_paths, max_files=500)

    calib_reader = FeatureCalibrationReader(calib_audios, input_name="audios")

    quantize_static(
        model_input="wavlm_feat_joint_trained_fp32.onnx",
        model_output="wavlm_feat_int8_static_conv_only.onnx",
        calibration_data_reader=calib_reader,
        quant_format = QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv"],)
    
    print("Saved: wavlm_feat_int8_static_conv_only.onnx")

if __name__ == "__main__":
    main()
    