from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="wavlm_feat_joint_trained_fp32.onnx",
    model_output="wavlm_feat_int8_dynamic_linear_only.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
)

print("Saved: wavlm_feat_int8_dynamic_linear_only.onnx")