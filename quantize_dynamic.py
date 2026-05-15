from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="ecapa_feat_fp32.onnx",
    model_output="ecapa_feat_int8_dynamic_linear_only.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
)

print("Saved: ecapa_feat_int8_dynamic_linear_only.onnx")


'''
(mtse) sidcs@csegpu1:~/codebase/wavlm_dual_embedding$ ls -lh ecapa_feat_fp32.onnx ecapa_feat_int8_dynamic_linear_only.onnx 
-rw-rw-r-- 1 sidcs sidcs 325M May 10 18:48 ecapa_feat_fp32.onnx
-rw-rw-r-- 1 sidcs sidcs 322M May 10 18:58 ecapa_feat_int8_dynamic_linear_only.onnx
'''