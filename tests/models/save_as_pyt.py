from hummingbird.ml import convert
import onnx

filename = "hospital_GradientBoostingClassifier_BINARY_20_estimators_MaxDepth_3_batch.onnx"
o = onnx.load(filename, "r")
t = convert(o, "torch")
t.save(filename + "as_pyt.zip")
