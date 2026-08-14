from pathlib import Path

# Model directory is anchored to the project root (…/attendance_system/models),
# independent of the current working directory.
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "anti_spoof"

# The minivision Silent-Face-Anti-Spoofing system ENSEMBLES two MiniFASNet
# variants trained at different crop scales. We use both and sum their softmax
# outputs (as the original does). ONNX exports + release assets from
# github.com/yakhyo/face-anti-spoofing.
MODELS = [
    {"file": "MiniFASNetV2.onnx",   "scale": 2.7,
     "url": "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx"},
    {"file": "MiniFASNetV1SE.onnx", "scale": 4.0,
     "url": "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx"},
]

INPUT_SIZE = (80, 80)          # (w, h) MiniFASNet input

# Output is a 3-class softmax; class index 1 = "real/live" (minivision & yakhyo
# convention). A face is accepted only if argmax == 1 AND its live probability
# clears this threshold.
# Real webcam faces score ~0.9 on the live class (validated on phone selfies:
# mean 0.93, min 0.55); a phone/print spoof collapses toward the spoof class.
# 0.5 gives real students a wide safety margin while still rejecting spoofs.
LIVE_CLASS = 1
LIVENESS_THRESHOLD = 0.5
