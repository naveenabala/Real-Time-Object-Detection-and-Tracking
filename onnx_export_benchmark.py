%%writefile onnx_export_benchmark.py
import time
import numpy as np
from ultralytics import YOLO

import config


def benchmark_pytorch(model, dummy_input, n_runs=30):
    for _ in range(5):
        model.predict(dummy_input, verbose=False)
    t0 = time.time()
    for _ in range(n_runs):
        model.predict(dummy_input, verbose=False)
    elapsed = time.time() - t0
    return elapsed / n_runs


def benchmark_onnx(onnx_path, dummy_input, n_runs=30):
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    img = dummy_input.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    for _ in range(5):
        session.run(None, {input_name: img})
    t0 = time.time()
    for _ in range(n_runs):
        session.run(None, {input_name: img})
    elapsed = time.time() - t0
    return elapsed / n_runs


def main():
    model = YOLO(config.MODEL_NAME)

    print("Exporting to ONNX ...")
    onnx_path = model.export(format="onnx", imgsz=640)
    print(f"Exported: {onnx_path}")

    dummy_input = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)

    pt_latency = benchmark_pytorch(model, dummy_input)
    onnx_latency = benchmark_onnx(onnx_path, dummy_input)

    print("\n--- Latency benchmark (per-frame, 640x640 dummy input) ---")
    print(f"PyTorch (.pt):  {pt_latency * 1000:.1f} ms/frame  ({1/pt_latency:.1f} FPS)")
    print(f"ONNX Runtime:   {onnx_latency * 1000:.1f} ms/frame  ({1/onnx_latency:.1f} FPS)")


if __name__ == "__main__":
    main()
