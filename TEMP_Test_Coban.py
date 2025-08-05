import sys
import os
import torch
from models.yolo import Model
import traceback

def print_layer_summary(model):
    print("\n📋 Model summary:")
    print(f"Total layers: {len(model.model)}")
    print("-" * 70)
    print(f"{'Idx':<4} {'Type':<30} {'Params':>10} {'Output Shape':>20}")
    print("-" * 70)
    for i, layer in enumerate(model.model):
        params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        print(f"{i:<4} {layer.__class__.__name__:<30} {params:>10}", end='')
        try:
            # Nếu lớp là Conv/C3/FSM/FEM/Get..., show output shape với dummy input
            if hasattr(layer, 'forward'):
                x = torch.randn(1, 3, 64, 64)  # dummy input
                if i == 0:
                    out = layer(x)
                else:
                    # Các lớp sau chỉ in ra type (nếu muốn test shape thật, cần mapping đúng input)
                    out = None
                if out is not None and isinstance(out, torch.Tensor):
                    print(f" {tuple(out.shape)}", end='')
                elif isinstance(out, (tuple, list)):
                    print(f" tuple(len={len(out)})", end='')
        except Exception as e:
            print(" [cannot test]", end='')
        print()
    print("-" * 70)

def test_model_full():
    try:
        print("🔍 Testing YOLOv5 model parse & forward...")
        # === 1. Parse model
        model = Model("D:/Lam/Lam Lab/PycharmProjects/Yolov5/ultralytics/yolov5-master/models/yolov5s_FSENet.yaml", ch=3, nc=80)
        print("✅ Model parsing successful!")
        print_layer_summary(model)

        # === 2. Test forward với ảnh giả
        print("\n🚦 Testing forward pass with random input...")
        dummy = torch.randn(1, 3, 640, 640)  # B=1, C=3, H=W=640
        with torch.no_grad():
            output = model(dummy)
        if isinstance(output, (tuple, list)):
            print("Output is tuple/list, length:", len(output))
            if hasattr(output[0], 'shape'):
                print("Output[0] shape:", output[0].shape)
        else:
            print("Output shape:", output.shape)
        print("✅ Forward pass successful!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        print("📍 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_full()
