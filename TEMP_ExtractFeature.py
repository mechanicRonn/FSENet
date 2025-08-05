import math

import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from models.common import DetectMultiBackend

# === 1. Load mô hình đã huấn luyện ===
weights = "runs/train/exp/weights/best.pt"  # đổi nếu cần
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(weights, device=device)
model.eval()

# === 2. Load ảnh ===
img_path = "D:/Lam/Lam Lab/PycharmProjects/Yolov5/2.jpg"
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    print("❌ Không load được ảnh. Kiểm tra đường dẫn!")
    exit()

print("✅ Ảnh đã load thành công.")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))
img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)

# === 3. Hook layer để lấy feature ===
feature_maps = {}


def hook_fn(module, input, output):
    print("🎯 Hook activated. Feature shape:", output.shape)
    feature_maps["target"] = output.detach().cpu()


# Chọn một layer phù hợp (Conv layer nông)
target_layer_index = 17  # 6 là index của layer Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1)) trong mô hình yolov5s
handle = model.model.model[target_layer_index].register_forward_hook(hook_fn)

# === 4. Forward để kích hoạt hook ===
with torch.no_grad():
    _ = model(img_tensor)

handle.remove()

# === 5. Hiển thị feature map ===
if "target" in feature_maps:
    fmap = feature_maps["target"][0]  # shape: [C, H, W]
    num_channels = min(16, fmap.shape[0])  # Số kênh cần hiển thị
    cols = 4
    rows = math.ceil(num_channels / cols)

    plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(num_channels):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(fmap[i], cmap="viridis")  # cmap có thể là: 'viridis', 'plasma', 'inferno', 'magma', v.v.
        plt.title(f"Channel {i}")
        plt.axis("off")

    plt.tight_layout()
    save_path = "D:/Lam/Lam Lab/PycharmProjects/Yolov5/TEMP_FeatureMap/Test/feature_map_output_layer17_temp.png"
    plt.savefig(save_path)
    print(f"✅ Feature map đã được lưu vào {save_path}")

else:
    print("❌ Không nhận được feature map. Kiểm tra hook hoặc layer index.")
