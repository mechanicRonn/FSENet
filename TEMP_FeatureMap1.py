import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from models.common import DetectMultiBackend

# === 1. Load mô hình đã huấn luyện ===
weights = "runs/train/exp/weights/best.pt"  # chỉnh lại đường dẫn nếu cần
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(weights, device=device)
model.eval()

# === 2. Load ảnh ===
img_path = "D:/Lam/Lam Lab/PycharmProjects/Yolov5/1.jpg"
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

# Lựa chọn channel bạn muốn hiển thị
channel_to_display = 5  # <-- Chọn số channel bạn muốn ở đây

def hook_fn(module, input, output):
    print("🎯 Hook activated. Feature shape:", output.shape)
    feature_maps["target"] = output.detach().cpu()

# Chọn một layer phù hợp để hook (ví dụ layer 17)
target_layer_index = 17
handle = model.model.model[target_layer_index].register_forward_hook(hook_fn)

# === 4. Forward để kích hoạt hook ===
with torch.no_grad():
    _ = model(img_tensor)

handle.remove()

# === 5. Hiển thị feature map ở channel đã chọn ===
if "target" in feature_maps:
    fmap = feature_maps["target"][0]  # [C, H, W]
    if channel_to_display >= fmap.shape[0]:
        print(f"❌ Channel {channel_to_display} vượt quá số lượng channel {fmap.shape[0]}.")
    else:
        plt.figure(figsize=(6, 6))
        plt.imshow(fmap[channel_to_display], cmap="viridis")
        #plt.title(f"Feature map at Channel {channel_to_display}")
        plt.title(f"Yolov5")
        plt.axis("off")

        save_path = f"D:/Lam/Lam Lab/PycharmProjects/Yolov5/TEMP_FeatureMap/anh1/feature_map_output_layer{target_layer_index}_channel{channel_to_display}.png"
        plt.savefig(save_path)
        print(f"✅ Feature map của channel {channel_to_display} đã được lưu vào {save_path}")
else:
    print("❌ Không nhận được feature map. Kiểm tra hook hoặc layer index.")
