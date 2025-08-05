import models.common
from models.yolo import Model

print("models.common loaded from:", models.common.__file__)

# Sửa lại đường dẫn cho đúng file yaml của bạn
model = Model("models/yolov5s_FSENet.yaml", ch=3, nc=80)


print("Layer | Output Channels | Layer Type")
print("-" * 40)
c_in = [3]  # khởi tạo số kênh đầu vào
for i, m in enumerate(model.model):
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # Nếu là Conv2d, C3, FSM, FEM, ... lấy số kênh output
    if hasattr(m, "conv") and hasattr(m.conv, "out_channels"):
        out_ch = m.conv.out_channels
    elif hasattr(m, "cv1") and hasattr(m.cv1.conv, "out_channels"):
        out_ch = m.cv1.conv.out_channels
    elif hasattr(m, "cv2") and hasattr(m.cv2.conv, "out_channels"):
        out_ch = m.cv2.conv.out_channels
    else:
        out_ch = "-"
    print(f"{i:<5} | {out_ch:<15} | {type(m).__name__}")
