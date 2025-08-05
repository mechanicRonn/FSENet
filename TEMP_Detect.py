import os
from tkinter import Tk, filedialog

# BƯỚC 1: Mở hộp thoại chọn ảnh
Tk().withdraw()
img_path = filedialog.askopenfilename(
    title="Chọn ảnh để detect",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

# BƯỚC 2: Kiểm tra và gọi detect.py
if not img_path:
    print("❌ Bạn chưa chọn ảnh nào.")
else:
    print(f"✅ Đang detect ảnh: {img_path}")
    os.system(
        f'python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source "{img_path}"'
    )

