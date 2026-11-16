"""CS264 การประมวลผลภาพเชิงดิจิทัล: โปรแกรมของบทที่ 2
รันในโฟลเดอร์ว่าง ตามลำดับจากบนลงล่าง (โปรแกรมหลัง ๆ ใช้ฟังก์ชันและไฟล์จากโปรแกรมก่อนหน้า)
"""

# ======================================================================
# โปรแกรม 2.1: การแปลงความเข้มแสงเชิงจุดด้วยตารางค้นหา
# ======================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# ภาพระดับเทา 8 บิต
# ภาพทดสอบของ scikit-image; ใช้ภาพของผู้อ่านได้ด้วย
# f = cv2.imread("ชื่อไฟล์", cv2.IMREAD_GRAYSCALE)
f = data.camera()
L = 256
r = np.arange(L, dtype=np.float64)        # r = 0, 1, ..., 255

lut_neg = (L - 1) - r                     # สมการที่ (2.3)
c = (L - 1) / np.log(1 + (L - 1))
lut_log = c * np.log(1 + r)               # สมการที่ (2.4)
gamma = 0.5
lut_gam = (L - 1) * (r / (L - 1)) ** gamma   # สมการที่ (2.5)

def to_u8(t):             # ปัดเศษและตัดค่าให้อยู่ใน [0, 255]
    return np.clip(np.round(t), 0, 255).astype(np.uint8)

g_neg = cv2.LUT(f, to_u8(lut_neg))
g_log = cv2.LUT(f, to_u8(lut_log))
g_gam = cv2.LUT(f, to_u8(lut_gam))

show = [("f", f), ("negative", g_neg),
        ("log", g_log), ("gamma 0.5", g_gam)]
fig, axes = plt.subplots(1, len(show), figsize=(14, 3.6))
for ax, (title, img) in zip(axes, show):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()


# ======================================================================
# โปรแกรม 2.2: การปรับฮิสโทแกรมให้เท่ากันและ CLAHE
# ======================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def hist_equalize(f, L=256):
    n = np.bincount(f.ravel(), minlength=L)    # n_k
    cdf = np.cumsum(n) / f.size                # ขั้นที่ 1-2
    # ขั้นที่ 3: สมการที่ (2.10)
    lut = np.floor((L - 1) * cdf + 0.5)
    return lut.astype(np.uint8)[f]             # ขั้นที่ 4: แทนค่า

# ตรวจกับตัวอย่างที่ 2.4 (L = 10)
A = np.array([[2, 3, 3, 2], [4, 2, 4, 3],
              [3, 2, 3, 5], [2, 4, 2, 4]], dtype=np.uint8)
# ต้องได้ภาพผลลัพธ์ตามตัวอย่างที่ 2.4
print(hist_equalize(A, L=10))

# ภาพทดสอบของ scikit-image; ใช้ภาพของผู้อ่านได้ด้วย
# f = cv2.imread("ชื่อไฟล์", cv2.IMREAD_GRAYSCALE)
f = data.camera()
g_he = cv2.equalizeHist(f)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
g_clahe = clahe.apply(f)

show = [("f", f), ("equalizeHist", g_he), ("CLAHE", g_clahe)]
fig, axes = plt.subplots(1, len(show), figsize=(10, 3.6))
for ax, (title, img) in zip(axes, show):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()


# ======================================================================
# โปรแกรม 2.3: การเพิ่มความคมชัดด้วยลาปลาเซียน unsharp masking และ DoG
# ======================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# ภาพทดสอบของ scikit-image; ใช้ภาพของผู้อ่านได้ด้วย
# f = cv2.imread("ชื่อไฟล์", cv2.IMREAD_GRAYSCALE)
f = data.camera()
f = f.astype(np.float32)
to_u8 = lambda t: np.clip(t, 0, 255).astype(np.uint8)

# (1) ลาปลาเซียน: สมการที่ (2.15) และ (2.17) ด้วย c = -1
w = np.array([[0, 1, 0],
              [1, -4, 1],
              [0, 1, 0]], dtype=np.float32)   # หน้ากากรูปที่ 2.7(ก)
lap = cv2.filter2D(f, -1, w, borderType=cv2.BORDER_REPLICATE)
g_lap = to_u8(f - lap)

# (2) unsharp masking / high-boost: สมการที่ (2.18)
blur = cv2.GaussianBlur(f, (0, 0), sigmaX=3)
k = 1.0                          # k > 1 คือ high-boost
g_usm = to_u8(f + k * (f - blur))

# (3) DoG: สมการที่ (2.21) ด้วย sigma2/sigma1 = 1.6
dog = (cv2.GaussianBlur(f, (0, 0), 1.0)
       - cv2.GaussianBlur(f, (0, 0), 1.6))
g_dog = to_u8(f + 2.0 * dog)

show = [("f", f), ("Laplacian", g_lap),
        ("unsharp", g_usm), ("DoG", g_dog)]
fig, axes = plt.subplots(1, len(show), figsize=(14, 3.6))
for ax, (title, img) in zip(axes, show):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()


# ======================================================================
# โปรแกรม 2.4: การกรองแบบโฮโมมอร์ฟิกด้วยการกรองเชิงพื้นที่
# ======================================================================
import cv2
import numpy as np

def homomorphic(f, sigma=30, gL=0.5, gH=1.5):
    """การกรองแบบโฮโมมอร์ฟิกด้วยการกรองเชิงพื้นที่ ตามสมการที่ (2.23)"""
    z = np.log1p(f.astype(np.float64))         # ln(1 + f) กัน ln 0
    low = cv2.GaussianBlur(z, (0, 0), sigma)   # ความถี่ต่ำ ≈ ln i
    z2 = gL * low + gH * (z - low)             # ลด i และเพิ่ม r
    g = np.expm1(z2)
    g = (g - g.min()) / (g.max() - g.min())    # ปรับช่วงเป็น 0–255
    return np.round(255 * g).astype(np.uint8)

# ภาพจำลอง f = i·r: r เป็นตารางหมากรุก 0.3/0.9 ช่องละ 32 พิกเซล
# i ลดจาก 1.0 ทางซ้ายเหลือ 0.15 ทางขวา
y, x = np.mgrid[0:256, 0:512]
r = np.where(((x // 32) + (y // 32)) % 2 == 0, 0.9, 0.3)
i = np.linspace(1.0, 0.15, 512)[None, :]
f = np.round(255 * i * r).astype(np.uint8)
g = homomorphic(f)

for name, img in [("ก่อนกรอง", f), ("หลังกรอง", g)]:
    a = img.astype(float)
    L, R = a[:, :64], a[:, -64:]
    print(f"{name}: ค่าเฉลี่ยซ้าย/ขวา = {L.mean() / R.mean():4.2f}, "
          f"ความเปรียบต่างของช่องฝั่งขวา = {R.max() - R.min():5.1f}")
