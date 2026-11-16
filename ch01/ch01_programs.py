"""CS264 การประมวลผลภาพเชิงดิจิทัล: โปรแกรมของบทที่ 1
รันในโฟลเดอร์ว่าง ตามลำดับจากบนลงล่าง (โปรแกรมหลัง ๆ ใช้ฟังก์ชันและไฟล์จากโปรแกรมก่อนหน้า)
"""

# ======================================================================
# โปรแกรม 1.1: การชักตัวอย่างและผลของความละเอียดเชิงพื้นที่
# ======================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

# เตรียมภาพทดสอบ (ทำครั้งเดียว) จาก scikit-image
from skimage import data
cv2.imwrite("cameraman.png", data.camera())

f = cv2.imread("cameraman.png", cv2.IMREAD_GRAYSCALE)
if f is None:
    raise FileNotFoundError("ไม่พบไฟล์ cameraman.png")
M, N = f.shape                      # M แถว, N คอลัมน์
print(f"ขนาดภาพ {M} x {N}, ชนิดข้อมูล {f.dtype}")

factors = [1, 2, 4, 8, 16]
fig, axes = plt.subplots(1, len(factors), figsize=(15, 3.2))
for ax, d in zip(axes, factors):
    # เก็บทุก ๆ d พิกเซลทั้งแนวแถวและแนวคอลัมน์
    g = f[::d, ::d]
    # ขยายกลับเป็น N x M เพื่อแสดงผลขนาดเท่ากัน (dsize เรียงเป็น กว้าง, สูง)
    h = cv2.resize(g, (N, M), interpolation=cv2.INTER_NEAREST)
    ax.imshow(h, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{g.shape[0]} x {g.shape[1]}")
    ax.axis("off")
plt.tight_layout()
plt.show()


# ======================================================================
# โปรแกรม 1.2: ปรากฏการณ์ aliasing จากการชักตัวอย่างไม่เพียงพอ
# ======================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

# สร้างภาพวงแหวนที่ความถี่เพิ่มขึ้นตามระยะจากจุดศูนย์กลาง
N = 512
y, x = np.mgrid[0:N, 0:N] - N / 2
r2 = x**2 + y**2
f = np.round(127.5 * (1 + np.cos(np.pi * r2 / (2 * N))))
f = f.astype(np.uint8)

d = 4
g1 = f[::d, ::d]                         # ชักตัวอย่างโดยตรง
# เฉลี่ยก่อนชักตัวอย่าง
g2 = cv2.resize(f, (N // d, N // d),
                interpolation=cv2.INTER_AREA)

titles = ["original 512x512", "direct sampling 128x128",
          "area averaging 128x128"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, t in zip(axes, [f, g1, g2], titles):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255,
              interpolation="nearest")
    ax.set_title(t)
    ax.axis("off")
plt.tight_layout()
plt.show()


# ======================================================================
# โปรแกรม 1.3: การควอนไทซ์และผลของจำนวนระดับความเข้ม
# ======================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

def quantize(f, k):
    """ลดภาพ uint8 ให้เหลือ L = 2**k ระดับ ตามสมการ (1.8)"""
    s = 8 - k
    # เลื่อนบิตขวาก่อน จึงไม่เกิดค่าล้นของ uint8
    return (f >> s) << s

f = cv2.imread("cameraman.png", cv2.IMREAD_GRAYSCALE)
if f is None:
    raise FileNotFoundError("ไม่พบไฟล์ cameraman.png")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, k in zip(axes, [8, 4, 2, 1]):
    g = quantize(f, k)
    e = f.astype(np.int16) - g      # ความคลาดเคลื่อนจากการควอนไทซ์
    print(f"k = {k}: ระดับที่ใช้จริง {np.unique(g).size}, "
          f"ความคลาดเคลื่อนสูงสุด {e.max()}")
    ax.imshow(g, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"k = {k}, L = {2**k}")
    ax.axis("off")
plt.tight_layout()
plt.show()


# ======================================================================
# โปรแกรม 1.4: การเข้าถึงพิกเซลและการจัดเก็บแบบเรียงตามแถว
# ======================================================================
import numpy as np

# ภาพระดับเทา 4 x 4 จากตัวอย่างที่ 1.6
f = np.array([[ 12,  15,  18,  21],
              [ 40,  45,  50,  55],
              [ 90,  95, 100, 105],
              [200, 210, 220, 230]], dtype=np.uint8)
M, N = f.shape                       # M แถว, N คอลัมน์
x, y = 2, 1
# NumPy เข้าถึงด้วย [แถว, คอลัมน์] = [y, x]
print(f"f({x},{y}) = {f[y, x]}")

flat = f.ravel()             # เรียงเป็นหนึ่งมิติแบบเรียงตามแถว
i = y * N + x                        # สมการ (1.11)
print(f"ดัชนี i = {i}, ค่า = {flat[i]}")
print(f"ย้อนกลับ: y = {i // N}, x = {i % N}")   # สมการ (1.12)
print("strides ของภาพระดับเทา:", f.strides)

# ภาพสีขนาด 640 x 480 ที่ OpenCV เก็บเป็นลำดับ B, G, R
img = np.zeros((480, 640, 3), dtype=np.uint8)
img[50, 100] = (10, 20, 30)          # (B, G, R) ที่ x = 100, y = 50
C = img.shape[2]
i = (50 * 640 + 100) * C + 2         # สมการ (1.13) ช่องสีแดง c = 2
print(f"ดัชนีช่องสีแดง i = {i}, ค่า = {img.ravel()[i]}")
print("strides ของภาพสี:", img.strides)


# ======================================================================
# โปรแกรม 1.5: เพื่อนบ้าน การเชื่อมต่อ และการวัดระยะทาง
# ======================================================================
import cv2
import numpy as np

def neighbors(x, y, M, N, kind="8"):
    """คืนพิกัดเพื่อนบ้านของพิกเซล (x, y) ที่อยู่ภายในภาพขนาด M x N"""
    n4 = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    nd = [(x + 1, y + 1), (x + 1, y - 1),
          (x - 1, y + 1), (x - 1, y - 1)]
    cand = {"4": n4, "D": nd, "8": n4 + nd}[kind]
    return [(u, v) for u, v in cand if 0 <= u < N and 0 <= v < M]

print("N4 ของ (0,2):", neighbors(0, 2, 4, 4, "4"))
print("N8 ของ (0,2):", neighbors(0, 2, 4, 4, "8"))

# ภาพสองระดับจากตัวอย่างที่ 1.9
b = np.array([[0, 0, 0, 0, 0],
              [0, 1, 1, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 1, 0],
              [0, 0, 0, 1, 0]], dtype=np.uint8)
for conn in (4, 8):
    n, labels = cv2.connectedComponents(b, connectivity=conn)
    print(f"การเชื่อมต่อแบบ {conn}: {n - 1} ชิ้นส่วน")   # n รวมพื้นหลังด้วย

# ระยะทางระหว่าง p = (1, 1) และ q = (4, 3)
(x, y), (s, t) = (1, 1), (4, 3)
De = np.hypot(x - s, y - t)
D4 = abs(x - s) + abs(y - t)
D8 = max(abs(x - s), abs(y - t))
print(f"De = {De:.2f}, D4 = {D4}, D8 = {D8}")


# ======================================================================
# โปรแกรม 1.6: การแปลงประเภทภาพและการคำนวณพื้นที่จัดเก็บ
# ======================================================================
import os
import cv2
import numpy as np

# เตรียมภาพทดสอบจาก scikit-image (ถ้ายังไม่มีไฟล์)
from skimage import data
if not os.path.exists("astronaut.png"):
    bgr = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    cv2.imwrite("astronaut.png", bgr)

# ภาพสี ลำดับ B, G, R
color = cv2.imread("astronaut.png", cv2.IMREAD_COLOR)
if color is None:
    raise FileNotFoundError("ไม่พบไฟล์ astronaut.png")
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)  # สมการ (1.21)
T = 128
binary = np.where(gray >= T, 1, 0).astype(np.uint8)  # (1.22)

cv2.setRNGSeed(0)                    # ให้ผล k-means ซ้ำได้
# ภาพแบบดัชนี 8 สี: หาตารางสีด้วย k-means (รายละเอียดในบทที่ 8)
K = 8
pixels = color.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20, 1.0)
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 3,
                                cv2.KMEANS_PP_CENTERS)
# เมทริกซ์ดัชนี และตารางสี K x 3
index = labels.reshape(gray.shape).astype(np.uint8)
colormap = np.round(centers).astype(np.uint8)
indexed_view = colormap[index]       # แปลงกลับเป็นภาพสีเพื่อแสดง

M, N = gray.shape
print(f"ขนาดภาพ {M} x {N}")
print(f"ภาพสี      : {color.nbytes:>9,} ไบต์")
print(f"ภาพระดับเทา: {gray.nbytes:>9,} ไบต์")
print(f"ภาพสองระดับ: {binary.nbytes:>9,} ไบต์ในหน่วยความจำ\n"
      f"             (ถ้าอัดเป็น 1 บิต/พิกเซล = {M * N // 8:,} ไบต์)")
print(f"ภาพแบบดัชนี: {index.nbytes + colormap.nbytes:>9,} ไบต์ "
      f"(ดัชนี {index.nbytes:,} + ตารางสี {colormap.nbytes})")
print("ค่าพิกเซล (x=200, y=100): สี(BGR) =", color[100, 200])
print("  เทา =", gray[100, 200], "สองระดับ =", binary[100, 200],
      "ดัชนี =", index[100, 200])

cv2.imwrite("astronaut_gray.png", gray)
size = os.path.getsize("astronaut_gray.png")
print(f"ไฟล์ PNG ภาพระดับเทา: {size:,} ไบต์")


# ======================================================================
# โปรแกรม 1.7: ความสว่างและความเปรียบต่างของภาพ
# ======================================================================
import cv2
import numpy as np

def brightness_contrast(f):
    """คืนค่าความสว่าง B, ความเปรียบต่าง C และ sigma
    ตามสมการ (1.26)-(1.28)"""
    f = f.astype(np.float64)
    B = f.mean()
    C = ((f - B) ** 2).mean()
    return B, C, np.sqrt(C)

# ภาพในตัวอย่างที่ 1.12
fA = np.array([[ 10,  20, 200, 220]] * 4, dtype=np.uint8).T
fB = np.array([[100, 110, 115, 125]] * 4, dtype=np.uint8).T
for name, f in [("A", fA), ("B", fB)]:
    B, C, s = brightness_contrast(f)
    print(f"ภาพ {name}: B = {B:.2f}, C = {C:.2f}, "
          f"sigma = {s:.2f}")

# ตรวจสอบสมการ (1.29)-(1.30) ด้วยการแปลงเชิงเส้น g = a f + b
f = cv2.imread("cameraman.png", cv2.IMREAD_GRAYSCALE)
if f is None:
    raise FileNotFoundError("ไม่พบไฟล์ cameraman.png")
a, b = 0.5, 40
B, C, s = brightness_contrast(f)
g = a * f.astype(np.float64) + b                     # ยังไม่ตัดค่า
Bg, Cg, sg = brightness_contrast(g)
print(f"f: B = {B:.2f}, sigma = {s:.2f}")
print(f"g: B = {Bg:.2f} (ทฤษฎี {a * B + b:.2f}), "
      f"sigma = {sg:.2f} (ทฤษฎี {abs(a) * s:.2f})")

# เมื่อแปลงแล้วค่าเกิน 0-255 ต้องตัดค่า ความสัมพันธ์จะไม่เป็นเชิงเส้นอีก
a, b = 2.0, 0
g8 = np.clip(a * f.astype(np.float64) + b, 0, 255)
g8 = g8.astype(np.uint8)
Bg, Cg, sg = brightness_contrast(g8)
print(f"a = 2 หลังตัดค่า: sigma = {sg:.2f} "
      f"(ทฤษฎีถ้าไม่ตัดค่า {abs(a) * s:.2f})")


# ======================================================================
# โปรแกรม 1.8: ตรวจสอบสภาพแวดล้อมการพัฒนา
# ======================================================================
# ติดตั้งครั้งแรก (บนเครื่องของผู้อ่าน):
#   pip install opencv-python numpy matplotlib scikit-image
import sys
import cv2
import numpy as np
import matplotlib
import skimage

print("Python      :", sys.version.split()[0])
print("OpenCV      :", cv2.__version__)
print("NumPy       :", np.__version__)
print("Matplotlib  :", matplotlib.__version__)
print("scikit-image:", skimage.__version__)


# ======================================================================
# โปรแกรม 1.9: การอ่าน แสดง ตัด แปลง และบันทึกภาพ
# ======================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("astronaut.png")   # อ่านเป็นภาพสี ลำดับ B, G, R
if img is None:
    raise FileNotFoundError("ไม่พบไฟล์ astronaut.png")
M, N, C = img.shape
print(f"ขนาด {M} x {N}, ช่องสี {C}, ชนิดข้อมูล {img.dtype}")
print("ค่าพิกเซล f(200,100) ในลำดับ B, G, R:", img[100, 200])

# ตัดบริเวณ y = 30..229, x = 130..329
roi = img[30:230, 130:330].copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ค่าที่มากกว่า 127 เป็น 255
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img)
axes[0].set_title("wrong: BGR shown as RGB")
axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[1].set_title("correct: converted to RGB")
axes[2].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
axes[2].set_title("ROI 200 x 200")
axes[3].imshow(binary, cmap="gray")
axes[3].set_title("binary, T = 127")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# cv2.imwrite รับภาพลำดับ B, G, R
cv2.imwrite("astronaut_roi.png", roi)


# ======================================================================
# โปรแกรม 1.10: ข้อผิดพลาดที่พบบ่อย: ค่าล้นของ uint8 และวิวกับสำเนา
# ======================================================================
import cv2
import numpy as np

a = np.array([[250]], dtype=np.uint8)
b = np.array([[10]], dtype=np.uint8)
print("NumPy  250 + 10 =", (a + b)[0, 0])        # วนค่ากลับ
print("OpenCV 250 + 10 =", cv2.add(a, b)[0, 0])  # อิ่มตัวที่ 255
print("NumPy  10 - 250 =", (b - a)[0, 0])
print("OpenCV 10 - 250 =", cv2.subtract(b, a)[0, 0])
print("แปลงก่อนคำนวณ  =", int(a[0, 0]) + int(b[0, 0]))

f = np.zeros((4, 4), dtype=np.uint8)
view = f[0:2, 0:2]            # วิว: ใช้หน่วยความจำร่วมกับ f
view[:] = 99
print("หลังแก้วิว   f[0,0] =", f[0, 0])             # f เปลี่ยนตาม
f[:] = 0
copy = f[0:2, 0:2].copy()     # สำเนา: แยกหน่วยความจำ
copy[:] = 99
print("หลังแก้สำเนา f[0,0] =", f[0, 0])             # f ไม่เปลี่ยน
