"""CS264 การประมวลผลภาพเชิงดิจิทัล: โปรแกรมของบทที่ 6
รันในโฟลเดอร์ว่าง ตามลำดับจากบนลงล่าง (โปรแกรมหลัง ๆ ใช้ฟังก์ชันและไฟล์จากโปรแกรมก่อนหน้า)
"""

# ======================================================================
# โปรแกรม 6.1: การกำหนดค่าขีดแบ่งด้วยวิธีของ Otsu (ตัวอย่างที่ 6.1)
# ======================================================================
import numpy as np
import cv2

# ภาพระดับเทาของตัวอย่างที่ 6.1
f = np.array([[30,  40, 200, 210, 35],
              [25, 190, 220, 205, 40],
              [20, 180, 230, 195, 50],
              [35,  60, 210, 185, 45],
              [30,  50,  55,  60, 40]], dtype=np.uint8)

# กำหนดค่าขีดแบ่งด้วยวิธีของ Otsu: ส่ง T = 0 แล้วให้ OpenCV คำนวณเอง
# ค่าที่กำหนดให้พิกเซลวัตถุคือ 1 ตามสมการที่ (6.2)
flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
T, b = cv2.threshold(f, 0, 1, flags)
print("T =", T)
print(b)

# ตรวจสมการที่ (6.3) ด้วยการคำนวณตรงทุกค่า T ที่แบ่งได้สองกลุ่ม
v = f.ravel().astype(float)
ts = [t for t in range(256) if 0 < (v <= t).sum() < v.size]

def sigma_b2(t):
    w0, w1 = (v <= t).mean(), (v > t).mean()
    return w0 * w1 * (v[v <= t].mean() - v[v > t].mean()) ** 2

# max คืนค่าแรกที่สูงสุด จึงได้ T ต่ำสุดของช่วง
t_best = max(ts, key=sigma_b2)
print("σ_B² สูงสุด = %.2f ที่ T = %d" % (sigma_b2(t_best), t_best))


# ======================================================================
# โปรแกรม 6.2: พื้นที่ จุดศูนย์กลางมวล การฉาย และขอบเขตของวัตถุ (ตัวอย่างที่ 6.2)
# ======================================================================
import numpy as np
import cv2

# ภาพสองระดับของตัวอย่างที่ 6.2 (แถว = y, คอลัมน์ = x เริ่มนับที่ 0)
b = np.array([[0, 0, 0, 0, 0],
              [1, 1, 0, 1, 1],
              [1, 1, 1, 1, 0],
              [0, 1, 1, 1, 1],
              [0, 1, 1, 1, 0]], dtype=np.uint8)

# พื้นที่: จำนวนพิกเซลที่มีค่า 1 (สมการที่ (6.6))
S = int(b.sum())

# จุดศูนย์กลางมวล: ค่าเฉลี่ยพิกัดของพิกเซลวัตถุ (สมการที่ (6.8))
ys, xs = np.nonzero(b)
x_bar, y_bar = float(xs.sum() / S), float(ys.sum() / S)

# การฉาย (สมการที่ (6.9)): H(x) รวมทีละคอลัมน์ (axis=0)
# และ V(y) รวมทีละแถว (axis=1)
H = b.sum(axis=0)
V = b.sum(axis=1)

# ขอบเขต: ตำแหน่งแรกและสุดท้ายที่ค่าการฉายไม่เป็นศูนย์
# (สมการที่ (6.12))
x_min, x_max = (int(i) for i in np.flatnonzero(H)[[0, -1]])
y_min, y_max = (int(i) for i in np.flatnonzero(V)[[0, -1]])

print("S =", S, " (x̄, ȳ) =", (x_bar, y_bar))
print("H(x) =", H, " V(y) =", V)
print("ซ้าย-ขวา =", (x_min, x_max), " บน-ล่าง =", (y_min, y_max))

# ตรวจกับโมเมนต์ของ OpenCV (บทที่ 7):
# m00 = S, m10/m00 = x̄, m01/m00 = ȳ
m = cv2.moments(b, binaryImage=True)
print("OpenCV:", m["m00"], m["m10"] / m["m00"],
      m["m01"] / m["m00"])


# ======================================================================
# โปรแกรม 6.3: การขยาย การกร่อน การเปิด การปิด การสกัดขอบ และการเติมรู (ตัวอย่างที่ 6.6–6.9 และ 6.11)
# ======================================================================
import numpy as np
import cv2

def from_points(points, size):
    """สร้างภาพสองระดับขนาด size x size จากรายการพิกัด (x, y)"""
    img = np.zeros((size, size), np.uint8)
    for x, y in points:
        img[y, x] = 1
    return img

# องค์ประกอบโครงสร้าง 3x3 จุดกำเนิดตรงกลาง
B_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
B_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# กำหนดนอกภาพเป็น 0 ทุกครั้ง ให้ตรงกับนิยามทางเซต
# (ค่าปริยายของ cv2.erode ถือว่านอกภาพเป็น 1
#  วัตถุที่ชิดขอบภาพจึงไม่ถูกกร่อนจากด้านนั้น)
kw = dict(borderType=cv2.BORDER_CONSTANT, borderValue=0)
dil = lambda X, B: cv2.dilate(X, B, **kw)
ero = lambda X, B: cv2.erode(X, B, **kw)

# ตัวอย่างที่ 6.6 และ 6.7(ก): เซต A จากตัวอย่างที่ 6.4
# กับ B แบบกากบาท
A = from_points([(1, 1), (3, 1), (1, 2), (2, 2), (3, 2), (2, 3)],
                5)
print("A ⊕ B =\n", dil(A, B_cross),
      "\n|A ⊕ B| =", int(dil(A, B_cross).sum()))
print("|A ⊖ B| =", int(ero(A, B_cross).sum()))   # 0 คือเซตว่าง

# ตรวจภาวะคู่กัน (สมการที่ (6.20)): B กากบาทสมมาตร จึงได้ B̂ = B
# นอกภาพของ A เป็น 0 จึงต้องให้นอกภาพของ A^c เป็น 1
lhs = 1 - ero(A, B_cross)
rhs = cv2.dilate(1 - A, B_cross, borderType=cv2.BORDER_CONSTANT,
                 borderValue=1)
print("ภาวะคู่กันเป็นจริง:", np.array_equal(lhs, rhs))

# ตัวอย่างที่ 6.8: การเปิดลบจุดรบกวนและส่วนยื่นออก
block = [(x, y) for x in range(1, 4) for y in range(2, 5)]
P = from_points([(5, 0)] + block + [(4, 3), (5, 3)], 7)
P_open = dil(ero(P, B_rect), B_rect)
print("A ∘ B =\n", P_open)

# ตัวอย่างที่ 6.9: การปิดอุดรูขนาดหนึ่งพิกเซล
ring = [(x, y) for x in range(2, 5) for y in range(2, 5)
        if (x, y) != (3, 3)]
Q = from_points(ring, 7)
Q_close = ero(dil(Q, B_rect), B_rect)
print("A • B =\n", Q_close)

# ตรวจสมบัติในสมการที่ (6.24):
# A∘B ⊆ A ⊆ A•B และการเปิดซ้ำให้ผลเท่าเดิม
print("A∘B ⊆ A:", bool(np.all(P_open <= P)),
      " A ⊆ A•B:", bool(np.all(Q <= Q_close)))
print("(A∘B)∘B = A∘B:",
      np.array_equal(dil(ero(P_open, B_rect), B_rect), P_open))

# การสกัดขอบ β(A) = A − (A ⊖ B) ของบล็อก 5x5 (สมการที่ (6.25))
K = from_points([(x, y) for x in range(1, 6)
                 for y in range(1, 6)], 7)
beta = K - ero(K, B_rect)
print("|β(A)| =", int(beta.sum()))               # ได้ 16 พิกเซล

# ตัวอย่างที่ 6.11 (หัวข้อ 6.4.7): การเติมรูของวงแหวน 5x5
from scipy import ndimage as ndi
Rg = from_points([(x, y) for x in range(1, 6) for y in range(1, 6)
                  if not (2 <= x <= 4 and 2 <= y <= 4)], 7)
X = from_points([(3, 3)], 7)                 # จุดเริ่มต้นในรู
k = 0
while True:
    Xn = dil(X, B_cross) & (1 - Rg)          # (X ⊕ B) ∩ A^c
    if np.array_equal(Xn, X):
        break
    X, k = Xn, k + 1
filled = Rg | X
print("จำนวนรอบจนคงที่ =", k, " พื้นที่ก่อนเติม =", int(Rg.sum()),
      " หลังเติม =", int(filled.sum()))
print("ตรงกับ SciPy:",
      np.array_equal(filled, ndi.binary_fill_holes(Rg).astype(np.uint8)))


# ======================================================================
# โปรแกรม 6.4: การนับเมล็ดด้วยการกำหนดค่าขีดแบ่ง การเปิด และการติดป้ายองค์ประกอบเชื่อมต่อ
# ======================================================================
import numpy as np
import cv2

# 1) สร้างภาพทดสอบระดับเทา: วัตถุรูปวงรี 12 ชิ้น (คล้ายเมล็ดข้าว)
#    บนพื้นหลังมืด
rng = np.random.default_rng(6)
f = np.full((240, 320), 40, np.uint8)
for _ in range(12):
    cx, cy = rng.integers(30, 290), rng.integers(30, 210)
    ang = int(rng.integers(0, 180))
    cv2.ellipse(f, (int(cx), int(cy)), (18, 7), ang,
                0, 360, 200, -1)
noise = rng.normal(0, 12, f.shape).clip(0, 255).astype(np.uint8)
f = cv2.add(f, noise)
f[rng.random(f.shape) < 0.01] = 255   # จุดรบกวนสีขาว 1% ของพิกเซล

# 2) กำหนดค่าขีดแบ่งด้วยวิธีของ Otsu:
#    b(x,y) = 1 เมื่อ f(x,y) > T (สมการที่ (6.2)–(6.3))
flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
T, b = cv2.threshold(f, 0, 1, flags)

# 3) การเปิดด้วย B แบบสี่เหลี่ยม 3x3 เพื่อลบจุดรบกวน (สมการที่ (6.21))
B = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
b_open = cv2.morphologyEx(b, cv2.MORPH_OPEN, B)

# 4) ติดป้ายองค์ประกอบเชื่อมต่อแบบ 8 ทิศ
#    ด้วยขั้นตอนวิธี Spaghetti ของ Bolelli และคณะ [11]
n0 = cv2.connectedComponents(b, connectivity=8)[0] - 1
ccl = cv2.connectedComponentsWithStatsWithAlgorithm
n1, lab, stats, cent = ccl(b_open, 8, cv2.CV_32S, cv2.CCL_BOLELLI)
n1 -= 1                                # ป้าย 0 คือพื้นหลัง
S = stats[1:, cv2.CC_STAT_AREA]
print(f"T = {T:.0f}, องค์ประกอบก่อนเปิด = {n0}, หลังเปิด = {n1}")

# 5) ประมาณจำนวนวัตถุจากพื้นที่: องค์ประกอบที่ใหญ่ผิดปกติ
#    อาจเป็นวัตถุหลายชิ้นที่ซ้อนกัน
S_ref = np.median(S)                   # พื้นที่ของวัตถุหนึ่งชิ้นโดยประมาณ
n_est = int(np.round(S / S_ref).sum())
print(f"พื้นที่มัธยฐาน = {S_ref:.0f} พิกเซล, "
      f"พื้นที่สูงสุด = {S.max()} พิกเซล, จำนวนที่ประมาณได้ = {n_est}")

# 6) ตรวจจุดศูนย์กลางมวลขององค์ประกอบแรกกับสมการที่ (6.8)
ys, xs = np.nonzero(lab == 1)
print(f"องค์ประกอบ 1: S = {xs.size}, "
      f"(x̄, ȳ) = ({xs.mean():.2f}, {ys.mean():.2f})\n"
      f"  OpenCV = ({cent[1, 0]:.2f}, {cent[1, 1]:.2f})")
