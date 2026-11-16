"""CS264 การประมวลผลภาพเชิงดิจิทัล: โปรแกรมของบทที่ 5
รันในโฟลเดอร์ว่าง ตามลำดับจากบนลงล่าง (โปรแกรมหลัง ๆ ใช้ฟังก์ชันและไฟล์จากโปรแกรมก่อนหน้า)
"""

# ======================================================================
# โปรแกรม 5.1: การแปลงแบบจำลองสี
# ======================================================================
import numpy as np
import cv2

def rgb2ycbcr(R, G, B):
    """สมการที่ (5.3)–(5.5): JPEG/JFIF ช่วงเต็ม 0–255"""
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 0.564 * (B - Y) + 128
    Cr = 0.713 * (R - Y) + 128
    return Y, Cb, Cr

def ycbcr2rgb(Y, Cb, Cr):
    """สมการที่ (5.6)–(5.8)"""
    R = Y + 1.403 * (Cr - 128)
    G = Y - 0.344 * (Cb - 128) - 0.714 * (Cr - 128)
    B = Y + 1.773 * (Cb - 128)
    return R, G, B

def rgb2hsi(R, G, B):
    """สมการที่ (5.9)–(5.12): R, G, B อยู่ในช่วง [0, 1]
    H เป็นองศา"""
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-12
    theta = np.degrees(np.arccos(np.clip(num / den, -1, 1)))
    H = theta if B <= G else 360 - theta
    I = (R + G + B) / 3
    S = 1 - min(R, G, B) / I if I > 0 else 0.0
    return H, S, I

def rgb2hsv(R, G, B):
    """สมการที่ (5.13)–(5.14): R, G, B อยู่ในช่วง [0, 1]"""
    V, m = max(R, G, B), min(R, G, B)
    C = V - m
    S = 0.0 if V == 0 else C / V
    if C == 0:   H = 0.0
    elif V == R: H = 60 * (((G - B) / C) % 6)
    elif V == G: H = 60 * ((B - R) / C + 2)
    else:        H = 60 * ((R - G) / C + 4)
    return H, S, V

# 1) ตัวอย่างที่ 5.1: RGB -> YCbCr และย้อนกลับ
Y, Cb, Cr = rgb2ycbcr(20, 100, 250)
print(f"(20,100,250) -> Y = {Y:.2f}, "
      f"Cb = {Cb:.2f}, Cr = {Cr:.2f}")
q = np.round([Y, Cb, Cr])
print("ปัดเป็นจำนวนเต็ม:", q.astype(int),
      "-> RGB =", np.round(ycbcr2rgb(*q), 2))

# 2) OpenCV: ลำดับช่องผลลัพธ์คือ Y, Cr, Cb และภาพนำเข้าเป็น BGR
px = np.uint8([[[250, 100, 20]]])  # B, G, R
print("cv2 COLOR_BGR2YCrCb:",
      cv2.cvtColor(px, cv2.COLOR_BGR2YCrCb)[0, 0])

# 3) ตัวอย่างที่ 5.2: HSI และ HSV ของ (R, G, B) = (0.8, 0.4, 0.2)
H, S, I = rgb2hsi(0.8, 0.4, 0.2)
print(f"HSI: H = {H:.2f} องศา, S = {S:.4f}, I = {I:.4f}")
H, S, V = rgb2hsv(0.8, 0.4, 0.2)
print(f"HSV: H = {H:.2f} องศา, S = {S:.4f}, V = {V:.4f}")
# (0.2, 0.4, 0.8) x 255 แบบ BGR
px = np.uint8([[[51, 102, 204]]])
hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
print("cv2 COLOR_BGR2HSV:", hsv, "(H เก็บเป็นองศา/2)")
lab_f = cv2.cvtColor(np.float32([[[0.2, 0.4, 0.8]]]),
                     cv2.COLOR_BGR2Lab)[0, 0]
lab_u8 = cv2.cvtColor(px, cv2.COLOR_BGR2Lab)[0, 0]
print("cv2 COLOR_BGR2Lab:", lab_u8,
      "| float:", np.round(lab_f, 2))

# 4) จำนวนระดับสีเมื่อแต่ละช่องมี n ระดับ (เช่น 5 x 5 x 5)
for n in [5, 8, 256]:
    bits = int(np.ceil(np.log2(n**3)))
    print(f"{n} ระดับต่อช่อง: {n**3:,} สี "
          f"ต้องใช้ {bits} บิตต่อพิกเซล")


# ======================================================================
# โปรแกรม 5.2: การปรับฮิสโทแกรมให้เท่ากันภาพสีด้วยแบบจำลองต่าง ๆ
# ======================================================================
import numpy as np
import cv2
from skimage import data

def hsv_full(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL).astype(float)

def hue_shift(a, b):
    """การเปลี่ยนเฉดสีเฉลี่ย (องศา) ถ่วงน้ำหนักด้วยความอิ่มตัว
    เพราะเฉดของสีเทาไม่มีความหมาย"""
    ha, sa = hsv_full(a)[..., 0], hsv_full(a)[..., 1]
    hb = hsv_full(b)[..., 0]
    d = np.abs(ha - hb) * 360 / 256
    d = np.minimum(d, 360 - d)  # มุมวนรอบ 360 องศา
    w = sa / 255
    return (d * w).sum() / w.sum()

def he_luma(img, code, inv, ch=0, clahe=None):
    """ปรับฮิสโทแกรมให้เท่ากันเฉพาะช่องความสว่าง
    ของปริภูมิสีที่กำหนด"""
    z = cv2.cvtColor(img, code)
    c = z[..., ch]
    z[..., ch] = clahe.apply(c) if clahe else cv2.equalizeHist(c)
    return cv2.cvtColor(z, inv)

def he_hsi(img):
    """ปรับเฉพาะ I = (R+G+B)/3 แล้วคูณทุกช่องด้วย I'/I
    (คงเฉดและความอิ่มตัวของ HSI)"""
    f = img.astype(np.float64)
    I = f.mean(axis=2)
    I2 = cv2.equalizeHist(np.round(I).astype(np.uint8))
    I2 = I2.astype(np.float64)
    g = f * (I2 / np.maximum(I, 1e-6))[..., None]
    # สัดส่วนพิกเซลที่ออกนอกขอบเขตสี
    clipped = np.mean(g.max(axis=2) > 255)
    return np.clip(np.round(g), 0, 255).astype(np.uint8), clipped

# ภาพทดสอบ: astronaut (สาธารณสมบัติ)
# ที่ลดความเปรียบต่างและความสว่างลง
f0 = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
# ภาพความเปรียบต่างต่ำ
f = np.clip(0.45 * f0 + 30, 0, 255).astype(np.uint8)
Y = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
LAB, LAB_INV = cv2.COLOR_BGR2Lab, cv2.COLOR_Lab2BGR

res = {"แยกช่อง R, G, B":
       cv2.merge([cv2.equalizeHist(c) for c in cv2.split(f)])}
res["ช่อง I ของ HSI"], clip_hsi = he_hsi(f)
res["ช่อง V ของ HSV"] = he_luma(f, cv2.COLOR_BGR2HSV,
                                cv2.COLOR_HSV2BGR, 2)
res["ช่อง Y ของ YCbCr"] = he_luma(f, cv2.COLOR_BGR2YCrCb,
                                  cv2.COLOR_YCrCb2BGR, 0)
res["ช่อง L* ของ CIELAB"] = he_luma(f, LAB, LAB_INV, 0)
res["CLAHE ช่อง L*"] = he_luma(f, LAB, LAB_INV, 0, clahe)

print("ภาพนำเข้า: ส่วนเบี่ยงเบนมาตรฐานของความสว่าง = "
      f"{Y(f).std():.1f}")
print(f"{'วิธี':20s}{'SD ความสว่าง':>14s}"
      f"{'เฉดเปลี่ยน (องศา)':>18s}{'ΔE*ab เฉลี่ย':>13s}")
lab0 = cv2.cvtColor(f.astype(np.float32) / 255, LAB)
for name, g in res.items():
    lab = cv2.cvtColor(g.astype(np.float32) / 255, LAB)
    # ต่างเฉพาะสี ไม่รวม L*
    dab = np.hypot(lab[..., 1] - lab0[..., 1],
                   lab[..., 2] - lab0[..., 2]).mean()
    print(f"{name:20s}{Y(g).std():14.1f}"
          f"{hue_shift(f, g):18.2f}{dab:13.2f}")
    cv2.imwrite(f"he_{list(res).index(name)}.png", g)
print(f"HSI: พิกเซลที่ต้องตัดค่าเพราะเกิน 255 = {100 * clip_hsi:.1f}%")


# ======================================================================
# โปรแกรม 5.3: การแบ่งส่วนวัตถุสีแดงด้วยค่าขีดแบ่งใน RGB และ HSV
# ======================================================================
import numpy as np
import cv2

def scene(gain, rng, n=256):
    """ฉากสังเคราะห์: วัตถุสีแดงและสีส้มบนพื้นเทา
    ส่องสว่างด้วยความเข้มลดลงจากซ้ายไปขวา"""
    lab = np.zeros((n, n), np.uint8)
    # วัตถุสีแดง (เป้าหมาย)
    cv2.circle(lab, (70, 128), 45, 1, -1)
    # วัตถุสีส้ม (ไม่ใช่เป้าหมาย)
    cv2.rectangle(lab, (150, 80), (230, 180), 2, -1)
    color = np.float32([[128, 128, 128],  # BGR
                        [40, 30, 200],
                        [30, 120, 230]])
    f = color[lab]
    # แสงไม่สม่ำเสมอ
    shade = np.linspace(1.0, 0.35, n, dtype=np.float32)
    f = f * shade[None, :, None] * gain
    f = f + rng.normal(0, 4, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8), lab == 1

def iou(a, b):
    return (a & b).sum() / (a | b).sum()

def seg_rgb(f):  # ค่าขีดแบ่งที่ปรับจากภาพที่แสงเต็ม
    B, G, R = [c.astype(int) for c in cv2.split(f)]
    return (R > 120) & (G < 80) & (B < 80)

def seg_hsv(f, wrap=True):
    # H อยู่ในช่วง 0–179
    H, S, V = cv2.split(cv2.cvtColor(f, cv2.COLOR_BGR2HSV))
    # สีแดงอยู่ทั้งสองปลายของวงล้อสี
    hue = (H <= 6) | (H >= 174) if wrap else (H <= 6)
    return hue & (S > 120) & (V > 30)

rng = np.random.default_rng(5)
print("gain  IoU(RGB)  IoU(HSV)  IoU(HSV ไม่คิดการวนของเฉด)")
for gain in [1.0, 0.8, 0.6, 0.4]:
    f, gt = scene(gain, rng)
    a = iou(seg_rgb(f), gt)
    b = iou(seg_hsv(f), gt)
    c = iou(seg_hsv(f, wrap=False), gt)
    print(f"{gain:4.1f}  {a:8.3f}  {b:8.3f}  {c:8.3f}")
f, gt = scene(1.0, rng)
H = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 0][gt]
print("ค่า H ในวัตถุสีแดง: ร้อยละ",
      round(100 * np.mean(H >= 174), 1), "อยู่ช่วง 174–179")


# ======================================================================
# โปรแกรม 5.4: เมทริกซ์การแปลงทางเรขาคณิตและการเปรียบเทียบกับ OpenCV
# ======================================================================
import numpy as np
import cv2

def Tr(tx, ty):  # สมการที่ (5.19)
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], float)

def S(sx, sy):  # สมการที่ (5.21)
    return np.diag([sx, sy, 1.0])

def Sh(shx, shy):  # สมการที่ (5.27)
    return np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]], float)

def Ro(deg):  # สมการที่ (5.23)
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t), np.cos(t), 0],
                     [0, 0, 1]])

def Rc(deg, xc, yc):  # สมการที่ (5.24)
    return Tr(xc, yc) @ Ro(deg) @ Tr(-xc, -yc)

np.set_printoptions(precision=4, suppress=True)
# 1) ตัวอย่างที่ 5.6: S -> Sh -> Rc -> Tr
Map = Tr(10, -5) @ Rc(30, 50, 50) @ Sh(0.2, 0) @ S(1.5, 1.5)
print("Map =\n", Map)
p = np.array([20, 10, 1.0])
print("(20, 10) ->", (Map @ p)[:2])
Swap = S(1.5, 1.5) @ Sh(0.2, 0) @ Rc(30, 50, 50) @ Tr(10, -5)
print("สลับลำดับเป็น S·Sh·Rc·Tr:", (Swap @ p)[:2])
print("ตรวจย้อนกลับด้วย Map^-1:", (np.linalg.inv(Map) @ Map @ p)[:2])

# 2) ทิศของการหมุนเมื่อแกน y ชี้ลง: หมุนจุด (1, 0) ไป 90 องศา
print("Ro(90)·(1, 0) =", (Ro(90) @ [1, 0, 1])[:2],
      "-> ชี้ลงล่างบนจอ = ตามเข็มนาฬิกา")

# 3) cv2.getRotationMatrix2D ใช้มุมบวก = ทวนเข็มนาฬิกาบนจอ
#    จึงเท่ากับ Rc(-θ)
A = cv2.getRotationMatrix2D((50, 50), 30, 1.0)
print("cv2.getRotationMatrix2D(30) =\n", A)
print("ตรงกับ Rc(-30)?", np.allclose(A, Rc(-30, 50, 50)[:2]),
      "| ตรงกับ Rc(+30)?", np.allclose(A, Rc(30, 50, 50)[:2]))

# 4) เมทริกซ์การหมุนเป็นเมทริกซ์เชิงตั้งฉาก: Ro^-1 = Ro^T
print("Ro(30)^-1 == Ro(30)^T ?",
      np.allclose(np.linalg.inv(Ro(30)), Ro(30).T))

# 5) cv2.warpAffine ทำการแมปย้อนกลับเอง:
#    ใส่เมทริกซ์ไปข้างหน้า แล้วฟังก์ชันหาอินเวิร์สให้
# จุดเดียวที่ (x, y) = (20, 10)
f = np.zeros((100, 100), np.uint8); f[10, 20] = 255
g = cv2.warpAffine(f, Map[:2], (100, 100),
                   flags=cv2.INTER_NEAREST)
ys, xs = np.nonzero(g)
print("ตำแหน่งจุดในภาพผลลัพธ์ (x, y):",
      list(zip(xs.tolist(), ys.tolist())))


# ======================================================================
# โปรแกรม 5.5: การแมปไปข้างหน้า การประมาณค่าเชิงเส้นสองทิศทาง และการเปรียบเทียบวิธีประมาณค่า
# ======================================================================
import time
import numpy as np
import cv2
from skimage import data

def forward_map(f, A):
    """การแมปไปข้างหน้า: ส่งพิกเซลต้นทางทุกตัว
    ไปยังตำแหน่งปลายทางที่ปัดเศษแล้ว"""
    M, N = f.shape
    y, x = np.mgrid[0:M, 0:N]
    u = np.rint(A[0, 0] * x + A[0, 1] * y + A[0, 2]).astype(int)
    v = np.rint(A[1, 0] * x + A[1, 1] * y + A[1, 2]).astype(int)
    g = np.zeros_like(f); hit = np.zeros(f.shape, bool)
    ok = (u >= 0) & (u < N) & (v >= 0) & (v < M)
    g[v[ok], u[ok]] = f[y[ok], x[ok]]; hit[v[ok], u[ok]] = True
    return g, hit

def bilinear(f, x, y):
    """สมการที่ (5.35): ประมาณค่า f ที่พิกัดจำนวนจริง (x, y)
    จาก 4 พิกเซลรอบข้าง"""
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    a, b = x - x0, y - y0
    return ((1 - a) * (1 - b) * f[y0, x0]
            + a * (1 - b) * f[y0, x0 + 1]
            + (1 - a) * b * f[y0 + 1, x0]
            + a * b * f[y0 + 1, x0 + 1])

f = data.camera()
M, N = f.shape
# 1) จุดโหว่ของการแมปไปข้างหน้า เทียบกับพื้นที่ที่ควรมีค่า
#    (คำนวณด้วยการแมปย้อนกลับ)
cases = [("หมุน 30 องศา",
          cv2.getRotationMatrix2D((N / 2, M / 2), 30, 1.0)),
         ("ขยาย 2 เท่า",
          np.float64([[2, 0, -N / 2], [0, 2, -M / 2]]))]
for name, A in cases:
    _, hit = forward_map(f, A)
    cover = cv2.warpAffine(np.ones_like(f), A, (N, M),
                           flags=cv2.INTER_NEAREST) > 0
    hole = 100 * np.mean(~hit[cover])
    print(f"{name}: จุดโหว่ = {hole:.1f}% ของพื้นที่ภาพผลลัพธ์")

# 2) ตัวอย่างที่ 5.7: bilinear ที่ (x, y) = (1.3, 2.6)
p = np.array([[0, 0, 0, 0], [0, 10, 20, 0],
              [0, 30, 60, 0], [0, 50, 100, 0]], float)
print("bilinear(1.3, 2.6) =", round(bilinear(p, 1.3, 2.6), 2))

# 3) หมุนสะสม 24 ครั้ง ครั้งละ 15 องศา (ครบ 360 องศา)
#    แล้วเทียบกับภาพเดิม
flags = {"Nearest": cv2.INTER_NEAREST,
         "Bilinear": cv2.INTER_LINEAR,
         "Bicubic": cv2.INTER_CUBIC,
         "Lanczos4": cv2.INTER_LANCZOS4}
A = cv2.getRotationMatrix2D((N / 2 - 0.5, M / 2 - 0.5), 15, 1.0)
yy, xx = np.mgrid[0:M, 0:N]
# บริเวณกลางภาพที่ไม่หลุดขอบ
roi = (xx - N / 2) ** 2 + (yy - M / 2) ** 2 < (0.35 * N) ** 2
print(f"{'วิธี':9s}{'PSNR (dB)':>10s}{'เวลา/ครั้ง (ms)':>17s}"
      f"{'overshoot ต่ำสุด/สูงสุด':>24s}")
# ขอบขั้นบันได 50 -> 200
step = np.zeros((64, 64), np.uint8)
step[:, 32:] = 200; step[:, :32] = 50
for name, fl in flags.items():
    g = f.astype(np.float32)
    for _ in range(24):
        g = cv2.warpAffine(g, A, (N, M), flags=fl,
                           borderMode=cv2.BORDER_REFLECT)
    mse = np.mean((g[roi] - f[roi]) ** 2)
    t0 = time.perf_counter()
    for _ in range(50):
        cv2.warpAffine(f, A, (N, M), flags=fl)
    t = (time.perf_counter() - t0) / 50 * 1000
    z = cv2.resize(step.astype(np.float32), None, fx=4, fy=4,
                   interpolation=fl)
    psnr = 10 * np.log10(255**2 / mse)
    print(f"{name:9s}{psnr:10.2f}{t:17.2f}"
          f"{z.min():12.1f} / {z.max():.1f}")


# ======================================================================
# โปรแกรม 5.6: การจำลองผลของวิธีย่อภาพและการเพิ่มความสว่างแบบ HSV กับเฟรมสระว่ายน้ำ
# ======================================================================
import numpy as np
import cv2

def pool_frame(rng, W=1920, H=1080):
    """เฟรมจำลองขนาด 1920 x 1080: น้ำสีฟ้า เชือกแบ่งลู่
    จุดสะท้อนแสง และผู้ว่ายน้ำ (ภาพ BGR)"""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    f = np.empty((H, W, 3), np.float32)
    f[...] = (170, 135, 50)  # สีน้ำ (B, G, R)
    # คลื่นผิวน้ำ คาบ 40 พิกเซล
    wave = np.sin(2 * np.pi * (x * 0.8 + y * 0.6) / 40)
    f += 12 * wave[..., None]
    # เชือกแบ่งลู่: ทุ่นสีแดง-ขาว คาบ 12 พิกเซล
    for y0 in range(120, H, 216):
        band = np.abs(y - y0) < 4
        red = (x // 6) % 2 == 0
        f[band & red] = (40, 40, 220)
        f[band & ~red] = (235, 235, 235)
    # ผู้ว่ายน้ำ
    cv2.ellipse(f, (960, 560), (40, 150), 70, 0, 360,
                (90, 130, 200), -1)
    for _ in range(60):  # จุดสะท้อนแสงจนอิ่มตัว
        c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        cv2.circle(f, c, int(rng.integers(3, 9)),
                   (255, 255, 255), -1)
    f = f + rng.normal(0, 3, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)

def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)

rng = np.random.default_rng(14)
f = pool_frame(rng)
# ย่อด้านยาวเหลือ 256 พิกเซล (7.5 เท่า)
size = (256, 144)
# ภาพอ้างอิง: กรองความถี่ต่ำก่อนชักตัวอย่างใหม่
# (σ ≈ 0.5 x อัตราการย่อ) แล้วจึงย่อ
ref = cv2.resize(cv2.GaussianBlur(f, (0, 0), 0.5 * 7.5), size,
                 interpolation=cv2.INTER_LINEAR)
rows = [int(round((y0 + 0.5) / 7.5 - 0.5))
        for y0 in range(120, 1080, 216)]
r = ref[rows, :, 2].astype(float)
print(f"ภาพอ้างอิง: ช่อง R บนแนวเชือก = "
      f"{r.mean():.1f} ± {r.std():.1f}")
print(f"{'วิธีย่อภาพ':12s}{'PSNR เทียบภาพอ้างอิง (dB)':>27s}"
      f"{'ช่อง R บนแนวเชือก (เฉลี่ย±SD)':>32s}")
for name, fl in [("NEAREST", cv2.INTER_NEAREST),
                 ("LINEAR", cv2.INTER_LINEAR),
                 ("CUBIC", cv2.INTER_CUBIC),
                 ("AREA", cv2.INTER_AREA)]:
    g = cv2.resize(f, size, interpolation=fl)
    rope = g[rows, :, 2].astype(float)  # ช่อง R บนแนวเชือก
    print(f"{name:12s}{psnr(g, ref):27.2f}"
          f"{rope.mean():20.1f} ± {rope.std():.1f}")

# การเพิ่มข้อมูลแบบ HSV ของ YOLOv8 (hsv_v = 0.4):
# สัดส่วนพิกเซลที่ V อิ่มตัวเมื่อเพิ่มความสว่างสูงสุด
small = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
V = hsv[..., 2].astype(float)
for gain in [0.0, 0.2, 0.4]:
    sat = 100 * np.mean(V * (1 + gain) >= 255)
    print(f"V x {1 + gain:.1f}: พิกเซลที่ V = 255 = {sat:5.1f}%")
# แสงจ้า: ความอิ่มตัวต่ำ ความสว่างสูง
glare = (hsv[..., 1] < 40) & (hsv[..., 2] > 230)
print("พิกเซลแสงจ้าตามเกณฑ์ S < 40 และ V > 230: "
      f"{100 * glare.mean():.2f}%")


# ======================================================================
# โปรแกรม 5.7: การจำลองผลของวิธีย่อภาพต่อผู้ว่ายน้ำขนาดเล็ก
# ======================================================================
import numpy as np
import cv2

def frame_with_swimmers(L, rng, W=1920, H=1080, n=24):
    """เฟรมจำลองที่มีผู้ว่ายน้ำรูปวงรียาว L พิกเซล n คน
    วางที่ตำแหน่งสุ่มแบบเศษพิกเซล"""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    f = np.empty((H, W, 3), np.float32); f[...] = (170, 135, 50)
    wave = np.sin(2 * np.pi * (x * 0.8 + y * 0.6) / 40)
    f += 12 * wave[..., None]
    cs = []
    for k in range(n):  # จัดเป็นกริด 6 x 4 แล้วเลื่อนแบบสุ่ม
        cx = 160 + 300 * (k % 6) + rng.uniform(-40, 40)
        cy = 135 + 270 * (k // 6) + rng.uniform(-40, 40)
        ang = rng.uniform(0, 180)
        m = np.zeros((H, W), np.uint8)
        # พิกัดละเอียด 1/16 พิกเซล
        center = (int(cx * 16), int(cy * 16))
        axes = (int(L / 2 * 16), int(L / 8 * 16))
        cv2.ellipse(m, center, axes, ang, 0, 360, 1, -1,
                    lineType=cv2.LINE_AA, shift=4)
        f[m > 0] = (90, 130, 200); cs.append((cx, cy))
    f = f + rng.normal(0, 3, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8), cs

def peak_contrast(g, cx, cy, s=7.5, r=3):
    """ความต่างสีสูงสุด (ระยะยุคลิดใน RGB) ระหว่างพิกเซล
    รอบตำแหน่งผู้ว่ายน้ำกับสีน้ำ"""
    u, v = int(round(cx / s)), int(round(cy / s))
    patch = g[v - r:v + r + 1, u - r:u + r + 1].astype(float)
    d = patch - np.array([170, 135, 50])
    return np.linalg.norm(d, axis=2).max()

rng = np.random.default_rng(27)
size = (256, 144)
full = np.linalg.norm(np.subtract((90, 130, 200), (170, 135, 50)))
print(f"ความต่างสีระหว่างผู้ว่ายน้ำกับน้ำในเฟรมเดิม = {full:.1f}")
print(f"{'ความยาว':>8s}{'หลังย่อ':>9s}{'LINEAR':>15s}"
      f"{'ต่ำสุด':>7s}{'AREA':>15s}{'ต่ำสุด':>7s}")
for L in [16, 24, 32, 48, 64]:
    f, cs = frame_with_swimmers(L, rng)
    row = f"{L:6d} px{L / 7.5:7.1f} px"
    for fl in [cv2.INTER_LINEAR, cv2.INTER_AREA]:
        g = cv2.resize(f, size, interpolation=fl)
        c = np.array([peak_contrast(g, *p) for p in cs])
        row += f"{c.mean():8.1f} ± {c.std():4.1f}{c.min():7.1f}"
    print(row)
