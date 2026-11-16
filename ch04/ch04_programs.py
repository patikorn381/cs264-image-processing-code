"""CS264 การประมวลผลภาพเชิงดิจิทัล: โปรแกรมของบทที่ 4
รันในโฟลเดอร์ว่าง ตามลำดับจากบนลงล่าง (โปรแกรมหลัง ๆ ใช้ฟังก์ชันและไฟล์จากโปรแกรมก่อนหน้า)
"""

# ======================================================================
# โปรแกรม 4.1: การคำนวณเกรเดียนต์ด้วยหน้ากาก Prewitt, Sobel และ Scharr
# ======================================================================
import numpy as np
import cv2
from skimage import data

# หน้ากากสำหรับสหสัมพันธ์ (x = คอลัมน์ไปทางขวา, y = แถวลงล่าง)
w = {"Prewitt": (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
                          np.float32),
                 np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
                          np.float32)),
     "Sobel":   (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          np.float32),
                 np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          np.float32)),
     "Scharr":  (np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                          np.float32),
                 np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
                          np.float32))}

# 1) ตรวจตัวอย่างที่ 4.3: ด้านซ้ายสว่าง ด้านขวามืด
f = np.array([[100, 0, 0]] * 3, np.float32)
for name, (wx, wy) in w.items():
    # สหสัมพันธ์ที่พิกเซลกึ่งกลาง
    gx, gy = (f * wx).sum(), (f * wy).sum()
    print(f"{name:7s} gx = {gx:6.0f}  gy = {gy:4.0f}  "
          f"M = {np.hypot(gx, gy):5.0f}  theta = "
          f"{np.degrees(np.arctan2(gy, gx)):6.1f} องศา")

# 2) ภาพจริง: cv2.Sobel และ cv2.Scharr ให้ผลเท่ากับสหสัมพันธ์
img = data.camera().astype(np.float32)
# ต้องใช้ชนิดทศนิยม ห้ามใช้ uint8
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
gx2 = cv2.filter2D(img, cv2.CV_32F, w["Sobel"][0])
print("ผลต่างสูงสุดระหว่าง cv2.Sobel กับ cv2.filter2D:",
      float(np.abs(gx - gx2).max()))
gs = cv2.Scharr(img, cv2.CV_32F, 1, 0)
gs2 = cv2.filter2D(img, cv2.CV_32F, w["Scharr"][0])
print("ผลต่างสูงสุดระหว่าง cv2.Scharr กับ cv2.filter2D:",
      float(np.abs(gs - gs2).max()))
M = np.hypot(gx, gy)                       # สมการที่ (4.2)
# สมการที่ (4.4) ด้วย atan2
theta = np.degrees(np.arctan2(gy, gx))
print(f"M สูงสุด = {M.max():.0f}, ร้อยละ 90 ของพิกเซลมี M ต่ำกว่า "
      f"{np.percentile(M, 90):.0f}")

# 3) ความผิดพลาดที่พบบ่อย: เก็บผลเป็น uint8 ทำให้ค่าลบถูกตัดทิ้ง
gx_u8 = cv2.Sobel(data.camera(), cv2.CV_8U, 1, 0, ksize=3)
print("ร้อยละของขอบที่ gx < -50 (สว่างไปมืดตามแนว x)",
      "ซึ่งหายไปเมื่อใช้ uint8:",
      round(100 * np.mean(gx_u8[gx < -50] == 0), 1))

# 4) ภาพขอบสองระดับด้วยค่าขีดแบ่ง τ (สมการที่ (4.13))
tau = np.percentile(M, 90)
E = (M >= tau).astype(np.uint8) * 255
cv2.imwrite("camera_sobel_edges.png", E)


# ======================================================================
# โปรแกรม 4.2: วิธี Marr–Hildreth และการเปรียบเทียบ LoG กับ DoG
# ======================================================================
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import data

def zero_crossing(L, t):
    """จุดตัดศูนย์: เครื่องหมายต่างกันในคู่เพื่อนบ้านตรงข้าม
    และผลต่างมากกว่า t (สมการที่ (4.15))"""
    P = np.pad(L, 1, mode="edge")
    H, W = L.shape
    z = np.zeros(L.shape, bool)
    # ซ้าย-ขวา, บน-ล่าง และแนวทแยงทั้งสอง
    pairs = [((1, 2), (1, 0)), ((2, 1), (0, 1)),
             ((2, 2), (0, 0)), ((2, 0), (0, 2))]
    for (a, b), (c, d) in pairs:
        p, q = P[a:a + H, b:b + W], P[c:c + H, d:d + W]
        z |= (np.sign(p) * np.sign(q) < 0) & (np.abs(p - q) > t)
    return z

# 1) ตรวจตัวอย่างที่ 4.4: อนุพันธ์อันดับสองของโพรไฟล์ทางลาด
p = np.array([20, 20, 20, 40, 80, 100, 100, 100], float)
print("∂²f/∂x² (x = 1..6):", p[2:] + p[:-2] - 2 * p[1:-1])

# 2) Marr–Hildreth: LoG ที่ σ ต่าง ๆ แล้วหาจุดตัดศูนย์
f = data.camera().astype(np.float64)
for sigma in [1.0, 2.0, 4.0]:
    # (∇²G_σ) * f  สมการที่ (4.14)
    L = ndi.gaussian_laplace(f, sigma)
    # ค่าขีดแบ่งของผลต่างคร่อมจุดตัด
    t = 0.04 * np.abs(L).max()
    z0 = zero_crossing(L, 0.0)        # ไม่ใช้ค่าขีดแบ่ง
    z = zero_crossing(L, t)
    print(f"σ = {sigma}: จุดตัดศูนย์ทั้งหมด {100*z0.mean():5.1f}% "
          f"ของภาพ, หลังใช้ค่าขีดแบ่ง {100*z.mean():4.1f}%")
    cv2.imwrite(f"camera_log_s{sigma:.0f}.png",
                z.astype(np.uint8) * 255)

# 3) เปรียบเทียบ LoG กับ DoG (σ2/σ1 = 1.6) ตามหัวข้อ 2.5.6
s1 = 2.0
dog = (ndi.gaussian_filter(f, 1.6 * s1)
       - ndi.gaussian_filter(f, s1))
L = ndi.gaussian_laplace(f, 1.26 * s1)   # σ ระหว่าง σ1 และ σ2
r = np.corrcoef(dog.ravel(), L.ravel())[0, 1]
print(f"สหสัมพันธ์ระหว่างภาพ DoG กับภาพ LoG: {r:.4f}")


# ======================================================================
# โปรแกรม 4.3: ตัวตรวจจับขอบแบบ Canny ทีละขั้น และการเปรียบเทียบกับ OpenCV
# ======================================================================
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import data

def nms(M, gx, gy):
    """ขั้นที่ 3 ของ Canny: เก็บเฉพาะค่าสูงสุดตามทิศเกรเดียนต์
    (ควอนไทซ์ 4 ทิศ)"""
    # มุมในช่วง [0, 180)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    P = np.pad(M, 1)                     # เติมขอบด้วยศูนย์
    c = P[1:-1, 1:-1]
    E, W = P[1:-1, 2:], P[1:-1, :-2]     # (x+1,y), (x-1,y)
    S, N = P[2:, 1:-1], P[:-2, 1:-1]     # (x,y+1), (x,y-1)
    SE, NW = P[2:, 2:], P[:-2, :-2]
    SW, NE = P[2:, :-2], P[:-2, 2:]
    d0 = (ang < 22.5) | (ang >= 157.5)
    d45 = (ang >= 22.5) & (ang < 67.5)
    d90 = (ang >= 67.5) & (ang < 112.5)
    d135 = (ang >= 112.5) & (ang < 157.5)
    keep = ((d0 & (c >= E) & (c >= W))
            | (d45 & (c >= SE) & (c >= NW))
            | (d90 & (c >= S) & (c >= N))
            | (d135 & (c >= SW) & (c >= NE)))
    return np.where(keep, M, 0.0)        # สมการที่ (4.19)

def hysteresis(N, t_low, t_high):
    """ขั้นที่ 4–5 ของ Canny: ค่าขีดแบ่งสองระดับ
    และการเชื่อมขอบแบบ 8 ทิศ"""
    # กลุ่มของพิกเซลขอบอ่อนและเข้ม
    lab, _ = ndi.label(N >= t_low, structure=np.ones((3, 3)))
    ids = np.unique(lab[N >= t_high])    # กลุ่มที่มีพิกเซลขอบเข้ม
    return np.isin(lab, ids[ids > 0])

def canny(f, sigma, t_low, t_high):
    """ตัวตรวจจับขอบแบบ Canny ตามขั้นตอน 5 ขั้นในหัวข้อ 4.5.2"""
    fs = cv2.GaussianBlur(f.astype(np.float32), (0, 0),
                          sigma)                        # ขั้นที่ 1
    gx = cv2.Sobel(fs, cv2.CV_32F, 1, 0, ksize=3)       # ขั้นที่ 2
    gy = cv2.Sobel(fs, cv2.CV_32F, 0, 1, ksize=3)
    M = np.hypot(gx, gy)
    N = nms(M, gx, gy)                                  # ขั้นที่ 3
    return hysteresis(N, t_low, t_high)                 # ขั้นที่ 4–5

# 1) ตรวจตัวอย่างที่ 4.5 (ทิศเกรเดียนต์ 0 องศาทุกพิกเซล)
M = np.array([[10, 40, 90, 50, 10, 10], [10, 50, 80, 60, 10, 20],
              [10, 30, 45, 40, 10, 50], [10, 35, 55, 30, 10, 20],
              [10, 60, 95, 40, 10, 10]], float)
N = nms(M, np.ones_like(M), np.zeros_like(M))
print("หลัง NMS:\n", N.astype(int))
print("หลัง hysteresis (40, 80):\n",
      hysteresis(N, 40, 80).astype(int))

# 2) เทียบกับ cv2.Canny ซึ่งไม่ได้กรองเกาส์เซียนให้ จึงต้องกรองก่อนเอง
f = data.camera()
fs = cv2.GaussianBlur(f, (0, 0), 1.5)
# threshold1 = ค่าขีดแบ่งต่ำ, threshold2 = ค่าขีดแบ่งสูง
ocv = cv2.Canny(fs, 50, 100, L2gradient=True) > 0
mine = canny(f, 1.5, 50, 100)
print(f"พิกเซลขอบ: OpenCV {ocv.sum()}, โปรแกรมนี้ {mine.sum()}, "
      f"ตรงกัน {100 * (ocv & mine).sum() / ocv.sum():.1f}% "
      "ของ OpenCV")

# 3) ผลของการไม่กรองก่อน และผลของ σ กับค่าขีดแบ่ง
print("ไม่กรองก่อน:",
      (cv2.Canny(f, 50, 100, L2gradient=True) > 0).sum(), "พิกเซล")
for sigma in [1.0, 2.0, 4.0]:
    e = cv2.Canny(cv2.GaussianBlur(f, (0, 0), sigma), 50, 100,
                  L2gradient=True)
    cv2.imwrite(f"camera_canny_s{sigma:.0f}.png", e)
    print(f"σ = {sigma}: {(e > 0).sum()} พิกเซล")
# (100, 100) คือค่าขีดแบ่งระดับเดียว
for lo, hi in [(25, 100), (50, 100), (100, 100)]:
    e = cv2.Canny(fs, lo, hi, L2gradient=True) > 0
    n, _ = cv2.connectedComponents(e.astype(np.uint8),
                                   connectivity=8)
    print(f"T_low = {lo:3d}, T_high = {hi}: {e.sum()} พิกเซล, "
          f"{n - 1} เส้นขอบ")


# ======================================================================
# โปรแกรม 4.4: ไฟล์ edge_lib.py ฟังก์ชัน NMS ฮิสเทอรีซิส จุดตัดศูนย์ และตัววัดผล
# ======================================================================
import numpy as np
import cv2
from scipy import ndimage as ndi

def nms(M, gx, gy):
    """ขั้นที่ 3 ของ Canny: เก็บเฉพาะค่าสูงสุดตามทิศเกรเดียนต์
    (ควอนไทซ์ 4 ทิศ)"""
    # มุมในช่วง [0, 180)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    P = np.pad(M, 1)                     # เติมขอบด้วยศูนย์
    c = P[1:-1, 1:-1]
    E, W = P[1:-1, 2:], P[1:-1, :-2]     # (x+1,y), (x-1,y)
    S, N = P[2:, 1:-1], P[:-2, 1:-1]     # (x,y+1), (x,y-1)
    SE, NW = P[2:, 2:], P[:-2, :-2]
    SW, NE = P[2:, :-2], P[:-2, 2:]
    d0 = (ang < 22.5) | (ang >= 157.5)
    d45 = (ang >= 22.5) & (ang < 67.5)
    d90 = (ang >= 67.5) & (ang < 112.5)
    d135 = (ang >= 112.5) & (ang < 157.5)
    keep = ((d0 & (c >= E) & (c >= W))
            | (d45 & (c >= SE) & (c >= NW))
            | (d90 & (c >= S) & (c >= N))
            | (d135 & (c >= SW) & (c >= NE)))
    return np.where(keep, M, 0.0)        # สมการที่ (4.19)

def hysteresis(N, t_low, t_high):
    """ขั้นที่ 4–5 ของ Canny: ค่าขีดแบ่งสองระดับ
    และการเชื่อมขอบแบบ 8 ทิศ"""
    # กลุ่มของพิกเซลขอบอ่อนและเข้ม
    lab, _ = ndi.label(N >= t_low, structure=np.ones((3, 3)))
    ids = np.unique(lab[N >= t_high])    # กลุ่มที่มีพิกเซลขอบเข้ม
    return np.isin(lab, ids[ids > 0])

def zero_crossing(L, t):
    """จุดตัดศูนย์: เครื่องหมายต่างกันในคู่เพื่อนบ้านตรงข้าม
    และผลต่างมากกว่า t (สมการที่ (4.15))"""
    P = np.pad(L, 1, mode="edge")
    H, W = L.shape
    z = np.zeros(L.shape, bool)
    # ซ้าย-ขวา, บน-ล่าง และแนวทแยงทั้งสอง
    pairs = [((1, 2), (1, 0)), ((2, 1), (0, 1)),
             ((2, 2), (0, 0)), ((2, 0), (0, 2))]
    for (a, b), (c, d) in pairs:
        p, q = P[a:a + H, b:b + W], P[c:c + H, d:d + W]
        z |= (np.sign(p) * np.sign(q) < 0) & (np.abs(p - q) > t)
    return z

def f1_score(det, gt, tol=1):
    """ความแม่น ความครบ และ F1 เมื่อยอมให้คลาดเคลื่อนได้ tol พิกเซล
    (สมการที่ (4.22))"""
    if det.sum() == 0:
        return 0.0, 0.0, 0.0
    P = np.mean(ndi.distance_transform_edt(~gt)[det] <= tol)
    R = np.mean(ndi.distance_transform_edt(~det)[gt] <= tol)
    return P, R, (0.0 if P + R == 0 else 2 * P * R / (P + R))


# ======================================================================
# โปรแกรม 4.5: การทดลองเปรียบเทียบตัวตรวจจับขอบ
# ======================================================================
import numpy as np
import cv2
from scipy import ndimage as ndi
# ฟังก์ชันจากโปรแกรม 4.4: ใน Colab ให้รันเซลล์นั้นก่อน
try:
    from edge_lib import nms, hysteresis, zero_crossing, f1_score
except ImportError:
    pass

def make_scene(n=256):
    """ภาพสังเคราะห์ที่รู้ตำแหน่งขอบจริง: วัตถุ 3 ชิ้น
    ความเปรียบต่าง 60, 40 และ 30"""
    lab = np.zeros((n, n), np.uint8)
    cv2.rectangle(lab, (30, 30), (110, 110), 1, -1)
    cv2.circle(lab, (180, 80), 45, 2, -1)
    tri = np.array([[60, 230], [130, 140], [200, 230]], np.int32)
    cv2.fillPoly(lab, [tri], 3)
    f = np.array([100, 160, 60, 130], np.float32)[lab]
    f = cv2.GaussianBlur(f, (0, 0), 1.0)      # ขอบแบบทางลาด
    gt = np.zeros(lab.shape, bool)             # ขอบจริง
    gt[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    gt[1:, :] |= lab[1:, :] != lab[:-1, :]
    return f, gt

def best(detect, ts, gt):
    """เลือกค่าขีดแบ่งที่ให้ F1 สูงสุด
    (ใช้ภาพขอบจริง จึงเป็นขอบเขตบนของแต่ละวิธี)"""
    return max((f1_score(detect(t), gt) for t in ts),
               key=lambda r: r[2])

def pct(a, lo, hi, n=80):
    return np.percentile(a, np.linspace(lo, hi, n))

W = {"prewitt": (np.array([[-1, 0, 1]] * 3, np.float32), None),
     "sobel": (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        np.float32), None)}

def grad(g, name):
    if name == "roberts":
        P = np.pad(g, ((0, 1), (0, 1)), mode="edge")
        return P[:-1, :-1] - P[1:, 1:], P[:-1, 1:] - P[1:, :-1]
    wx = W[name][0]
    return cv2.filter2D(g, -1, wx), cv2.filter2D(g, -1, wx.T)

f0, gt = make_scene()
methods = ["Roberts", "Prewitt", "Sobel", "เกาส์เซียน+Sobel",
           "เกาส์เซียน+Sobel+NMS", "Canny (โปรแกรม 4.3)",
           "cv2.Canny", "LoG σ=2"]
table = {m: [] for m in methods}
for s_eta in [5, 10, 20]:
    rng = np.random.default_rng(400 + s_eta)   # seed คงที่
    g = f0 + rng.normal(0, s_eta, f0.shape)
    g = np.clip(g, 0, 255).astype(np.float32)
    for m in ["Roberts", "Prewitt", "Sobel"]:
        M = np.hypot(*grad(g, m.lower()))
        table[m].append(best(lambda t: M >= t,
                             pct(M, 50, 99.5), gt))
    gs = cv2.GaussianBlur(g, (0, 0), 1.5)
    gx, gy = grad(gs, "sobel")
    M = np.hypot(gx, gy)
    N = nms(M, gx, gy)
    table["เกาส์เซียน+Sobel"].append(
        best(lambda t: M >= t, pct(M, 50, 99.5), gt))
    table["เกาส์เซียน+Sobel+NMS"].append(
        best(lambda t: N >= t, pct(N[N > 0], 1, 99), gt))
    table["Canny (โปรแกรม 4.3)"].append(
        best(lambda t: hysteresis(N, 0.5 * t, t),
             pct(N[N > 0], 1, 99.5), gt))
    u8 = np.clip(gs, 0, 255).astype(np.uint8)
    table["cv2.Canny"].append(
        best(lambda t: cv2.Canny(u8, 0.5 * t, t,
                                 L2gradient=True) > 0,
             np.linspace(5, 400, 80), gt))
    L = ndi.gaussian_laplace(g.astype(np.float64), 2.0)
    table["LoG σ=2"].append(
        best(lambda t: zero_crossing(L, t),
             np.linspace(0, 0.5 * np.abs(L).max(), 80), gt))

print(f"{'วิธี':24s}{'σ_η = 5':>12s}{'σ_η = 10':>12s}"
      f"{'σ_η = 20':>12s}   (P / R / F1)")
for m in methods:
    print(f"{m:24s}" + "".join(f"  {P:.2f}/{R:.2f}/{F:.3f}"
                              for P, R, F in table[m]))

# ผลของ σ ต่อการตรวจพบและความแม่นยำของตำแหน่ง (σ_η = 10)
rng = np.random.default_rng(410)
g = np.clip(f0 + rng.normal(0, 10, f0.shape), 0, 255)
g = g.astype(np.float32)
d_gt = ndi.distance_transform_edt(~gt)
print("\nσ     F1     ระยะเฉลี่ยจากขอบจริง (พิกเซล)")
for sigma in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
    gs = cv2.GaussianBlur(g, (0, 0), sigma)
    u8 = np.clip(gs, 0, 255).astype(np.uint8)
    det = lambda t: cv2.Canny(u8, 0.5 * t, t,
                              L2gradient=True) > 0
    ts = np.linspace(5, 400, 80)
    t_best = max(ts, key=lambda t: f1_score(det(t), gt)[2])
    e = det(t_best)
    print(f"{sigma:3.1f}  {f1_score(e, gt)[2]:.3f}   "
          f"{d_gt[e & (d_gt <= 3)].mean():.2f}")


# ======================================================================
# โปรแกรม 4.6: การจำลองผลของการกรองก่อน Canny กับภาพผิวน้ำ
# ======================================================================
import numpy as np
import cv2
from scipy import ndimage as ndi

def pool_scene(A_wave, rng, n=256):
    """ภาพจำลองผิวน้ำ: คลื่น + จุดสะท้อนแสง + วัตถุรูปวงรี
    (ความเปรียบต่าง 60)"""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    water = np.full((n, n), 120, np.float32)
    # คลื่น 3 ชุด ทิศและคาบสุ่ม
    for _ in range(3):
        a, T = rng.uniform(0, np.pi), rng.uniform(10, 20)
        u = x * np.cos(a) + y * np.sin(a)
        water += A_wave / 3 * np.sin(2 * np.pi * u / T
                                     + rng.uniform(0, 2 * np.pi))
    for _ in range(40):              # จุดสะท้อนแสงจนอิ่มตัว
        c = (int(rng.integers(0, n)), int(rng.integers(0, n)))
        cv2.circle(water, c, int(rng.integers(1, 4)), 255, -1)
    body = np.zeros((n, n), np.uint8)
    cv2.ellipse(body, (128, 128), (25, 60), 20, 0, 360, 1, -1)
    f = np.where(body > 0, 180, water).astype(np.float32)
    f = cv2.GaussianBlur(f, (0, 0), 1.0)
    f = f + rng.normal(0, 5, (n, n))
    k = np.ones((3, 3), np.uint8)
    edge = cv2.morphologyEx(body, cv2.MORPH_GRADIENT, k) > 0
    return np.clip(f, 0, 255).astype(np.uint8), edge

pre = {"ไม่กรอง": lambda f: f,
       "เกาส์เซียน σ = 2":
           lambda f: cv2.GaussianBlur(f, (0, 0), 2),
       "สองด้าน (หัวข้อ 3.6.4)":
           lambda f: cv2.bilateralFilter(f, 12, 100, 150),
       "สองด้าน σr = 30":
           lambda f: cv2.bilateralFilter(f, -1, 30, 2)}
print("คลื่น  การกรองก่อน Canny       (T_low, T_high)  "
      "ขอบวัตถุที่พบ  ขอบปลอมในน้ำ")
for A in [10, 30, 60]:
    f, gt = pool_scene(A, np.random.default_rng(7))
    # แถบรอบขอบวัตถุ
    near = ndi.distance_transform_edt(~gt) <= 3
    for name, fn in pre.items():
        # (100, 200) คือค่าขีดแบ่งตัวอย่าง
        for lo, hi in [(100, 200), (30, 60)]:
            e = cv2.Canny(fn(f), lo, hi) > 0
            d = ndi.distance_transform_edt(~e)
            recall = np.mean(d[gt] <= 2)
            false = np.mean(e[~near])
            print(f"{A:4d}  {name:22s}  ({lo:3d}, {hi:3d})  "
                  f"{100 * recall:10.1f}%  {100 * false:10.2f}%")
