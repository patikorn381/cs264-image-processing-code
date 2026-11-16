"""CS264 การประมวลผลภาพเชิงดิจิทัล: โปรแกรมของบทที่ 3
รันในโฟลเดอร์ว่าง ตามลำดับจากบนลงล่าง (โปรแกรมหลัง ๆ ใช้ฟังก์ชันและไฟล์จากโปรแกรมก่อนหน้า)
"""

# ======================================================================
# โปรแกรม 3.1: การจำลองสัญญาณรบกวนแบบเกาส์เซียนและแบบเกลือและพริกไทย
# ======================================================================
import numpy as np
import cv2
from skimage import data

def add_gaussian_noise(f, sigma, rng):
    """เติมสัญญาณรบกวนแบบเกาส์เซียน
    ค่าเฉลี่ย 0 ส่วนเบี่ยงเบนมาตรฐาน sigma"""
    eta = rng.normal(0.0, sigma, f.shape)  # η(x,y)
    g = f.astype(np.float64) + eta  # คำนวณด้วย float ป้องกันค่าล้น
    return np.clip(np.round(g), 0, 255).astype(np.uint8)

def add_salt_pepper_noise(f, Ps, Pp, rng):
    """เติมสัญญาณรบกวนแบบเกลือและพริกไทย ด้วยความน่าจะเป็น Ps และ Pp"""
    g = f.copy()
    u = rng.random(f.shape)  # ค่าสุ่มสม่ำเสมอในช่วง [0, 1)
    g[u < Pp] = 0  # พริกไทย
    g[(u >= Pp) & (u < Pp + Ps)] = 255  # เกลือ
    return g

rng = np.random.default_rng(264)  # กำหนด seed ให้ผลซ้ำได้
f = data.camera()  # ภาพระดับเทา 512 × 512 (CC0)
g1 = add_gaussian_noise(f, 20, rng)
g2 = add_salt_pepper_noise(f, 0.025, 0.025, rng)
cv2.imwrite("camera_gauss20.png", g1)
cv2.imwrite("camera_sp05.png", g2)
clipped = np.mean((g1 == 0) | (g1 == 255))
print(f"ร้อยละของพิกเซลที่ถูกตัดค่า: {100*clipped:.2f}")
print(f"ร้อยละของพิกเซลเกลือและพริกไทย: {100*np.mean(g2 != f):.2f}")


# ======================================================================
# โปรแกรม 3.2: การคำนวณ MSE, PSNR และ SSIM
# ======================================================================
import numpy as np
import cv2
from skimage import data
from skimage.metrics import structural_similarity

def mse(f, g):
    d = f.astype(np.float64) - g.astype(np.float64)
    return np.mean(d ** 2)  # สมการที่ (3.9)

def psnr(f, g, L=256):
    e = mse(f, g)
    # สมการที่ (3.10)
    return np.inf if e == 0 else 10 * np.log10((L - 1) ** 2 / e)

def ssim(f, g):
    # ตั้งค่าตาม Wang และคณะ [7]:
    # หน้าต่างเกาส์เซียน σ = 1.5, K1 = 0.01, K2 = 0.03
    return structural_similarity(
        f, g, data_range=255, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False)

f = data.camera()  # ภาพต้นฉบับเดียวกับโปรแกรม 3.1
for name in ["camera_gauss20.png", "camera_sp05.png"]:
    g = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    print(f"{name:19s}MSE = {mse(f, g):8.2f}  "
          f"PSNR = {psnr(f, g):5.2f} dB  SSIM = {ssim(f, g):.4f}")
g = cv2.imread("camera_gauss20.png", cv2.IMREAD_GRAYSCALE)
print("ตรวจสอบกับ OpenCV:", round(cv2.PSNR(f, g), 2))


# ======================================================================
# โปรแกรม 3.3: ตัวกรองค่าเฉลี่ยและตัวกรองเกาส์เซียน
# ======================================================================
import numpy as np
import cv2
from skimage import data
from skimage.metrics import structural_similarity

def ssim(f, g):  # ตั้งค่าเดียวกับโปรแกรม 3.2
    return structural_similarity(
        f, g, data_range=255, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False)

f = data.camera()
# ภาพจากโปรแกรม 3.1
g = cv2.imread("camera_gauss20.png", cv2.IMREAD_GRAYSCALE)

# 1) หน้ากากเกาส์เซียนขนาด 3 × 3 ที่ OpenCV ใช้เมื่อไม่กำหนด σ
k = cv2.getGaussianKernel(3, 0)
print("หน้ากาก 1 มิติ:", k.ravel(),
      " หน้ากาก 2 มิติ × 16:\n", (k @ k.T) * 16)

# 2) ตัวกรองค่าเฉลี่ยขนาดต่าง ๆ
for m in [3, 5, 7, 9]:
    out = cv2.blur(g, (m, m))  # BORDER_REFLECT_101 เป็นค่าปริยาย
    p, q = cv2.PSNR(f, out), ssim(f, out)
    print(f"ค่าเฉลี่ย {m}×{m}     PSNR = {p:5.2f} dB  "
          f"SSIM = {q:.4f}")

# 3) ตัวกรองเกาส์เซียนที่ σ ต่าง ๆ
#    (ksize = (0, 0) ให้ OpenCV คำนวณขนาดจาก σ)
for s in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]:
    out = cv2.GaussianBlur(g, (0, 0), s)
    p, q = cv2.PSNR(f, out), ssim(f, out)
    print(f"เกาส์เซียน σ = {s:3.1f}  PSNR = {p:5.2f} dB  "
          f"SSIM = {q:.4f}")

# 4) ตรวจสมการที่ (3.15) ในบริเวณเรียบ
flat = np.full((512, 512), 128.0)
noisy = flat + np.random.default_rng(1).normal(0, 20, flat.shape)
inner = (slice(20, -20), slice(20, -20))  # ตัดแถบขอบออก
out = cv2.blur(noisy, (3, 3))
print("σ หลังค่าเฉลี่ย 3×3   :", round(out[inner].std(), 2))
out = cv2.GaussianBlur(noisy, (3, 3), 0)
print("σ หลังเกาส์เซียน 3×3 :", round(out[inner].std(), 2))


# ======================================================================
# โปรแกรม 3.4: ตัวกรองมัธยฐานและตัวกรองมัธยฐานแบบปรับตัวได้
# ======================================================================
import numpy as np
import cv2
from skimage import data

def adaptive_median(g, s_max=7):
    """ตัวกรองมัธยฐานแบบปรับตัวได้ ตามขั้นตอนในหัวข้อ 3.4.2"""
    out = g.copy()
    # พิกเซลที่ยังไม่ได้ข้อสรุปจากขั้น A
    undecided = np.ones(g.shape, dtype=bool)
    for k in range(3, s_max + 1, 2):
        se = np.ones((k, k), np.uint8)
        z_min = cv2.erode(g, se)  # ค่าต่ำสุดในหน้าต่าง k × k
        z_max = cv2.dilate(g, se)  # ค่าสูงสุดในหน้าต่าง k × k
        z_med = cv2.medianBlur(g, k)  # มัธยฐานในหน้าต่าง k × k
        # ขั้น A ผ่าน
        level_a = undecided & (z_min < z_med) & (z_med < z_max)
        keep = (z_min < g) & (g < z_max)  # ขั้น B: ไม่ใช่อิมพัลส์
        out[level_a & keep] = g[level_a & keep]  # คงค่าเดิม
        # แทนด้วยมัธยฐาน
        out[level_a & ~keep] = z_med[level_a & ~keep]
        undecided &= ~level_a
    out[undecided] = z_med[undecided]  # ถึง s_max แล้ว ใช้มัธยฐาน
    return out

def add_salt_pepper_noise(f, Ps, Pp, rng):  # เหมือนโปรแกรม 3.1
    g = f.copy()
    u = rng.random(f.shape)
    g[u < Pp] = 0
    g[(u >= Pp) & (u < Pp + Ps)] = 255
    return g

f = data.camera()
def read(name):
    return cv2.imread(name, cv2.IMREAD_GRAYSCALE)

cases = {
    "เกาส์เซียน σ = 20": read("camera_gauss20.png"),
    "เกลือและพริกไทย P = 0.05": read("camera_sp05.png"),
    "เกลือและพริกไทย P = 0.30": add_salt_pepper_noise(
        f, 0.15, 0.15, np.random.default_rng(30))}
for name, g in cases.items():
    print(name)
    filters = {"ไม่กรอง": g,
               "ค่าเฉลี่ย 3×3": cv2.blur(g, (3, 3)),
               "มัธยฐาน 3×3": cv2.medianBlur(g, 3),
               "มัธยฐาน 5×5": cv2.medianBlur(g, 5),
               "มัธยฐานแบบปรับตัว": adaptive_median(g, 7)}
    for fname, out in filters.items():
        print(f"   {fname:17s} PSNR = {cv2.PSNR(f, out):5.2f} dB")

# ส่วนเบี่ยงเบนมาตรฐานของสัญญาณรบกวนเกาส์เซียนหลังกรอง
# ในบริเวณเรียบ (เทียบกับสมการที่ (3.16))
flat = np.full((512, 512), 128.0)
noisy = flat + np.random.default_rng(1).normal(0, 20, flat.shape)
noisy = noisy.astype(np.float32)
inner = (slice(20, -20), slice(20, -20))
for m in [3, 5]:
    s_mean = cv2.blur(noisy, (m, m))[inner].std()
    s_med = cv2.medianBlur(noisy, m)[inner].std()
    print(f"σ หลังกรอง {m}×{m}: ค่าเฉลี่ย = {s_mean:.2f}, "
          f"มัธยฐาน = {s_med:.2f}")


# ======================================================================
# โปรแกรม 3.5: การแพร่แบบแอนไอโซทรอปิกของ Perona–Malik
# ======================================================================
import numpy as np
import cv2
from skimage import data
from skimage.metrics import structural_similarity

def ssim(f, g):  # ตั้งค่าเดียวกับโปรแกรม 3.2
    return structural_similarity(
        f, g, data_range=255, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False)

def anisotropic_diffusion(g, n_iter, K, lam=0.2):
    """การแพร่แบบแอนไอโซทรอปิกของ Perona–Malik ตามสมการที่ (3.30)"""
    I = g.astype(np.float64)
    for _ in range(n_iter):
        P = np.pad(I, 1, mode="edge")  # ทำซ้ำค่าขอบ
        grads = [P[:-2, 1:-1] - I,  # ทิศเหนือ
                 P[2:, 1:-1] - I,  # ทิศใต้
                 P[1:-1, 2:] - I,  # ทิศตะวันออก
                 P[1:-1, :-2] - I]  # ทิศตะวันตก
        # c ตามสมการที่ (3.29) และการปรับค่าตามสมการที่ (3.30)
        I = I + lam * sum(np.exp(-(d / K) ** 2) * d
                          for d in grads)
    return np.clip(np.round(I), 0, 255).astype(np.uint8)

f = data.camera()
# ภาพจากโปรแกรม 3.1
g = cv2.imread("camera_gauss20.png", cv2.IMREAD_GRAYSCALE)
for K in [10, 20, 30, 40]:
    for n in [5, 10, 20, 50]:
        out = anisotropic_diffusion(g, n, K)
        print(f"K = {K:2d}  รอบ = {n:2d}  "
              f"PSNR = {cv2.PSNR(f, out):5.2f} dB  "
              f"SSIM = {ssim(f, out):.4f}")


# ======================================================================
# โปรแกรม 3.6: ตัวกรองแบบสองด้าน
# ======================================================================
import numpy as np
import cv2
from skimage import data
from skimage.metrics import structural_similarity

def ssim(f, g):  # ตั้งค่าเดียวกับโปรแกรม 3.2
    return structural_similarity(
        f, g, data_range=255, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False)

def bilateral(g, radius, sigma_d, sigma_r):
    """ตัวกรองแบบสองด้านตามสมการที่ (3.33)–(3.36)
    หน้าต่างวงกลมรัศมี radius"""
    g = g.astype(np.float64)
    # mode="reflect" ตรงกับ BORDER_REFLECT_101
    P = np.pad(g, radius, mode="reflect")
    M, N = g.shape
    num = np.zeros_like(g)
    W = np.zeros_like(g)
    r = radius
    for t in range(-r, r + 1):
        for s in range(-r, r + 1):
            if s * s + t * t > r * r:  # นอกวงกลม ไม่นำมาคิด
                continue
            # q คือ g(x+s, y+t) ของทุกพิกเซลพร้อมกัน
            q = P[r + t:r + t + M, r + s:r + s + N]
            w_d = np.exp(-(s * s + t * t) / (2 * sigma_d ** 2))
            w_r = np.exp(-(q - g) ** 2 / (2 * sigma_r ** 2))
            w = w_d * w_r
            num += w * q
            W += w
    return num / W

f = data.camera()
# ภาพจากโปรแกรม 3.1
g = cv2.imread("camera_gauss20.png", cv2.IMREAD_GRAYSCALE)

# 1) ตรวจว่าโปรแกรมนี้ให้ผลเท่ากับ OpenCV (d = 7 คือรัศมี 3)
mine = bilateral(g, 3, 2.0, 40.0)
ocv = cv2.bilateralFilter(g.astype(np.float32), 7, 40, 2.0)
diff = float(np.abs(mine - ocv).max())
print("ผลต่างสูงสุดจาก OpenCV:", round(diff, 4))

# 2) ผลของ σ_r ที่ σ_d = 2
#    (d = -1 ให้ OpenCV ใช้รัศมี round(1.5 σ_d))
for sr in [10, 20, 30, 40, 50, 60, 80, 100, 150]:
    out = cv2.bilateralFilter(g, -1, sr, 2)
    print(f"σ_r = {sr:3d}  PSNR = {cv2.PSNR(f, out):5.2f} dB  "
          f"SSIM = {ssim(f, out):.4f}")

# 3) พารามิเตอร์ที่ปรับเรียบอย่างแรง (หัวข้อ 3.6.4)
out = cv2.bilateralFilter(g, 12, 100, 150)
print(f"d = 12, σcolor = 100, σspace = 150  "
      f"PSNR = {cv2.PSNR(f, out):5.2f} dB  "
      f"SSIM = {ssim(f, out):.4f}")

# 4) สัญญาณรบกวนแบบเกลือและพริกไทย
g_sp = cv2.imread("camera_sp05.png", cv2.IMREAD_GRAYSCALE)
for sr in [30, 50, 100]:
    out = cv2.bilateralFilter(g_sp, -1, sr, 2)
    print(f"เกลือและพริกไทย σ_r = {sr:3d}  "
          f"PSNR = {cv2.PSNR(f, out):5.2f} dB")


# ======================================================================
# โปรแกรม 3.7: การทดลองเปรียบเทียบตัวกรอง (ใช้ฟังก์ชันจากโปรแกรม 3.4 และ 3.5)
# ======================================================================
import time
import numpy as np
import cv2
from skimage import data
from skimage.metrics import structural_similarity
# adaptive_median จากโปรแกรม 3.4 และ anisotropic_diffusion
# จากโปรแกรม 3.5: ใน Colab ให้รันเซลล์ทั้งสองก่อน
# หรือบันทึกฟังก์ชันทั้งสองไว้ในไฟล์ prog_lib.py
try:
    from prog_lib import anisotropic_diffusion, adaptive_median
except ImportError:
    pass

def ssim(f, g):
    return structural_similarity(
        f, g, data_range=255, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False)

def gaussian_noise(f, sigma, rng):
    g = np.round(f + rng.normal(0, sigma, f.shape))
    return np.clip(g, 0, 255).astype(np.uint8)

def salt_pepper(f, P, rng):
    g = f.copy(); u = rng.random(f.shape)
    g[u < P / 2] = 0
    g[(u >= P / 2) & (u < P)] = 255
    return g

def gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# ภาพทดสอบ CC0 จาก scikit-image [5] แปลงเป็นระดับเทา
images = {"camera": data.camera(),
          "chelsea": gray(data.chelsea()),
          "coffee": gray(data.coffee())}
noises = [("G10", 10), ("G20", 20), ("G30", 30),
          ("SP05", 0.05), ("SP20", 0.20)]

def filters(g, s):
    """พารามิเตอร์ตามแนวทางในบท ไม่ได้ปรับให้เหมาะกับแต่ละภาพ
       s = σ_η ที่ประมาณได้ (กรณีอิมพัลส์ใช้ s = 20
       เสมือนผู้ใช้ไม่รู้ชนิดของสัญญาณรบกวน)"""
    return {
        "ค่าเฉลี่ย 3×3": lambda: cv2.blur(g, (3, 3)),
        "เกาส์เซียน σ=1": lambda: cv2.GaussianBlur(g, (0, 0), 1.0),
        "มัธยฐาน 3×3": lambda: cv2.medianBlur(g, 3),
        "มัธยฐานปรับตัว": lambda: adaptive_median(g, 7),
        "แอนไอโซทรอปิก":
            lambda: anisotropic_diffusion(g, 10, 1.5 * s),
        "สองด้าน": lambda: cv2.bilateralFilter(g, -1, 2.5 * s, 2),
        "NLM": lambda: cv2.fastNlMeansDenoising(
            g, None, h=s, templateWindowSize=7,
            searchWindowSize=21)}

rows = []
for k, (name, f) in enumerate(images.items()):
    for j, (nname, level) in enumerate(noises):
        # seed คงที่ทุกชุด
        rng = np.random.default_rng(1000 + 10 * k + j)
        if nname[0] == "G":
            g = gaussian_noise(f, level, rng)
        else:
            g = salt_pepper(f, level, rng)
        s = level if nname[0] == "G" else 20
        rows.append((name, nname, "ไม่กรอง",
                     cv2.PSNR(f, g), ssim(f, g)))
        for fname, run in filters(g, s).items():
            out = run()
            rows.append((name, nname, fname,
                         cv2.PSNR(f, out), ssim(f, out)))

# บันทึกผลละเอียดทุกภาพ
with open("experiment_3_7.csv", "w", encoding="utf-8") as fp:
    fp.write("image,noise,filter,psnr,ssim\n")
    for r in rows:
        fp.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.2f},{r[4]:.4f}\n")

fnames = ["ไม่กรอง"] + list(filters(None, 20).keys())
print("ค่าเฉลี่ยของ 3 ภาพ: PSNR (dB) / SSIM")
print(f"{'':16s}" + "".join(f"{n:>14s}" for n, _ in noises))
for fn in fnames:
    line = f"{fn:16s}"
    for nname, _ in noises:
        sel = [r for r in rows if r[1] == nname and r[2] == fn]
        p = np.mean([r[3] for r in sel])
        q = np.mean([r[4] for r in sel])
        line += f"  {p:5.2f}/{q:.3f}"
    print(line)

# เวลาคำนวณกับภาพ camera 512 × 512 (มัธยฐานของ 5 ครั้ง)
g = gaussian_noise(images["camera"], 20, np.random.default_rng(0))
print("\nเวลาคำนวณ (มิลลิวินาที)")
for fname, run in filters(g, 20).items():
    t = []
    for _ in range(5):
        t0 = time.perf_counter()
        run()
        t.append(time.perf_counter() - t0)
    print(f"  {fname:16s} {1000 * np.median(t):8.1f}")

# สัญญาณรบกวนผสม: เกาส์เซียน σ_η = 20
# ร่วมกับเกลือและพริกไทย P = 0.05 (ภาพ camera)
f = images["camera"]
g = gaussian_noise(f, 20, np.random.default_rng(77))
g = salt_pepper(g, 0.05, np.random.default_rng(78))
def bil(x):
    return cv2.bilateralFilter(x, -1, 50, 2)
mixed = {"ไม่กรอง": g,
         "สองด้านอย่างเดียว": bil(g),
         "มัธยฐาน 3×3 อย่างเดียว": cv2.medianBlur(g, 3),
         "มัธยฐาน 3×3 แล้วสองด้าน": bil(cv2.medianBlur(g, 3)),
         "มัธยฐานปรับตัว แล้วสองด้าน": bil(adaptive_median(g, 7))}
print("\nสัญญาณรบกวนผสม")
for n, out in mixed.items():
    print(f"  {n:26s} PSNR = {cv2.PSNR(f, out):5.2f} dB  "
          f"SSIM = {ssim(f, out):.4f}")


# ======================================================================
# โปรแกรม 3.8: การฟื้นฟูภาพเบลอจากการเคลื่อนที่ด้วยตัวกรองผกผันและตัวกรองวีเนอร์
# ======================================================================
import numpy as np
from scipy import ndimage as ndi
from skimage import data, restoration
from skimage.metrics import peak_signal_noise_ratio as psnr

# ภาพต้นฉบับในช่วง [0, 1] และ PSF ของการเคลื่อนที่แนวนอน 9 พิกเซล
f = data.camera().astype(np.float64) / 255
L = 9
psf = np.ones((1, L)) / L                 # สมการที่ (3.37)

def degrade(f, psf, sigma, rng):
    """g = h * f + η ตามสมการที่ (3.4)"""
    g = ndi.convolve(f, psf, mode="wrap")  # สังวัตนาการแบบวน
    return g + rng.normal(0, sigma, f.shape)

rng = np.random.default_rng(264)
cols = ["ภาพเบลอ", "ผกผัน", "K=0.001", "K=0.01", "K=0.1"]
print("σ_η  " + "".join(f"{h:>10s}" for h in cols))
for sigma in [0.0, 0.005, 0.02]:
    g = degrade(f, psf, sigma, rng)
    row = [psnr(f, np.clip(g, 0, 1), data_range=1)]
    for K in [1e-6, 1e-3, 1e-2, 1e-1]:
        # reg = 1 ให้ตรงกับสมการที่ (3.40) โดย K = balance
        fh = restoration.wiener(g, psf, balance=K,
                                reg=np.ones((1, 1)), clip=True)
        row.append(psnr(f, fh, data_range=1))
    print(f"{sigma:5.3f}" + "".join(f"{v:10.2f}" for v in row))


# ======================================================================
# โปรแกรม 3.9: การกำจัดสัญญาณรบกวนเป็นคาบด้วยตัวกรองตัดเฉพาะจุด
# ======================================================================
import numpy as np
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr

# ภาพต้นฉบับในช่วง [0, 1] และสัญญาณรบกวนเป็นคาบ
f = data.camera().astype(np.float64) / 255     # M = N = 512
M, N = f.shape
y, x = np.mgrid[0:M, 0:N]
# ความถี่ u0, v0 (รอบต่อภาพ) และแอมพลิจูด A
u0, v0, A = 40, 30, 0.1
eta = A * np.sin(2 * np.pi * (u0 * x / N + v0 * y / M))
g = f + eta

# สเปกตรัมที่เลื่อนจุดศูนย์กลาง: ความถี่ (0, 0) อยู่ที่ (M/2, N/2)
G = np.fft.fftshift(np.fft.fft2(g))
mag = np.abs(G)
cy, cx = M // 2, N // 2
# ไม่นับย่านความถี่ต่ำรัศมี 10 รอบจุดศูนย์กลาง
mask_dc = (x - cx) ** 2 + (y - cy) ** 2 > 10 ** 2
idx = np.argsort((mag * mask_dc).ravel())[-2:]  # จุดยอด 2 จุด
peaks = [(int(i % N - cx), int(i // N - cy)) for i in idx]
print("ตำแหน่งจุดยอด (u, v) เทียบจุดศูนย์กลาง:", sorted(peaks))

# ตัวกรองตัดเฉพาะจุด (notch reject) รัศมี 3 รอบจุดยอดทั้งสอง
H = np.ones((M, N))
for du, dv in peaks:
    H[(x - (cx + du)) ** 2 + (y - (cy + dv)) ** 2 <= 3 ** 2] = 0
fh = np.real(np.fft.ifft2(np.fft.ifftshift(H * G)))

print(f"PSNR ก่อนกรอง  = {psnr(f, g, data_range=1):.2f} dB")
print(f"PSNR หลังกรอง  = {psnr(f, fh, data_range=1):.2f} dB")
print(f"สัมประสิทธิ์ที่ถูกตัด = {100 * (1 - H.mean()):.3f}%")
