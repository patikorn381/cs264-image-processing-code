"""CS264 การประมวลผลภาพเชิงดิจิทัล: โปรแกรมของบทที่ 8
รันในโฟลเดอร์ว่าง ตามลำดับจากบนลงล่าง (โปรแกรมหลัง ๆ ใช้ฟังก์ชันและไฟล์จากโปรแกรมก่อนหน้า)
"""

# ======================================================================
# โปรแกรม 8.1: การแบ่งส่วนภาพด้วยการขยายบริเวณ
# ======================================================================
import numpy as np
import cv2
from skimage import data

def region_grow(f, seed, T, trace=False):
    """การขยายบริเวณจากจุดเริ่มต้นเดียว (เชื่อมโยงแบบ 4 ทิศ)
    รับพิกเซลข้างเคียงที่ δ = |f(x, y) - ค่าเฉลี่ยของบริเวณ| น้อยที่สุด
    ทีละพิกเซล จนกว่า δ ที่น้อยที่สุดจะเกิน T"""
    H, W = f.shape
    R, total = {seed}, float(f[seed[1], seed[0]])
    def nbrs(x, y):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= x + dx < W and 0 <= y + dy < H:
                yield (x + dx, y + dy)
    cand = set(nbrs(*seed))
    while cand:
        m = total / len(R)
        p = min(cand, key=lambda q: abs(f[q[1], q[0]] - m))
        d = abs(f[p[1], p[0]] - m)
        if trace:
            print(f"ค่าเฉลี่ย {m:6.2f} เลือก (x, y) = {p} "
                  f"ค่า {f[p[1], p[0]]:3.0f} δ = {d:5.2f}"
                  + ("" if d <= T else "  > T หยุด"))
        if d > T:
            break
        R.add(p); total += f[p[1], p[0]]; cand.discard(p)
        cand |= set(nbrs(*p)) - R
    return R, total / len(R)

# 1) ตรวจตัวอย่างที่ 8.1: จุดเริ่มต้น (x, y) = (2, 1), T = 10
f = np.array([[10, 12, 50, 55], [11, 14, 52, 60],
              [13, 40, 46, 58], [15, 18, 44, 90]], float)
R, m = region_grow(f, (2, 1), 10, trace=True)
print(f"บริเวณ {len(R)} พิกเซล ค่าเฉลี่ย {m:.2f}")
R15, m15 = region_grow(f, (2, 1), 15)
print(f"T = 15: บริเวณ {len(R15)} พิกเซล ค่าเฉลี่ย {m15:.2f}")

# 2) ภาพเหรียญ (บริเวณ 100 x 100) จุดเริ่มต้นอยู่บนเหรียญ
#    เทียบเกณฑ์ค่าเฉลี่ยของบริเวณ กับ cv2.floodFill
#    ซึ่งเทียบกับค่าของจุดเริ่มต้น (FLOODFILL_FIXED_RANGE)
g = data.coins()[10:110, 10:110]
seed = (50, 40)
flags = (4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
         | (255 << 8))
print(f"ค่าที่จุดเริ่มต้น = {g[seed[1], seed[0]]}")
print("  T  ค่าเฉลี่ยบริเวณ (พิกเซล)  floodFill (พิกเซล)")
for T in [20, 30, 40, 50, 60]:
    R, _ = region_grow(g.astype(float), seed, T)
    mask = np.zeros((g.shape[0] + 2, g.shape[1] + 2), np.uint8)
    n = cv2.floodFill(g.copy(), mask, seed, 255, T, T, flags)[0]
    print(f"{T:3d}{len(R):16d}{n:22d}")


# ======================================================================
# โปรแกรม 8.2: K-Means บนภาพ 4 × 4 ของตัวอย่างที่ 8.2
# ======================================================================
import numpy as np

def kmeans_1d(z, labels, K, max_iter=20):
    """K-Means แบบลอยด์บนค่าความเข้ม 1 มิติ ตามขั้นตอนในหัวข้อ 8.2.1"""
    for t in range(1, max_iter + 1):
        # ขั้นที่ 2: ค่าเฉลี่ยของกลุ่ม
        m = np.array([z[labels == k].mean() for k in range(K)])
        # สมการที่ (8.2)
        J = sum(((z[labels == k] - m[k]) ** 2).sum()
                for k in range(K))
        # ขั้นที่ 3: ย้ายไปกลุ่มที่ใกล้สุด
        new = np.argmin(np.abs(z[:, None] - m[None, :]), axis=1)
        moved = int((new != labels).sum())
        print(f"รอบที่ {t}: ค่าเฉลี่ย = {np.round(m, 2)}")
        print(f"    J = {J:,.2f}, ย้ายกลุ่ม {moved} พิกเซล")
        if moved == 0:                    # ขั้นที่ 4: ลู่เข้า
            return new, m
        labels = new
    return labels, m

# ภาพนำเข้า 4 x 4 ของตัวอย่างที่ 8.2 (P1–P16 เรียงตามแถว)
f = np.array([[255, 91, 10, 23], [77, 1, 65, 250],
              [21, 82, 5, 98], [89, 230, 6, 96]], float)
z = f.ravel()
# แบ่งเริ่มต้นเกือบเท่ากัน 6/5/5
init = np.array([0] * 6 + [1] * 5 + [2] * 5)
labels, m = kmeans_1d(z, init, K=3)
print("ผลการจัดกลุ่ม:\n",
      np.array(list("ABC"))[labels].reshape(4, 4))

# ภาพผลลัพธ์ g(x,y): แทนพิกเซลแต่ละตัวด้วยค่าเฉลี่ยของกลุ่ม
# (การควอนไทซ์ด้วย K-Means)
g = np.round(m[labels]).reshape(4, 4).astype(int)
print("g(x,y) =\n", g)

# ค่าเริ่มต้นที่ไม่ดี: จุดศูนย์กลางเริ่มต้น (230, 250, 255)
# ซึ่งอยู่ใกล้กันทั้งหมด
print("--- เริ่มจากจุดศูนย์กลาง (230, 250, 255) ---")
c0 = np.array([230, 250, 255])
init2 = np.argmin(np.abs(z[:, None] - c0[None, :]), axis=1)
labels2, m2 = kmeans_1d(z, init2, K=3)


# ======================================================================
# โปรแกรม 8.3: การแบ่งส่วนภาพสีด้วย cv2.kmeans และการเปรียบเทียบกับวิธีของ Otsu ด้วย IoU และ Dice
# ======================================================================
import numpy as np
import cv2
from skimage import data
from skimage.filters import threshold_otsu

# ภาพสี 400 x 600 (สาธารณสมบัติ)
f = cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR)
# เวกเตอร์คุณลักษณะ = (B, G, R) ของแต่ละพิกเซล
Z = f.reshape(-1, 3).astype(np.float32)
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100, 0.2)

# 1) ผลของจำนวนกลุ่ม K: J ต่อพิกเซล (compactness / จำนวนพิกเซล)
print(" K   J/พิกเซล   ลดลงจาก K-1 (%)")
prev = None
for K in range(1, 9):
    cv2.setRNGSeed(0)
    J, lab, cen = cv2.kmeans(Z, K, None, crit, 10,
                             cv2.KMEANS_PP_CENTERS)
    J /= len(Z)
    drop = ("" if prev is None
            else f"{100 * (prev - J) / prev:15.1f}")
    print(f"{K:2d}{J:11.1f}" + drop)
    prev = J
    if K in (2, 3, 4):      # บันทึกภาพผลลัพธ์ g = ค่ากลางของกลุ่ม
        g = cen[lab.ravel()].reshape(f.shape).astype(np.uint8)
        cv2.imwrite(f"km_K{K}.png", g)

# 2) ความไวต่อค่าเริ่มต้น (K = 8): สุ่มจุดศูนย์กลางเริ่มต้น 20 ครั้ง
#    ครั้งละ 1 รอบ เทียบกับ k-means++ 10 รอบ
Js = []
for seed in range(20):
    cv2.setRNGSeed(seed)
    J, _, _ = cv2.kmeans(Z, 8, None, crit, 1,
                         cv2.KMEANS_RANDOM_CENTERS)
    Js.append(J / len(Z))
Js = np.array(Js)
cv2.setRNGSeed(0)
Jpp, _, _ = cv2.kmeans(Z, 8, None, crit, 10,
                       cv2.KMEANS_PP_CENTERS)
print(f"K=8 สุ่มค่าเริ่มต้น 20 ครั้ง: J/พิกเซล ต่ำสุด {Js.min():.1f} "
      f"สูงสุด {Js.max():.1f} เฉลี่ย {Js.mean():.1f}")
print(f"K=8 k-means++ 10 รอบ: J/พิกเซล {Jpp / len(Z):.1f}")

# 3) K-Means สองกลุ่มบนภาพระดับเทา เทียบกับค่าขีดแบ่งของ Otsu
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
z = gray.reshape(-1, 1).astype(np.float32)
cv2.setRNGSeed(0)
_, lab, cen = cv2.kmeans(z, 2, None, crit, 10,
                         cv2.KMEANS_PP_CENTERS)
c = np.sort(cen.ravel())
T_km = c.mean()       # ขอบเขตการตัดสินใจ = กึ่งกลางของจุดศูนย์กลาง
T_otsu = threshold_otsu(gray)
agree = np.mean((gray > T_km) == (gray > T_otsu))
print(f"K-Means K=2: จุดศูนย์กลาง = {np.round(c, 1)}, "
      f"ขอบเขต = {T_km:.1f}")
print(f"Otsu: T = {T_otsu} | พิกเซลที่ตรงกัน {100 * agree:.2f}%")
# 4) บริเวณสว่างของ K-Means บนสี (K = 2) เทียบกับมาส์กของ Otsu
#    ด้วย IoU และ Dice (หัวข้อ 8.1.6)
cv2.setRNGSeed(0)
_, lab, cen = cv2.kmeans(Z, 2, None, crit, 10,
                         cv2.KMEANS_PP_CENTERS)
bright = int(np.argmax(cen.sum(axis=1)))     # กลุ่มที่สว่างกว่า
P = (lab.ravel() == bright).reshape(gray.shape)
G = gray > T_otsu
inter = np.logical_and(P, G).sum()
union = np.logical_or(P, G).sum()
print(f"K-Means สี เทียบ Otsu: |P| = {P.sum()}, |G| = {G.sum()}, "
      f"|P∩G| = {inter}")
print(f"IoU = {inter / union:.4f}, "
      f"Dice = {2 * inter / (P.sum() + G.sum()):.4f}")


# ======================================================================
# โปรแกรม 8.4: เอนโทรปี รหัสฮัฟฟ์แมน และการเข้ารหัสความยาวช่วงของตัวอย่างที่ 8.3
# ======================================================================
import heapq
import numpy as np

def huffman(prob):
    """สร้างรหัสฮัฟฟ์แมน: รวมสองสัญลักษณ์ที่มีความน่าจะเป็นต่ำสุด
    ซ้ำจนเหลือหนึ่งโหนด"""
    heap = [(p, i, {s: ""})
            for i, (s, p) in enumerate(prob.items())]
    heapq.heapify(heap)
    n = len(heap)
    while len(heap) > 1:
        p1, _, c1 = heapq.heappop(heap)   # ต่ำสุด -> เติมบิต 1
        p0, _, c0 = heapq.heappop(heap)   # ต่ำเป็นอันดับสอง -> บิต 0
        merged = ({s: "0" + c for s, c in c0.items()}
                  | {s: "1" + c for s, c in c1.items()})
        heapq.heappush(heap, (p0 + p1, n, merged)); n += 1
    return heap[0][2]

def rle(seq):
    """การเข้ารหัสความยาวช่วง: เก็บคู่ (ค่า, จำนวนที่ซ้ำติดกัน)"""
    out, run = [], 1
    for a, b in zip(seq[:-1], seq[1:]):
        if a == b: run += 1
        else: out.append((int(a), run)); run = 1
    out.append((int(seq[-1]), run))
    return out

f = np.array([[50, 50, 50, 50], [50, 50, 50, 100],
              [100, 100, 100, 150], [150, 150, 200, 200]])
vals, cnt = np.unique(f, return_counts=True)
p = cnt / f.size
H = -(p * np.log2(p)).sum()        # สมการที่ (8.6): เอนโทรปี
code = huffman(dict(zip(vals.tolist(), p.tolist())))
# สมการที่ (8.7): ความยาวรหัสเฉลี่ย
Lavg = sum(p[i] * len(code[v]) for i, v in enumerate(vals))
for v, c, pi in zip(vals, cnt, p):
    print(f"ระดับเทา {v:3d}: จำนวน {c:2d}, p = {pi:.4f}, "
          f"รหัส = {code[v]}")
print(f"เอนโทรปี H = {H:.4f} บิต/พิกเซล")
print(f"ความยาวเฉลี่ย L = {Lavg:.4f} บิต/พิกเซล")
print(f"CR เทียบรหัส 8 บิต = {8 / Lavg:.2f}, "
      f"เทียบรหัสยาวคงที่ 2 บิต = {2 / Lavg:.3f}")
bits = "".join(code[v] for v in f.ravel())
print(f"บิตสตรีม ({len(bits)} บิต): {bits}")
r = rle(f.ravel())
print(f"RLE ตามแนวแถว: {r}")
print(f"  -> {2 * len(r)} ไบต์ เทียบกับ {f.size} ไบต์, "
      f"CR = {f.size / (2 * len(r)):.1f}")


# ======================================================================
# โปรแกรม 8.5: การเข้ารหัสและถอดรหัส JPEG บนบล็อก 8 × 8
# ======================================================================
import numpy as np
import cv2

# บล็อกภาพ 8 x 8 ของตัวอย่างที่ 8.4 (ช่อง Y)
f = np.array([[52, 55, 61, 66, 70, 61, 64, 73],
              [63, 59, 55, 90, 109, 85, 69, 72],
              [62, 59, 68, 113, 144, 104, 66, 73],
              [63, 58, 71, 122, 154, 106, 70, 69],
              [67, 61, 68, 104, 126, 88, 68, 70],
              [79, 65, 60, 70, 77, 68, 58, 75],
              [85, 71, 64, 59, 55, 61, 65, 83],
              [87, 79, 69, 68, 65, 76, 78, 94]], float)
# ตารางการควอนไทซ์ความสว่างมาตรฐาน [18]
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]], float)

# เมทริกซ์ DCT: C[u, x] = α(u) cos((2x+1)uπ/16)
# ตามสมการที่ (8.10)–(8.11)
N = 8
a = np.array([np.sqrt(1 / N)] + [np.sqrt(2 / N)] * (N - 1))
x, u = np.arange(N)[None, :], np.arange(N)[:, None]
C = a[:, None] * np.cos((2 * x + 1) * u * np.pi / (2 * N))

s = f - 128             # สมการที่ (8.8): ปรับระดับสัญญาณ
F = C @ s @ C.T         # สมการที่ (8.11): DCT สองมิติแบบเมทริกซ์
E = (F**2).sum()
print("ตรงกับ cv2.dct?", np.allclose(F, cv2.dct(s)))
print(f"F(0,0) = {F[0, 0]:.2f}, "
      f"พลังงานใน F(0,0) = {100 * F[0, 0]**2 / E:.1f}%")
print(f"พลังงานใน 3 x 3 มุมบนซ้าย = "
      f"{100 * (F[:3, :3]**2).sum() / E:.1f}%")
print("พลังงานเท่ากันทั้งสองโดเมน?", np.isclose((s**2).sum(), E))

Fq = np.round(F / Q).astype(int)   # สมการที่ (8.12): การควอนไทซ์

# การสแกนแบบซิกแซก: เรียงตามเส้นทแยง u+v และสลับทิศทุกเส้น
def zkey(p):
    d = p[0] + p[1]
    return (d, p[1] if d % 2 == 0 else p[0])
order = sorted(((r, c) for r in range(N) for c in range(N)),
               key=zkey)
zz = [int(Fq[r, c]) for r, c in order]
last = max(i for i, v in enumerate(zz) if v != 0)
print("ลำดับซิกแซก:")
print(zz[:13])
print(zz[13:last + 1], "+ EOB")
print(f"เก็บ {last + 1} จาก 64 ค่า "
      f"(ไม่เป็นศูนย์ {np.count_nonzero(Fq)} ค่า)")

# การถอดรหัส: คูณกลับ -> IDCT -> บวก 128 -> ปัดเศษและตัดค่า
f_hat = np.clip(np.round(C.T @ (Fq * Q) @ C + 128), 0, 255)
e = f - f_hat
mae, mse = np.abs(e).mean(), (e**2).mean()
print("ภาพที่ถอดรหัส:\n", f_hat.astype(int))
print(f"MAE = {mae:.3f}, MSE = {mse:.4f}, "
      f"PSNR = {10 * np.log10(255**2 / mse):.2f} dB")
print(f"|e| สูงสุด = {np.abs(e).max():.0f}")


# ======================================================================
# โปรแกรม 8.6: ผลของระดับคุณภาพ JPEG ต่อขนาดไฟล์และคุณภาพของภาพ
# ======================================================================
import numpy as np
import cv2
from skimage import data
from skimage.metrics import structural_similarity as ssim

# ภาพสี 300 x 451 (สาธารณสมบัติ)
f = cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2BGR)
raw = f.size            # ขนาดข้อมูลดิบ 300 x 451 x 3 ไบต์
npx = f.shape[0] * f.shape[1]

def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)

def gray(a): return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

# แบบไม่สูญเสีย
ok, png = cv2.imencode(".png", f,
                       [cv2.IMWRITE_PNG_COMPRESSION, 9])
same = np.array_equal(cv2.imdecode(png, cv2.IMREAD_COLOR), f)
print(f"ข้อมูลดิบ {raw:,} ไบต์ | PNG {png.size:,} ไบต์, "
      f"CR = {raw / png.size:.2f}")
print(f"PNG {8 * png.size / npx:.2f} บิต/พิกเซล, "
      f"ถอดรหัสได้ภาพเดิม: {same}")
print(f"{'คุณภาพ':>6s}{'ขนาด (ไบต์)':>13s}{'CR':>8s}"
      f"{'บิต/พิกเซล':>12s}{'PSNR (dB)':>11s}{'SSIM':>8s}")
for q in [95, 75, 50, 25, 10, 5, 1]:
    ok, buf = cv2.imencode(".jpg", f,
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    g = cv2.imdecode(buf, cv2.IMREAD_COLOR)    # ภาพที่ถอดรหัส
    bpp = 8 * buf.size / npx
    s = ssim(gray(f), gray(g), data_range=255)
    print(f"{q:6d}{buf.size:13,d}{raw / buf.size:8.1f}{bpp:12.3f}"
          f"{psnr(f, g):11.2f}{s:8.4f}")
    # เก็บผลเป็น PNG เพื่อแสดงในรูปที่ 8.6
    cv2.imwrite(f"jpg_q{q}.png", g)
