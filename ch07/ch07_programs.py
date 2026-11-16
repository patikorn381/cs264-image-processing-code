"""CS264 การประมวลผลภาพเชิงดิจิทัล: โปรแกรมของบทที่ 7
รันในโฟลเดอร์ว่าง ตามลำดับจากบนลงล่าง (โปรแกรมหลัง ๆ ใช้ฟังก์ชันและไฟล์จากโปรแกรมก่อนหน้า)
"""

# ======================================================================
# โปรแกรม 7.1: การจับคู่แผ่นแบบด้วย SSD สหสัมพันธ์ และ NCC
# ======================================================================
import numpy as np
import cv2
from skimage import data

def ssd_map(f, w):
    """สมการที่ (7.1): ผลรวมกำลังสองของผลต่าง
    ณ ทุกตำแหน่งที่แผ่นแบบอยู่ภายในภาพ"""
    M, N = f.shape
    m, n = w.shape
    R = np.zeros((M - m + 1, N - n + 1))
    for y in range(M - m + 1):
        for x in range(N - n + 1):
            R[y, x] = np.sum((w - f[y:y + m, x:x + n]) ** 2)
    return R

# 1) ตัวอย่างที่ 7.1: ภาพสองระดับ 5 x 5 กับแผ่นแบบรูปตัว T ขนาด 3 x 3
f = np.array([[0, 0, 0, 0, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1]], np.float32)
w = np.array([[1, 1, 1],
              [0, 1, 0],
              [0, 1, 0]], np.float32)
R = ssd_map(f, w)
print("ขนาดของ R (แถว x คอลัมน์):", R.shape)
print("R =\n", R.astype(int))
R_cv = cv2.matchTemplate(f, w, cv2.TM_SQDIFF)
mn, _, loc, _ = cv2.minMaxLoc(R_cv)
print(f"ตรงกับ cv2.TM_SQDIFF: {np.allclose(R, R_cv)}")
print(f"ค่าต่ำสุด {mn:.1f} ที่ (x, y) = {loc}")

# 2) ภาพจริง: ตัดแผ่นแบบจากภาพ camera แล้วเปลี่ยนความสว่าง
#    หรือความเปรียบต่างของภาพค้นหา
img = data.camera().astype(np.float32)
# บริเวณศีรษะของช่างภาพ ขนาด 60 x 60
x0, y0, n, m = 250, 90, 60, 60
w = img[y0:y0 + m, x0:x0 + n].copy()
print(f"ตำแหน่งที่ถูกต้อง (x, y) = {(x0, y0)}")
print(f"{'ภาพค้นหา':8s}{'TM_SQDIFF':>11s}{'TM_CCORR':>11s}"
      f"{'TM_CCOEFF_NORMED':>17s}{'ค่าสูงสุด NCC':>14s}")
cases = [("f", img), ("f + 60", img + 60), ("0.5f", 0.5 * img),
         ("1.5f", 1.5 * img)]
for name, g in cases:
    g = np.clip(g, 0, 255)      # ภาพ 8 บิตเก็บค่าได้ไม่เกิน 255
    def best(method, k):
        return cv2.minMaxLoc(cv2.matchTemplate(g, w, method))[k]
    r_sq = best(cv2.TM_SQDIFF, 2)  # SSD: ตำแหน่งค่าต่ำสุด
    r_cc = best(cv2.TM_CCORR, 3)   # สหสัมพันธ์: ตำแหน่งค่าสูงสุด
    _, peak, _, r_nc = cv2.minMaxLoc(
        cv2.matchTemplate(g, w, cv2.TM_CCOEFF_NORMED))
    print(f"{name:8s}{str(r_sq):>11s}{str(r_cc):>11s}"
          f"{str(r_nc):>17s}{peak:14.3f}")


# ======================================================================
# โปรแกรม 7.2: โมเมนต์เชิงพื้นที่ โมเมนต์ศูนย์กลาง มุมของแกนหลัก และโมเมนต์ไม่แปรเปลี่ยนของฮู
# ======================================================================
import numpy as np
import cv2

def raw_moment(f, p, q):
    """สมการที่ (7.5): M_pq = ΣΣ x^p y^q f(x, y)
    โดย x คือคอลัมน์ y คือแถว เริ่มที่ 0"""
    y, x = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    return np.sum(x ** p * y ** q * f)

def central_moment(f, p, q):
    """สมการที่ (7.7): μ_pq = ΣΣ (x - x̄)^p (y - ȳ)^q f(x, y)"""
    m00 = raw_moment(f, 0, 0)
    xb = raw_moment(f, 1, 0) / m00
    yb = raw_moment(f, 0, 1) / m00
    y, x = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    return np.sum((x - xb) ** p * (y - yb) ** q * f)

def eta(f, p, q):
    """สมการที่ (7.20): โมเมนต์ศูนย์กลางนอร์มัลไลซ์"""
    g = (p + q) / 2 + 1
    return central_moment(f, p, q) / central_moment(f, 0, 0) ** g

def hu(f):
    """สมการที่ (7.22)–(7.28): โมเมนต์ไม่แปรเปลี่ยนของฮู 7 ค่า"""
    n20, n02, n11 = eta(f, 2, 0), eta(f, 0, 2), eta(f, 1, 1)
    n30, n03 = eta(f, 3, 0), eta(f, 0, 3)
    n21, n12 = eta(f, 2, 1), eta(f, 1, 2)
    a, b = n30 + n12, n21 + n03          # พจน์ที่ใช้ซ้ำ
    c, d = n30 - 3 * n12, 3 * n21 - n03
    return np.array([
        n20 + n02,
        (n20 - n02) ** 2 + 4 * n11 ** 2,
        c ** 2 + d ** 2,
        a ** 2 + b ** 2,
        c * a * (a ** 2 - 3 * b ** 2)
        + d * b * (3 * a ** 2 - b ** 2),
        (n20 - n02) * (a ** 2 - b ** 2) + 4 * n11 * a * b,
        d * a * (a ** 2 - 3 * b ** 2)
        - c * b * (3 * a ** 2 - b ** 2)])

def orientation(f):
    """สมการที่ (7.18): มุมของแกนหลัก (องศา)
    วัดจากแกน x ตามเข็มนาฬิกาบนจอ"""
    mu20, mu02 = central_moment(f, 2, 0), central_moment(f, 0, 2)
    mu11 = central_moment(f, 1, 1)
    return 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))

# ภาพในตัวอย่างที่ 7.3–7.5
f = np.array([[1, 1, 0],
              [0, 1, 0],
              [1, 0, 1]], np.float64)
np.set_printoptions(formatter={"float": lambda v: f"{v: .4e}"},
                    linewidth=60)
# 1) โมเมนต์เชิงพื้นที่และจุดศูนย์กลางมวล (ตัวอย่างที่ 7.3)
for p, q in [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]:
    print(f"M{p}{q} = {raw_moment(f, p, q):g}", end="  ")
m00 = raw_moment(f, 0, 0)
print(f"\nจุดศูนย์กลางมวล = ({raw_moment(f, 1, 0) / m00:g}, "
      f"{raw_moment(f, 0, 1) / m00:g})")

# 2) เลื่อนภาพไป 1 พิกเซลทั้งสองแกน (เท่ากับนับพิกัดเริ่มที่ 1)
g = np.zeros((4, 4)); g[1:, 1:] = f
print("หลังเลื่อน: M10 =", raw_moment(g, 1, 0),
      "M20 =", raw_moment(g, 2, 0))
print("  μ20 =", round(central_moment(g, 2, 0), 4),
      "เท่าเดิม", round(central_moment(f, 2, 0), 4))

# 3) โมเมนต์ศูนย์กลาง อันดับ 2 และ 3 (ตัวอย่างที่ 7.4–7.5)
#    และมุมของแกนหลัก
pq = [(2, 0), (0, 2), (1, 1), (3, 0), (0, 3), (2, 1), (1, 2)]
for k, (p, q) in enumerate(pq):
    print(f"μ{p}{q} = {central_moment(f, p, q):.4f}",
          end="\n" if k in (2, 6) else "  ")
mu20, mu02 = central_moment(f, 2, 0), central_moment(f, 0, 2)
mu11 = central_moment(f, 1, 1)
th_arctan = 0.5 * np.degrees(np.arctan(2 * mu11 / (mu20 - mu02)))
print(f"θ (atan2) = {orientation(f):.2f} องศา | "
      f"θ (arctan) = {th_arctan:.2f} องศา")
def I(t):
    s, c = np.sin(t), np.cos(t)
    return mu20 * s ** 2 - 2 * mu11 * s * c + mu02 * c ** 2
print("โมเมนต์ความเฉื่อยรอบเส้นที่ "
      f"60.48° = {I(np.radians(60.48)):.4f}, "
      f"ที่ -29.52° = {I(np.radians(-29.52)):.4f}")

# 4) โมเมนต์ไม่แปรเปลี่ยนของฮู เทียบกับ OpenCV
m = cv2.moments(f.astype(np.uint8), binaryImage=True)
print("cv2.moments: m10 =", m["m10"],
      "mu20 =", round(m["mu20"], 4),
      "nu20 =", round(m["nu20"], 4))
print("Hu (สมการ):\n", hu(f))
print("Hu (OpenCV):\n", cv2.HuMoments(m).ravel())

# 5) การสะท้อนซ้าย-ขวา: ϕ1–ϕ6 คงเดิม ϕ7 เปลี่ยนเครื่องหมาย
print("Hu ภาพสะท้อน:\n", hu(f[:, ::-1]))

# คืนรูปแบบการพิมพ์ของ NumPy เป็นค่าปริยาย
# ไม่ให้กระทบโปรแกรมอื่นในสมุดบันทึกเดียวกัน
np.set_printoptions(formatter=None, linewidth=75)


# ======================================================================
# โปรแกรม 7.3: การทดลองจำแนกรูปทรงที่หมุนและสเกลด้วยการจับคู่แผ่นแบบและโมเมนต์ของฮู
# ======================================================================
import numpy as np
import cv2

# รูปทรงต้นแบบ 5 ชนิด กำหนดเป็นจุดยอดของรูปหลายเหลี่ยม
# ในหน่วยปกติ (ความยาวประมาณ 1)
SHAPES = {
    "เครื่องบิน": [(0, -.5), (.06, -.38), (.06, -.08), (.5, .1),
                   (.5, .18), (.06, .08), (.05, .32), (.2, .42),
                   (.2, .48), (0, .44), (-.2, .48), (-.2, .42),
                   (-.05, .32), (-.06, .08), (-.5, .18),
                   (-.5, .1), (-.06, -.08), (-.06, -.38)],
    "สามเหลี่ยม": [(-.5, .3), (.5, .3), (-.1, -.4)],
    "ตัว L": [(-.25, -.5), (-.05, -.5), (-.05, .3), (.35, .3),
              (.35, .5), (-.25, .5)],
    "ตัว L สะท้อน": [(.25, -.5), (.05, -.5), (.05, .3), (-.35, .3),
                     (-.35, .5), (.25, .5)],
    "สี่เหลี่ยมผืนผ้า": [(-.5, -.25), (.5, -.25), (.5, .25),
                         (-.5, .25)],
}

def render(name, size, deg=0.0, cx=200.0, cy=200.0, canvas=400):
    """วาดรูปทรงยาว size พิกเซล หมุน deg องศา
    ที่จุดศูนย์กลาง (cx, cy) เป็นภาพสองระดับ"""
    P = np.array(SHAPES[name]) * size
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    Rm = np.array([[c, -s], [s, c]])
    P = P @ Rm.T + (cx, cy)
    f = np.zeros((canvas, canvas), np.uint8)
    # พิกัดละเอียด 1/16 พิกเซล
    pts = np.round(P * 16).astype(np.int32)
    cv2.fillPoly(f, [pts], 1, shift=4)
    return f

def log_hu(f, eps=0.0):
    """สมการที่ (7.30): h_i = -sign(ϕ_i) log10|ϕ_i|
    ถ้า eps > 0 ให้ถือว่า |ϕ_i| < eps เท่ากับ eps และไม่มีเครื่องหมาย"""
    phi = cv2.HuMoments(cv2.moments(f, binaryImage=True)).ravel()
    if eps == 0:
        return -np.sign(phi) * np.log10(np.abs(phi) + 1e-30)
    sgn = np.where(np.abs(phi) < eps, 0, np.sign(phi))
    return -sgn * np.log10(np.maximum(np.abs(phi), eps))

def classify(f, refs, eps):
    """เพื่อนบ้านใกล้สุด: เลือกต้นแบบที่ระยะยุคลิดของ log Hu
    น้อยที่สุด"""
    h = log_hu(f, eps)
    return int(np.argmin([np.linalg.norm(h - r) for r in refs]))

names = list(SHAPES)
# ต้นแบบ: ยาว 200 พิกเซล ไม่หมุน
protos = [render(k, 200) for k in names]
tmpl = []
for p in protos:           # แผ่นแบบ: ตัดกรอบรอบวัตถุต้นแบบ
    x, y, w, h = cv2.boundingRect(p)
    tmpl.append(p[y - 2:y + h + 2, x - 2:x + w + 2]
                .astype(np.float32))
refs = {e: [log_hu(p, e) for p in protos]
        for e in [0.0, 1e-7, 1e-5, 1e-4]}

def tm_classify(f):
    g = f.astype(np.float32)
    s = [cv2.minMaxLoc(
             cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED))[1]
         for t in tmpl]
    return int(np.argmax(s))

# 0) ตรวจสอบเบื้องต้น: ถ้าไม่หมุนและขนาดเท่าต้นแบบ
#    การจับคู่แผ่นแบบต้องถูกทุกครั้ง
rng = np.random.default_rng(7)
ok = [tm_classify(render(k, 200, 0, *rng.uniform(150, 250, 2)))
      == i for i, k in enumerate(names)]
print("จับคู่แผ่นแบบ เมื่อไม่หมุนและขนาดเท่าต้นแบบ: ถูก",
      sum(ok), "จาก", len(ok))

# 1) ทดสอบ: ทุกรูปทรง 20 ภาพต่อขนาด หมุนและเลื่อนแบบสุ่ม
print(f"{'ยาว(px)':>8s}{'Template':>10s}{'Hu':>8s}"
      f"{'Hu ε=1e-5':>11s}{'matchShapes':>13s}"
      f"{'ϕ1 คลาด(%)':>12s}")
sens = {e: [] for e in refs}
for size in [200, 100, 50, 25, 12]:
    ok = np.zeros(4); n = 0; err = []; ok_e = {e: 0 for e in refs}
    for i, k in enumerate(names):
        m_ref = cv2.moments(protos[i], True)
        phi1_ref = cv2.HuMoments(m_ref)[0, 0]
        for _ in range(20):
            f = render(k, size, rng.uniform(0, 360),
                       *rng.uniform(150, 250, 2))
            d_ms = [cv2.matchShapes(f, p,
                                    cv2.CONTOURS_MATCH_I2, 0)
                    for p in protos]
            ok += [tm_classify(f) == i,
                   classify(f, refs[0.0], 0.0) == i,
                   classify(f, refs[1e-5], 1e-5) == i,
                   int(np.argmin(d_ms)) == i]
            for e in refs:
                ok_e[e] += classify(f, refs[e], e) == i
            phi1 = cv2.HuMoments(cv2.moments(f, True))[0, 0]
            err.append(abs(phi1 - phi1_ref) / phi1_ref * 100)
            n += 1
    for e in refs:
        sens[e].append(100 * ok_e[e] / n)
    a = 100 * ok / n
    print(f"{size:8d}{a[0]:9.0f}%{a[1]:7.0f}%{a[2]:10.0f}%"
          f"{a[3]:12.0f}%{np.mean(err):12.2f}")
print("ความไวต่อ ε (ความแม่นยำที่ความยาว 200, 100, 50, 25, 12"
      " พิกเซล):")
for e, v in sens.items():
    print(f"  ε = {e:g}:", [f"{x:.0f}%" for x in v])
