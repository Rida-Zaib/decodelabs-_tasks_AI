import cv2, numpy as np, math
from PIL import Image, ImageDraw, ImageFont
rng = np.random.default_rng(7)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# 1. text along an arc (curved writing)
def arc():
    W, H = 900, 600
    im = Image.new("RGB", (W, H), (250, 245, 230)); f = ImageFont.truetype(FONT, 54)
    text, cx, cy, R = "CURVED TEXT ON AN ARC", W//2, 520, 330
    span = math.radians(120); a0 = -math.pi/2 - span/2
    for i, ch in enumerate(text):
        a = a0 + span * i / (len(text)-1)
        tile = Image.new("RGBA", (80, 80), (0,0,0,0)); ImageDraw.Draw(tile).text((20, 10), ch, font=f, fill=(30,30,30,255))
        tile = tile.rotate(-math.degrees(a + math.pi/2), resample=Image.BICUBIC)
        im.paste(tile, (int(cx + R*math.cos(a) - 40), int(cy + R*math.sin(a) - 40)), tile)
    im.save("samples/1_curved_arc.png")

# 2. wavy banner
def wavy():
    im = np.full((300, 900, 3), 255, np.uint8)
    cv2.putText(im, "Wavy Banner Text 2026", (40, 170), cv2.FONT_HERSHEY_DUPLEX, 1.8, (40, 40, 160), 3, cv2.LINE_AA)
    h, w = im.shape[:2]; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = cv2.remap(im, xx, yy + 22*np.sin(xx/70), None, cv2.INTER_CUBIC, borderValue=(255,255,255))
    cv2.imwrite("samples/2_wavy_banner.png", out)

# 3. low light / uneven shadow
def dim():
    im = np.full((350, 900, 3), 225, np.uint8)
    for i, t in enumerate(["Low light document scan", "Meeting at 10:30 in Room B-204"]):
        cv2.putText(im, t, (40, 120 + i*100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (150,150,150), 3, cv2.LINE_AA)
    grad = np.tile(np.linspace(0.35, 1.0, 900, dtype=np.float32), (350, 1))[..., None]
    cv2.imwrite("samples/3_low_light_shadow.png", np.clip(im*grad, 0, 255).astype(np.uint8))

# 4. blurry + salt-and-pepper noise
def noisy():
    im = np.full((350, 900, 3), 255, np.uint8)
    for i, t in enumerate(["Blurry noisy photo text", "Order #A-48291  Qty: 12"]):
        cv2.putText(im, t, (40, 120 + i*100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20,20,20), 3, cv2.LINE_AA)
    im = cv2.GaussianBlur(im, (7, 7), 0)
    m = rng.random(im.shape[:2]); im[m < 0.03] = 0; im[m > 0.97] = 255
    cv2.imwrite("samples/4_blur_noise.png", im)

# 5. handwriting-like cursive with jitter on lined paper
def hand():
    im = np.full((420, 900, 3), (245, 248, 252), np.uint8)
    for y in range(110, 420, 90): cv2.line(im, (0, y), (900, y), (220, 200, 180), 2)
    for i, t in enumerate(["Hello my name is Vishal", "This is handwritten text", "OCR struggles with cursive"]):
        x = 40
        for ch in t:
            layer = np.full((120, 90, 3), 255, np.uint8)
            cv2.putText(layer, ch, (10, 80), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.7, (120, 40, 20), int(rng.integers(2, 4)), cv2.LINE_AA)
            M = cv2.getRotationMatrix2D((45, 60), float(rng.normal(0, 6)), 1)
            layer = cv2.warpAffine(layer, M, (90, 120), borderValue=(255,255,255))
            y0 = 30 + i*90 + int(rng.integers(-4, 5)); roi = im[y0:y0+120, x:x+90]
            if roi.shape[:2] == (120, 90): im[y0:y0+120, x:x+90] = np.minimum(roi, layer)
            x += 28 if ch != " " else 18
    cv2.imwrite("samples/5_handwriting_cursive.png", im)

# 6. white text on dark colour, rotated 15 degrees (sign)
def sign():
    im = np.full((380, 900, 3), (110, 60, 20), np.uint8)
    cv2.putText(im, "EXIT GATE 7", (70, 170), cv2.FONT_HERSHEY_DUPLEX, 3, (255,255,255), 5, cv2.LINE_AA)
    cv2.putText(im, "Platform Open 24 Hours", (70, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 3, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((450, 190), 12, 1)
    cv2.imwrite("samples/6_rotated_inverted_sign.png", cv2.warpAffine(im, M, (900, 380), borderValue=(110,60,20)))

for f in (arc, wavy, dim, noisy, hand, sign): f()
