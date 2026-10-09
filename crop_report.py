"""Crop high-resolution report pages into top/bottom halves and save to ASCII-only paths"""
from PIL import Image
import os

# ASCII-only output dir
out_dir = r'C:\Users\86152\AppData\Local\Temp\report_crops'
os.makedirs(out_dir, exist_ok=True)

src_files = [
    r'd:\代码复现\report_hq_p1.png',
    r'd:\代码复现\report_hq_p2.png',
    r'd:\代码复现\report_hq_p3.png',
]

crop_idx = 0
for src in src_files:
    img = Image.open(src)
    w, h = img.size
    print(f'{src}: {w}x{h}')

    # Split into 4 vertical strips for finer OCR (top-left, top-right, bottom-left, bottom-right)
    # Actually, let's split into top/bottom halves and also left/right halves for 4 quadrants
    crops = [
        ('TL', (0, 0, w // 2, h // 2)),
        ('TR', (w // 2, 0, w, h // 2)),
        ('BL', (0, h // 2, w // 2, h)),
        ('BR', (w // 2, h // 2, w, h)),
    ]
    page_num = os.path.basename(src).split('_p')[1].split('.')[0]
    for name, box in crops:
        crop = img.crop(box)
        # Upscale 2x for better OCR
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        out = os.path.join(out_dir, f'crop_p{page_num}_{name}.png')
        crop.save(out)
        print(f'  saved {out}  size={crop.size}')
        crop_idx += 1

print(f'\nTotal crops: {crop_idx}')
print(f'Dir: {out_dir}')
