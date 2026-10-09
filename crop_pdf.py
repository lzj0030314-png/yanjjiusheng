# -*- coding: utf-8 -*-
"""裁剪 PDF 图像的关键区域以查看修复时间表"""
from PIL import Image
import os

src = r'd:\代码复现\pdf_pages\page_h_2.png'
img = Image.open(src)
print(f'Image size: {img.size}')

# 裁剪图像分上下半部分和中间部分
w, h = img.size
out_dir = r'd:\代码复现\pdf_pages\crops'
os.makedirs(out_dir, exist_ok=True)

# 上 1/3
img.crop((0, 0, w, h//3)).save(os.path.join(out_dir, 'p2_top.png'))
# 中 1/3
img.crop((0, h//3, w, 2*h//3)).save(os.path.join(out_dir, 'p2_mid.png'))
# 下 1/3
img.crop((0, 2*h//3, w, h)).save(os.path.join(out_dir, 'p2_bot.png'))
print('Done. Saved 3 crops to', out_dir)

# 也对第3页做相同处理
src3 = r'd:\代码复现\pdf_pages\page_h_3.png'
img3 = Image.open(src3)
w3, h3 = img3.size
img3.crop((0, 0, w3, h3//3)).save(os.path.join(out_dir, 'p3_top.png'))
img3.crop((0, h3//3, w3, 2*h3//3)).save(os.path.join(out_dir, 'p3_mid.png'))
img3.crop((0, 2*h3//3, w3, h3)).save(os.path.join(out_dir, 'p3_bot.png'))

src1 = r'd:\代码复现\pdf_pages\page_h_1.png'
img1 = Image.open(src1)
w1, h1 = img1.size
img1.crop((0, 0, w1, h1//3)).save(os.path.join(out_dir, 'p1_top.png'))
img1.crop((0, h1//3, w1, 2*h1//3)).save(os.path.join(out_dir, 'p1_mid.png'))
img1.crop((0, 2*h1//3, w1, h1)).save(os.path.join(out_dir, 'p1_bot.png'))
print('Done with all pages')
