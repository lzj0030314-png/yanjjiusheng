# -*- coding: utf-8 -*-
"""用 pypdfium2 将 PDF 转为 PNG 图片以便查看"""
import pypdfium2 as pdfium
import os

pdf_path = r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例\中震项目报告成功版 .pdf'
out_dir = r'd:\代码复现\pdf_pages'
os.makedirs(out_dir, exist_ok=True)

pdf = pdfium.PdfDocument(pdf_path)
print(f'Pages: {len(pdf)}')
for i, page in enumerate(pdf):
    pil_image = page.render(scale=2.0).to_pil()  # scale=2 for higher resolution
    out_path = os.path.join(out_dir, f'page_{i+1}.png')
    pil_image.save(out_path)
    print(f'Saved {out_path} size={pil_image.size}')
