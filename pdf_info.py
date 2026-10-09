"""检查 PDF 是否为图像，并渲染成 PNG 以便观察"""
import pypdfium2 as pdfium
import os

for src, prefix in [
    (r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例\中震项目报告成功版 .pdf', 'report'),
    (r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例\抗震标准副本.pdf', 'standard'),
]:
    pdf = pdfium.PdfDocument(src)
    print(f'{src}: {len(pdf)} pages')
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=2)
        img = bitmap.to_pil()
        out = rf'd:\代码复现\{prefix}_p{i+1}.png'
        img.save(out)
        print(f'  saved {out}')
