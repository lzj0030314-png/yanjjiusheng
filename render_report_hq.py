"""高分辨率渲染报告 PDF 3 页"""
import pypdfium2 as pdfium

src = r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例\中震项目报告成功版 .pdf'
pdf = pdfium.PdfDocument(src)
print(f'pages: {len(pdf)}')
for i in range(len(pdf)):
    page = pdf[i]
    # 高分辨率渲染 (scale=4 ~ 288 DPI)
    bitmap = page.render(scale=4)
    img = bitmap.to_pil()
    out = rf'd:\代码复现\report_hq_p{i+1}.png'
    img.save(out)
    print(f'  saved {out}  size={img.size}')
