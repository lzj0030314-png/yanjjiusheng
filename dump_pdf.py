"""提取两个 PDF 的全部文本以便分析 - 写在根目录避免中文路径问题"""
import pdfplumber
import os

files = [
    (r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例\中震项目报告成功版 .pdf', r'd:\代码复现\report.txt'),
    (r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例\抗震标准副本.pdf', r'd:\代码复现\standard.txt'),
]
for src, out in files:
    print(f'\n>>> extracting {src}')
    print(f'   exists: {os.path.exists(src)}')
    with pdfplumber.open(src) as pdf, open(out, 'w', encoding='utf-8') as f:
        f.write(f'==== {src} ====\n')
        f.write(f'Total pages: {len(pdf.pages)}\n\n')
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ''
            f.write(f'\n----- Page {i+1} -----\n')
            f.write(txt)
            f.write('\n')
            tables = page.extract_tables()
            for ti, tb in enumerate(tables):
                f.write(f'\n  >>> Table {ti+1} on page {i+1}:\n')
                for row in tb:
                    f.write('  | ' + ' | '.join('' if c is None else str(c) for c in row) + '\n')
    print(f'    saved to {out}')
