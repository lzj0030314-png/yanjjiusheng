# -*- coding: utf-8 -*-
"""提取报告 PDF 的文本以确认修复时间数据"""
import os

pdf_path = r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例\中震项目报告成功版 .pdf'

# 尝试 pdfplumber
try:
    import pdfplumber
    print('=== 使用 pdfplumber ===')
    with pdfplumber.open(pdf_path) as pdf:
        print(f'总页数: {len(pdf.pages)}')
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                print(f'\n--- Page {i+1} ---')
                print(text)
except ImportError:
    print('pdfplumber 不可用')

# 尝试 PyPDF2
try:
    import PyPDF2
    print('\n=== 使用 PyPDF2 ===')
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        print(f'总页数: {len(reader.pages)}')
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                print(f'\n--- Page {i+1} ---')
                print(text)
except ImportError:
    print('PyPDF2 不可用')

# 尝试 pypdf
try:
    import pypdf
    print('\n=== 使用 pypdf ===')
    reader = pypdf.PdfReader(pdf_path)
    print(f'总页数: {len(reader.pages)}')
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            print(f'\n--- Page {i+1} ---')
            print(text)
except ImportError:
    print('pypdf 不可用')
