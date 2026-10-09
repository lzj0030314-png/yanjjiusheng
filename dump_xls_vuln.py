# -*- coding: utf-8 -*-
"""检查 xls 中的易损性表内容"""
import pandas as pd

base = r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例'
xls = pd.ExcelFile(base + r'\中震三层数据导出版.xls')

for sheet in ['结构构件易损性', '非结构构件易损性']:
    print(f'\n========== {sheet} ==========')
    df = pd.read_excel(xls, sheet)
    print(f'shape: {df.shape}')
    print(f'columns: {list(df.columns)}')
    print(df.to_string())
