# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

xls = pd.ExcelFile(r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例\中震三层数据导出版.xls')
print('Sheet names:', xls.sheet_names)
print()

print('=== 结构响应 sheet ===')
df = pd.read_excel(xls, '结构响应', header=None)
print('Shape:', df.shape)
print('Row 0:', list(df.iloc[0, :5]))
print('Row 1 (wave):', list(df.iloc[1, :5]))
print()
print('Col 0 (楼层) all rows:')
for i in range(df.shape[0]):
    print(f'  row{i}:', df.iloc[i, 0])
print()
print('Row 2 (楼层0) col1-3 (X drift):', list(df.iloc[2, 1:4]))
print('Row 2 (楼层0) col12-14 (X accel):', list(df.iloc[2, 12:15]))
print('Row 3 (楼层1) col1-3 (X drift):', list(df.iloc[3, 1:4]))
print('Row 3 (楼层1) col12-14 (X accel):', list(df.iloc[3, 12:15]))
print('Row 4 (楼层2) col1-3 (X drift):', list(df.iloc[4, 1:4]))
print('Row 4 (楼层2) col12-14 (X accel):', list(df.iloc[4, 12:15]))
print('Row 5 (楼层3) col1-3 (X drift):', list(df.iloc[5, 1:4]))
print('Row 5 (楼层3) col12-14 (X accel):', list(df.iloc[5, 12:15]))
print()
print('Y drift (col 23-25) row2-5:')
for i in [2,3,4,5]:
    print(f'  row{i}:', list(df.iloc[i, 23:26]))
print('Y accel (col 34-36) row2-5:')
for i in [2,3,4,5]:
    print(f'  row{i}:', list(df.iloc[i, 34:37]))
