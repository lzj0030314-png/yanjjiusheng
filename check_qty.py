# -*- coding: utf-8 -*-
"""检查每层各构件类型的数量"""
import pandas as pd

base = r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例'
xls = pd.ExcelFile(base + r'\中震三层数据导出版.xls')

print('=== 结构构件信息 ===')
s = pd.read_excel(xls, '结构构件信息')
print('Columns:', list(s.columns))
print(s.to_string())
print()

print('=== 非结构构件信息 ===')
ns = pd.read_excel(xls, '非结构构件信息')
print('Columns:', list(ns.columns))
print(ns.to_string())
print()

print('=== 每层数量汇总 ===')
for f in [1, 2, 3]:
    # 结构
    s_mask = (s['起始楼层'] <= f) & (s['终止楼层'] >= f)
    s_qty = s.loc[s_mask, 'X方向易损性数据'].sum()
    # 位移
    b98_mask = ns['易损性编号'].str.startswith('B.98')
    ns_mask = (ns['起始楼层'] <= f) & (ns['终止楼层'] >= f)
    disp_qty = ns.loc[ns_mask & b98_mask, 'X方向易损性数据'].sum()
    # 加速度
    b01_mask = ns['易损性编号'].str.startswith('B.01')
    accel_qty = ns.loc[ns_mask & b01_mask, '无方向易损性数据'].sum()
    print(f'Floor {f}: struct_X={s_qty}, disp_X={disp_qty}, accel_nodir={accel_qty}')

# 楼层信息
print()
print('=== 楼层信息 ===')
floors = pd.read_excel(xls, '建筑信息2')
print(floors.to_string())
