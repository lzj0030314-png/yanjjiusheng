# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

base = r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例'

# Full dump of vulnerability database
for f in ['3-结构构件易损性.xlsx', '个人库易损性.xlsx']:
    p = base + '\\' + f
    xls = pd.ExcelFile(p)
    sn = xls.sheet_names[0]
    df = pd.read_excel(xls, sn)
    print(f'=== {f} / [{sn}] shape={df.shape} ===')
    print('Columns:', list(df.columns))
    print()
    # Show key rows: ID, name, cost, type, edp, and damage params
    for i in range(df.shape[0]):
        row = df.iloc[i]
        print(f'--- Row {i} ---')
        print(f'  ID: {row.get("易损性编号", row.get("构件名称"))}')
        print(f'  name: {row.get("构件名称")}')
        print(f'  cost: {row.get("构件造价")}')
        print(f'  type: {row.get("构件类型")}')
        print(f'  edp: {row.get("EDP类型")}')
        # print all non-null columns
        for c in df.columns:
            v = row[c]
            if pd.notna(v) and c not in ['易损性编号','构件名称','构件造价','构件类型','EDP类型']:
                print(f'  {c}: {v}')
    print()
