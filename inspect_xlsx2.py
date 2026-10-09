# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

base = r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例'

# Check individual xlsx files for vulnerability data
files = [
    '3-结构构件易损性.xlsx',
    '个人库易损性.xlsx',
    '4-结构构件信息.xlsx',
    '6-非结构构件信息.xlsx',
    '2-建筑信息1.xlsx',
    '2-建筑信息2.xlsx',
    '7-地震信息1.xlsx',
    '7-地震信息2.xlsx',
    '8-结构响应-中震.xlsx',
]
for f in files:
    p = base + '\\' + f
    try:
        xls = pd.ExcelFile(p)
        print(f'=== {f} ===')
        print('Sheets:', xls.sheet_names)
        for sn in xls.sheet_names:
            df = pd.read_excel(xls, sn)
            print(f'  [{sn}] shape={df.shape} cols={list(df.columns)[:8]}')
            if df.shape[0] > 0:
                print(f'    row0: {dict(list(df.iloc[0].items())[:8])}')
        print()
    except Exception as e:
        print(f'ERROR {f}: {e}')
        print()
