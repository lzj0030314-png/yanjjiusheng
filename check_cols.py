# -*- coding: utf-8 -*-
import pandas as pd
xls = pd.ExcelFile(r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例\中震三层数据导出版.xls')
for sn in ['结构构件信息', '非结构构件信息']:
    df = pd.read_excel(xls, sn)
    print(f'=== {sn} ===')
    print(f'Columns: {list(df.columns)}')
    print(df.to_string())
    print()
