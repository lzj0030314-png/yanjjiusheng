# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

base = r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例'

# Check personal library for all IDs
p = base + r'\个人库易损性.xlsx'
df = pd.read_excel(p)
print('=== 个人库易损性.xlsx ===')
print('shape:', df.shape)
# forward fill ID column to group damage states
id_col = '易损性编号' if '易损性编号' in df.columns else '构件名称'
print(f'ID col: {id_col}')
# get unique components by forward-filling
df['ID_filled'] = df[id_col].ffill()
groups = df.groupby('ID_filled')
print(f'Unique components: {len(groups)}')
for gid, g in groups:
    if gid is None or (isinstance(gid, float) and np.isnan(gid)):
        continue
    r0 = g.iloc[0]
    print(f'\n--- {gid} ---')
    print(f'  name: {r0.get("构件名称")}')
    print(f'  cost: {r0.get("构件造价")}')
    print(f'  type: {r0.get("构件类型")}')
    print(f'  edp: {r0.get("EDP类型")}')
    print(f'  repair work: {r0.get("所属修复工作内容")}')
    print(f'  direction: {r0.get("是否具有方向性")}')
    # damage states
    for _, ds_row in g.iterrows():
        ds_name = ds_row.get('损伤状态名称')
        med = ds_row.get('中位值')
        beta = ds_row.get('对数方差')
        loss = ds_row.get('损失系数')
        rf = ds_row.get('修复系数')
        rt = ds_row.get('修复时间（人/天）')
        vol_cost = ds_row.get('工程量折减系数（修复费用）.1')
        vol_time = ds_row.get('工程量折减系数（修复时间）.1')
        print(f'  DS{ds_name}: median={med}, beta={beta}, loss={loss}, repair_factor={rf}, repair_time={rt}, vol_cost={vol_cost}, vol_time={vol_time}')
