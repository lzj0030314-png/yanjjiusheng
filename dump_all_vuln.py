# -*- coding: utf-8 -*-
"""检查分离的易损性 xlsx 文件"""
import pandas as pd
import os

base = r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例'

for fname in ['3-结构构件易损性.xlsx', '个人库易损性.xlsx']:
    p = os.path.join(base, fname)
    if not os.path.exists(p):
        print(f'{fname}: NOT FOUND')
        continue
    df = pd.read_excel(p)
    print(f'\n========== {fname} (shape={df.shape}) ==========')
    print(f'columns: {list(df.columns)}')

    # 找到 A.98 系列
    id_col = '易损性编号' if '易损性编号' in df.columns else '构件名称'
    df['ID_filled'] = df[id_col].ffill()

    # 显示前 20 行原始数据
    print('\n--- 前 20 行原始数据 ---')
    cols_show = [c for c in ['易损性编号', '构件名称', '构件造价', '构件类型', 'EDP类型',
                              '损伤状态名称', '中位值', '对数标准差',
                              '损失系数', '修复系数', '修复时间（人/天）',
                              '工程量折减系数（修复费用）', '工程量折减系数（修复费用）.1',
                              '工程量折减系数（修复时间）', '工程量折减系数（修复时间）.1']
                 if c in df.columns]
    print(df[cols_show].head(20).to_string())

    # 显示所有唯一构件
    print('\n--- 所有唯一构件 ---')
    for gid, g in df.groupby('ID_filled'):
        if pd.isna(gid):
            continue
        r0 = g.iloc[0]
        print(f'\n  ID: {gid}')
        print(f'    name: {r0.get("构件名称")}')
        print(f'    cost: {r0.get("构件造价")}')
        print(f'    type: {r0.get("构件类型")}')
        print(f'    edp: {r0.get("EDP类型")}')
        print(f'    repair_work: {r0.get("所属修复工作内容")}')
        for _, ds_row in g.iterrows():
            ds_name = ds_row.get('损伤状态名称')
            med = ds_row.get('中位值')
            beta = ds_row.get('对数标准差') or ds_row.get('对数方差')
            loss = ds_row.get('损失系数')
            rf = ds_row.get('修复系数')
            rt = ds_row.get('修复时间（人/天）')
            vol_cost = ds_row.get('工程量折减系数（修复费用）.1') or ds_row.get('工程量折减系数（修复费用）')
            vol_time = ds_row.get('工程量折减系数（修复时间）.1') or ds_row.get('工程量折减系数（修复时间）')
            print(f'    DS{ds_name}: med={med}, beta={beta}, loss={loss}, rf={rf}, rt={rt}, vol_cost={vol_cost}, vol_time={vol_time}')
