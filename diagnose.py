# -*- coding: utf-8 -*-
"""诊断脚本：检查 EDP 扩充和 DS 概率分布"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

base = r'd:\代码复现\建筑抗震韧性评价系统用户端使用手册\计算算例'
xls = pd.ExcelFile(base + r'\中震三层数据导出版.xls')

# EDP 数据
edp_raw = pd.read_excel(xls, '结构响应', header=None)
print('=== 原始 EDP 数据 ===')
for i in [3, 4, 5]:  # 楼层 1, 2, 3
    drift_row = edp_raw.iloc[i, 1:12].astype(float).values
    accel_row = edp_raw.iloc[i, 12:23].astype(float).values
    print(f'楼层{i-2}:')
    print(f'  X drift (11波): min={drift_row.min():.6f}, max={drift_row.max():.6f}, mean={drift_row.mean():.6f}, std={drift_row.std(ddof=1):.6f}')
    print(f'  X accel m/s²:  min={accel_row.min():.4f}, max={accel_row.max():.4f}, mean={accel_row.mean():.4f}, std={accel_row.std(ddof=1):.4f}')
    print(f'  X accel g:     min={accel_row.min()/9.81:.4f}, max={accel_row.max()/9.81:.4f}, mean={accel_row.mean()/9.81:.4f}')

# 计算 log 空间的均值/标准差
print()
print('=== Log 空间统计 (用于 EDP 扩充) ===')
for i in [3, 4, 5]:
    drift_row = edp_raw.iloc[i, 1:12].astype(float).values
    accel_row = edp_raw.iloc[i, 12:23].astype(float).values
    log_d = np.log(drift_row)
    log_a = np.log(accel_row)
    print(f'楼层{i-2} drift  log: mu={log_d.mean():.4f}, sigma={log_d.std(ddof=1):.4f}, exp(mu)={np.exp(log_d.mean()):.6f}, exp(mu+sigma)={np.exp(log_d.mean()+log_d.std(ddof=1)):.6f}')
    print(f'楼层{i-2} accel  log: mu={log_a.mean():.4f}, sigma={log_a.std(ddof=1):.4f}, exp(mu)={np.exp(log_a.mean()):.4f}, exp(mu+sigma)={np.exp(log_a.mean()+log_a.std(ddof=1)):.4f}')

# 计算各楼层在均值 EDP 下的 DS 概率
print()
print('=== DS 概率 (使用均值 EDP) ===')

# Struct
struct_medians = [0.0035, 0.0050, 0.0080, 0.0130]
struct_betas = [0.37, 0.30, 0.32, 0.46]

# Disp
disp_medians = [0.0020, 0.0040, 0.0080]
disp_betas = [0.30, 0.30, 0.30]

# Accel (medians in g)
accel_medians = [0.80, 1.10, 1.69]
accel_betas = [0.38, 0.32, 0.25]


def calc_probs(edp_val, medians, betas):
    exceed = np.array([norm.cdf(np.log(edp_val / m) / b) for m, b in zip(medians, betas)])
    probs = np.zeros(len(medians) + 1)
    probs[0] = 1.0 - exceed[0]
    for j in range(len(exceed) - 1):
        probs[j + 1] = exceed[j] - exceed[j + 1]
    probs[-1] = exceed[-1]
    probs = np.clip(probs, 0, 1)
    s = probs.sum()
    if s > 0:
        probs /= s
    return probs


for i in [3, 4, 5]:
    drift_row = edp_raw.iloc[i, 1:12].astype(float).values
    accel_row = edp_raw.iloc[i, 12:23].astype(float).values
    floor = i - 2
    mean_drift = drift_row.mean()
    mean_accel_g = accel_row.mean() / 9.81
    print(f'楼层{floor} (drift={mean_drift:.6f}, accel_g={mean_accel_g:.4f}):')
    s_p = calc_probs(mean_drift, struct_medians, struct_betas)
    print(f'  Struct probs [P0,P1,P2,P3,P4]: {[f"{p:.4f}" for p in s_p]}')
    d_p = calc_probs(mean_drift, disp_medians, disp_betas)
    print(f'  Disp   probs [P0,P1,P2,P3]:   {[f"{p:.4f}" for p in d_p]}')
    a_p = calc_probs(mean_accel_g, accel_medians, accel_betas)
    print(f'  Accel  probs [P0,P1,P2,P3]:   {[f"{p:.4f}" for p in a_p]}')

# 84th 百分位的 EDP（用 lognormal 估计）
print()
print('=== 84th 百分位 EDP (lognormal fit) ===')
for i in [3, 4, 5]:
    drift_row = edp_raw.iloc[i, 1:12].astype(float).values
    accel_row = edp_raw.iloc[i, 12:23].astype(float).values
    floor = i - 2
    log_d = np.log(drift_row)
    log_a = np.log(accel_row)
    p84_drift = np.exp(log_d.mean() + log_d.std(ddof=1))
    p84_accel = np.exp(log_a.mean() + log_a.std(ddof=1)) / 9.81
    print(f'楼层{floor}: 84th drift={p84_drift:.6f}, 84th accel_g={p84_accel:.4f}')

    # 在 84th EDP 下计算 DS 概率
    s_p = calc_probs(p84_drift, struct_medians, struct_betas)
    print(f'         Struct probs: {[f"{p:.3f}" for p in s_p]}')
    d_p = calc_probs(p84_drift, disp_medians, disp_betas)
    print(f'         Disp probs:   {[f"{p:.3f}" for p in d_p]}')
    a_p = calc_probs(p84_accel, accel_medians, accel_betas)
    print(f'         Accel probs:  {[f"{p:.3f}" for p in a_p]}')
