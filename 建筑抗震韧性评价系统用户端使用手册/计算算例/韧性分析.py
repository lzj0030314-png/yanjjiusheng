# -*- coding: utf-8 -*-
"""
建筑抗震韧性评价 - 基于 GB/T 38591-2020 标准
从 xls 文件读取真实数据，实现蒙特卡洛模拟 + 对数正态 EDP 扩充 + 84% 保证率

复现目标（来自报告）:
  修复费用 = 2.39542% (加速度0.10039%, 位移1.63427%, 结构0.66076%)
  修复时间 = 29.86天 (第一阶段28.79, 第二阶段1.06)
  抗震韧性等级 = 一星
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# ============================================================
# 易损性参数数据库
# 基于 GB/T 38591-2020 附录 C/E 及系统内置库
# A.98.Z.Z.Z.813/814 = 框架结构构件（结构构件，位移敏感）
# B.98.Z.Z.Z.161 = 填充墙（位移敏感型非结构构件）
# B.01.B.B.A.001 + B.01.A.A.A.001 = 吊顶+管线（加速度敏感型非结构构件）
# ============================================================
FRAGILITY_DB = {
    'Struct': {
        'cat': '结构构件',
        'edp': 'drift',
        'cost': 500,            # 构件造价（元/个）— 合并 A.98.Z.Z.Z.813/814, qty=4
        'medians':  [0.0035, 0.0050, 0.0080, 0.0130],   # DS1-DS4 中位值
        'betas':    [0.37, 0.30, 0.32, 0.46],           # 对数标准差
        'loss_ratios':    [0.0, 0.10, 0.20, 0.50, 1.00], # DS0-DS4 损失系数（单调递增）
        'repair_factors': [0.0, 1.20, 1.07, 1.15, 3.57], # DS0-DS4 修复系数
        'repair_times':   [0.0, 2.6, 6.2, 9.4, 27.8],     # DS0-DS4 修复时间(人/天)
        'vol_discount': 0.85,   # 工程量折减系数
        'stage': 1,             # 所属修复工作阶段（1=结构+楼梯）
    },
    'Disp_NonStruct': {
        'cat': '位移敏感型非结构构件',
        'edp': 'drift',
        'cost': 75,            # B.98.Z.Z.Z.161 填充墙, qty=37
        'medians':  [0.0020, 0.0040, 0.0080],
        'betas':    [0.30, 0.30, 0.30],
        'loss_ratios':    [0.0, 0.22, 0.50, 1.00],  # 修正：DS1<DS2 单调递增
        'repair_factors': [0.0, 1.15, 1.21, 1.27],
        'repair_times':   [0.0, 0.015, 0.029, 0.056],
        'vol_discount': 0.89,
        'stage': 2,
    },
    'Accel_NonStruct': {
        'cat': '加速度敏感型非结构构件',
        'edp': 'accel',
        'cost': 5,             # B.01 系列 (吊顶+管线), qty=81
        'medians':  [0.80, 1.10, 1.69],   # 单位: g
        'betas':    [0.38, 0.32, 0.25],
        'loss_ratios':    [0.0, 0.10, 0.50, 1.00],  # 修正：DS1<DS2 单调递增
        'repair_factors': [0.0, 1.93, 1.49, 1.31],
        'repair_times':   [0.0, 0.014, 0.111, 0.230],
        'vol_discount': 0.96,
        'stage': 2,
    },
}

# 楼层最大破坏等级对应的人员伤亡名义率
CASUALTY_RATES = {
    # damage_grade: (injury_rate, death_rate)
    1: (0.0, 0.0),                    # I 级: 基本完好
    2: (1/80000, 0.0),                # II 级: 轻微破坏
    3: (1/20000, 1/80000),            # III 级: 轻度破坏
    4: (1/8000, 1/80000),             # IV 级: 中度破坏
    5: (1/140, 1/800),                # V 级: 重度破坏
}


class ResilienceEvaluator:
    def __init__(self, file_name='中震三层数据导出版.xls'):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.data_dir, file_name)
        self.num_simulations = 1000
        np.random.seed(42)
        self._load_data()

    def _load_data(self):
        """从 xls 读取所有输入数据"""
        print(f"加载文件: {self.file_path}")
        xls = pd.ExcelFile(self.file_path)

        # --- 建筑信息1 ---
        bldg = pd.read_excel(xls, '建筑信息1').iloc[0]
        self.total_area = float(bldg['建筑总面积（平方米）'])
        self.unit_cost = float(bldg['单位造价（元/平方米）'])
        self.bldg_total_cost = self.total_area * self.unit_cost
        self.num_floors = int(bldg['地上层数'])

        # --- 建筑信息2（楼层信息）---
        floors = pd.read_excel(xls, '建筑信息2')
        self.floor_ids = floors['楼层（层）'].tolist()
        self.floor_areas = floors['楼层面积（m^2）'].astype(float).values
        self.floor_heights = floors['楼层高度（米）'].astype(float).values
        self.floor_influence = floors['楼层影响系数'].astype(float).values
        self.floor_pop_density = floors['楼层人口密度'].astype(float).values

        # --- 结构构件信息 ---
        self.struct_info = pd.read_excel(xls, '结构构件信息')

        # --- 非结构构件信息 ---
        self.nonstruct_info = pd.read_excel(xls, '非结构构件信息')

        # --- 结构响应（EDP）---
        # 布局: row0=标题, row1=波名, row2=楼层0(基底), row3=楼层1, row4=楼层2, row5=楼层3
        # col0=楼层, col1-11=X方向层间位移角(11波), col12-22=X方向楼面加速度
        # col23-33=Y方向层间位移角, col34-44=Y方向楼面加速度
        edp_raw = pd.read_excel(xls, '结构响应', header=None)
        # 取楼层 1~N (row 3 到 3+N), 使用 X 方向（不利方向）
        n = self.num_floors
        row_start = 3  # 楼层1从 row 3 开始
        # X方向位移角: col 1-11, X方向加速度: col 12-22
        self.edp_drift = edp_raw.iloc[row_start:row_start+n, 1:12].replace('--', np.nan)
        self.edp_drift = self.edp_drift.astype(float).values  # shape (n_floors, 11)
        self.edp_accel = edp_raw.iloc[row_start:row_start+n, 12:23].replace('--', np.nan)
        self.edp_accel = self.edp_accel.astype(float).values  # m/s²

        print(f"  建筑总造价: {self.bldg_total_cost:.0f} 元")
        print(f"  楼层数: {n}")
        print(f"  EDP位移角矩阵 shape: {self.edp_drift.shape}")
        print(f"  EDP加速度矩阵 shape: {self.edp_accel.shape}")

    def _get_component_qty(self, floor_idx):
        """获取每层各构件类型的数量"""
        floor_id = floor_idx + 1  # 楼层编号从1开始
        qty = {}

        # 结构构件 (A.98系列): X方向数量
        s = self.struct_info
        mask = (s['起始楼层'] <= floor_id) & (s['终止楼层'] >= floor_id)
        struct_qty = int(s.loc[mask, 'X方向易损性数据'].sum())

        # 非结构构件: 位移敏感(B.98) + 加速度敏感(B.01)
        ns = self.nonstruct_info
        mask_ns = (ns['起始楼层'] <= floor_id) & (ns['终止楼层'] >= floor_id)

        # B.98 位移敏感: X方向数量
        b98_mask = ns['易损性编号'].str.startswith('B.98')
        disp_qty = int(ns.loc[mask_ns & b98_mask, 'X方向易损性数据'].sum())

        # B.01 加速度敏感: 无方向数量
        b01_mask = ns['易损性编号'].str.startswith('B.01')
        accel_qty = int(ns.loc[mask_ns & b01_mask, '无方向易损性数据'].sum())

        return {'Struct': struct_qty, 'Disp_NonStruct': disp_qty, 'Accel_NonStruct': accel_qty}

    def _expand_edp(self, edp_matrix):
        """EDP联合对数正态扩充 (GB/T 38591 附录G)
        基于原始 11 波数据估计 μ 和 Σ, 用多元正态采样, exp() 还原
        使用协方差矩阵特征值正则化保证半正定
        """
        edp_matrix = np.array(edp_matrix, dtype=float)
        # 处理空值: 用该层均值填充
        for i in range(edp_matrix.shape[0]):
            row = edp_matrix[i]
            valid = row[~np.isnan(row)]
            if len(valid) > 0 and np.isnan(row).any():
                edp_matrix[i][np.isnan(row)] = valid.mean()
            elif len(valid) == 0:
                edp_matrix[i] = 1e-6

        edp_matrix = np.clip(edp_matrix, 1e-8, None)
        log_edp = np.log(edp_matrix)  # shape (n_floors, 11)

        # 联合多元正态采样（捕获楼层间相关性）
        mean_log = np.mean(log_edp, axis=1)  # shape (n_floors,)
        cov_log = np.cov(log_edp)  # shape (n_floors, n_floors)

        # 特征值正则化: 将负/微小特征值截断为正值，保证协方差矩阵半正定
        eigvals, eigvecs = np.linalg.eigh(cov_log)
        eigvals = np.maximum(eigvals, 1e-8)
        cov_log_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # 检查每个楼层的方差是否过小（如加速度楼层1全为常数2.0），使用独立正态作为退化情况
        diag_var = np.diag(cov_log_pd)
        if np.any(diag_var < 1e-6):
            # 退化情况：存在零方差楼层，回退到每层独立正态采样
            n_floors = log_edp.shape[0]
            expanded = np.zeros((self.num_simulations, n_floors))
            for f in range(n_floors):
                log_vals = log_edp[f]
                mu = np.mean(log_vals)
                sigma = np.std(log_vals, ddof=1)
                if sigma < 1e-6:
                    sigma = 0.1  # 避免零方差
                expanded[:, f] = np.random.normal(mu, sigma, self.num_simulations)
            return np.exp(expanded)

        sampled_log = np.random.multivariate_normal(mean_log, cov_log_pd, self.num_simulations)
        return np.exp(sampled_log)  # shape (num_sim, n_floors)

    def _calc_damage_probs(self, edp_val, medians, betas):
        """计算各损伤状态概率 P(DS=j)
        medians, betas: DS1..DSn 的参数
        返回 [P(DS=0), P(DS=1), ..., P(DS=n)]
        """
        n_ds = len(medians)
        edp_val = max(edp_val, 1e-10)
        # 超越概率 P(DS >= j)
        exceed = np.array([norm.cdf(np.log(edp_val / m) / b) for m, b in zip(medians, betas)])
        # 各状态概率
        probs = np.zeros(n_ds + 1)
        probs[0] = 1.0 - exceed[0]  # P(DS=0)
        for j in range(n_ds - 1):
            probs[j + 1] = exceed[j] - exceed[j + 1]
        probs[-1] = exceed[-1]  # P(DS=n)
        probs = np.clip(probs, 0, 1)
        s = probs.sum()
        if s > 0:
            probs /= s
        return probs

    def _calc_84th(self, arr):
        """计算 84% 保证率值 = exp(μ + σ), μ/σ 为 log 后的均值/标准差"""
        arr = np.array(arr, dtype=float)
        valid = arr[arr > 1e-10]
        if len(valid) == 0:
            return 0.0
        log_valid = np.log(valid)
        return float(np.exp(np.mean(log_valid) + np.std(log_valid, ddof=1)))

    def evaluate(self):
        """核心评估"""
        np.random.seed(42)

        # 扩充 EDP
        sim_drifts = self._expand_edp(self.edp_drift)   # (1000, n_floors)
        sim_accels_ms2 = self._expand_edp(self.edp_accel)  # (1000, n_floors) in m/s²
        # 加速度中位值单位是 g, 需将 EDP 从 m/s² 转为 g
        sim_accels = sim_accels_ms2 / 9.81  # 转为 g

        n = self.num_floors
        comp_types = ['Struct', 'Disp_NonStruct', 'Accel_NonStruct']

        # 存储每次模拟的结果
        floor_costs = {f: {ct: [] for ct in comp_types} for f in range(n)}
        floor_times_s1 = {f: [] for f in range(n)}
        floor_times_s2 = {f: [] for f in range(n)}
        sim_injuries = []
        sim_deaths = []

        for sim in range(self.num_simulations):
            total_inj, total_death = 0.0, 0.0

            for f in range(n):
                floor_drift = sim_drifts[sim, f]
                floor_accel = sim_accels[sim, f]
                floor_area = self.floor_areas[f]
                # GB/T 38591: 单层工人上限 = 0.026 × 楼层面积, 至少 1 人
                max_workers = max(1.0, 0.026 * floor_area)

                qty = self._get_component_qty(f)

                f_cost = {ct: 0.0 for ct in comp_types}
                f_time_s1 = 0.0
                f_time_s2 = 0.0
                max_ds_overall = 0

                for ct in comp_types:
                    db = FRAGILITY_DB[ct]
                    n_qty = qty[ct]
                    if n_qty <= 0:
                        continue

                    edp_val = floor_drift if db['edp'] == 'drift' else floor_accel
                    probs = self._calc_damage_probs(edp_val, db['medians'], db['betas'])

                    # 期望值法: E[cost|EDP] = Σ_j P(DS=j) × cost_j
                    # 比随机采样DS更稳定，方差小，收敛快
                    exp_cost = 0.0
                    exp_time = 0.0
                    for j in range(1, len(probs)):
                        p_j = probs[j]
                        if p_j <= 0:
                            continue
                        # cost_j = 造价 × 数量 × 损失系数 × 修复系数 × 工程量折减
                        c_j = (db['cost'] * n_qty *
                               db['loss_ratios'][j] *
                               db['repair_factors'][j] *
                               db['vol_discount'])
                        # time_j = 修复时间(人/天) × 数量 × 工程量折减 / 工人数
                        t_j = (db['repair_times'][j] * n_qty *
                               db['vol_discount'] / max_workers)
                        exp_cost += p_j * c_j
                        exp_time += p_j * t_j

                    f_cost[ct] += exp_cost
                    if db['stage'] == 1:
                        f_time_s1 = max(f_time_s1, exp_time)
                    else:
                        f_time_s2 = max(f_time_s2, exp_time)

                    # 仅用于人员伤亡的最大DS采样（保留随机性以反映破坏不确定性）
                    ds = np.random.choice(len(probs), p=probs)
                    if ds > max_ds_overall:
                        max_ds_overall = ds

                for ct in comp_types:
                    floor_costs[f][ct].append(f_cost[ct])
                floor_times_s1[f].append(f_time_s1)
                floor_times_s2[f].append(f_time_s2)

                # 人员伤亡
                floor_pop = floor_area * self.floor_pop_density[f]
                # 将构件DS映射为楼层破坏等级 (1-5)
                if max_ds_overall >= 4:
                    grade = 5
                elif max_ds_overall == 3:
                    grade = 4
                elif max_ds_overall == 2:
                    grade = 3
                elif max_ds_overall == 1:
                    grade = 2
                else:
                    grade = 1
                inj_rate, death_rate = CASUALTY_RATES.get(grade, (0, 0))
                total_inj += floor_pop * inj_rate
                total_death += floor_pop * death_rate

            sim_injuries.append(total_inj)
            sim_deaths.append(total_death)

        # ================= 汇总 84% 保证率 =================
        result = {'floor_cost': [], 'floor_time': []}
        tot_str = tot_disp = tot_acc = 0.0

        for f in range(n):
            fc_str = (self._calc_84th(floor_costs[f]['Struct']) / self.bldg_total_cost) * 100
            fc_dsp = (self._calc_84th(floor_costs[f]['Disp_NonStruct']) / self.bldg_total_cost) * 100
            fc_acc = (self._calc_84th(floor_costs[f]['Accel_NonStruct']) / self.bldg_total_cost) * 100
            fc_tot = fc_str + fc_dsp + fc_acc
            result['floor_cost'].append((f + 1, fc_tot, fc_acc, fc_dsp, fc_str))
            tot_str += fc_str
            tot_disp += fc_dsp
            tot_acc += fc_acc

            ft_s1 = self._calc_84th(floor_times_s1[f])
            ft_s2 = self._calc_84th(floor_times_s2[f])
            result['floor_time'].append((f + 1, ft_s1 + ft_s2, ft_s1, ft_s2))

        result['cost_struct'] = tot_str
        result['cost_disp'] = tot_disp
        result['cost_accel'] = tot_acc
        result['cost_total'] = tot_str + tot_disp + tot_acc

        # 修复时间: 第一阶段=结构(串行), 第二阶段=非结构(并行)
        result['time_s1'] = max([r[2] for r in result['floor_time']])
        result['time_s2'] = max([r[3] for r in result['floor_time']])
        result['time_total'] = result['time_s1'] + result['time_s2']

        # 人员伤亡率
        total_pop = (self.floor_areas * self.floor_pop_density).sum()
        result['injury_rate'] = (self._calc_84th(sim_injuries) / total_pop) if total_pop > 0 else 0
        result['death_rate'] = (self._calc_84th(sim_deaths) / total_pop) if total_pop > 0 else 0

        return result

    def generate_report(self):
        result = self.evaluate()

        print("=" * 80)
        print("建筑抗震韧性评级 - 综合汇总报告")
        print("=" * 80)

        print(f"\n[修复费用评级]")
        print(f"总体修复费用: {result['cost_total']:.5f}%")
        print(f"  加速度敏感型: {result['cost_accel']:.5f}%")
        print(f"  位移敏感型: {result['cost_disp']:.5f}%")
        print(f"  结构构件: {result['cost_struct']:.5f}%")

        print(f"\n各楼层修复类型对修复费用的贡献")
        print("-" * 75)
        print(f"{'楼层':<6} | {'总计(%)':<14} | {'加速度(%)':<14} | {'位移(%)':<14} | {'结构(%)'}")
        for floor, tot, acc, dsp, st in result['floor_cost']:
            print(f"{floor:<6} | {tot:<14.8f} | {acc:<14.8f} | {dsp:<14.8f} | {st:.8f}")

        print(f"\n[修复时间评级]")
        print(f"总体修复时间: {result['time_total']:.2f}天")
        print(f"  第一阶段: {result['time_s1']:.2f}天")
        print(f"  第二阶段: {result['time_s2']:.2f}天")

        print(f"\n各楼层阶段性修复时间")
        print("-" * 60)
        print(f"{'楼层':<6} | {'修复时间':<10} | {'第一阶段(天)':<14} | {'第二阶段(天)'}")
        for floor, tot, s1, s2 in result['floor_time']:
            print(f"{floor:<6} | {tot:<12.2f} | {s1:<16.2f} | {s2:.2f}")

        print(f"\n[人员损失评级]")
        print(f"受伤率: {result['injury_rate']:.6e}")
        print(f"死亡率: {result['death_rate']:.6e}")

        # 星级评定
        star_cost = "一星" if result['cost_total'] <= 10.0 else "无星级"
        star_time = "一星" if result['time_total'] <= 30.0 else "无星级"
        star_casualty = "一星" if (result['injury_rate'] <= 1e-3 and result['death_rate'] <= 1e-4) else "无星级"
        stars = [star_cost, star_time, star_casualty]
        overall = "一星" if all(s == "一星" for s in stars) else "无星级"

        print(f"\n[最终评级]")
        print(f"  修复费用等级: {star_cost}")
        print(f"  修复时间等级: {star_time}")
        print(f"  人员损失等级: {star_casualty}")
        print(f"  抗震韧性等级: {overall}")
        print("=" * 80)

        # 对比目标
        print("\n[与目标报告对比]")
        print(f"  修复费用: {result['cost_total']:.5f}% (目标: 2.39542%)")
        print(f"  修复时间: {result['time_total']:.2f}天 (目标: 29.86天)")


if __name__ == "__main__":
    evaluator = ResilienceEvaluator()
    evaluator.generate_report()
