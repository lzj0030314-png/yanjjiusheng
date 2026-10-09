import os
import numpy as np
import pandas as pd
from scipy.stats import norm

class TrueResilienceEvaluator:
    def __init__(self, file_name='中震三层数据导出版.xls'):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.data_dir, file_name)
        
        self.num_simulations = 1000  
        self.load_data()
        
    def load_data(self):
        print(f"正在加载真实数据文件: {self.file_path} ...")
        xls = pd.ExcelFile(self.file_path)
        
        bldg_info1 = pd.read_excel(xls, '建筑信息1').iloc[0]
        self.total_area = float(bldg_info1.get('建筑总面积（平方米）', 111.50))
        self.unit_cost = float(bldg_info1.get('单位造价（元/平方米）', 3000.0))
        self.bldg_total_cost = self.total_area * self.unit_cost
        
        self.floor_info = pd.read_excel(xls, '建筑信息2')
        self.floor_info.set_index('楼层（层）', inplace=True)
        self.num_floors = len(self.floor_info)
        self.floor_areas = self.floor_info['楼层面积（m^2）'].values  
        
        self.struct_info = pd.read_excel(xls, '结构构件信息')
        self.nonstruct_info = pd.read_excel(xls, '非结构构件信息')

        df_seismic = pd.read_excel(xls, '地震信息1')
        self.seismic_level = str(df_seismic['地震水准'].iloc[0]).strip()

        # 完整挂载全参数易损性数据库
        self.fragility_db = {
            'A.98.Z.Z.Z.814': {
                'cat': '结构构件', 'edp': 'drift', 'cost': 1500,
                'medians': [0.005, 0.010, 0.020, 0.030], 'betas': [0.4, 0.4, 0.4, 0.4],
                'loss_ratios': [0.1, 0.2, 0.6, 1.0], 'repair_factors': [1.22, 1.18, 1.06, 3.15],
                'repair_times': [3.8, 5.6, 11.3, 25.0],
                # GB/T 38591-2020 表C.10/C.12: ≤10→1.0, 11~49→0.85/0.75, ≥50→0.85/0.75 (阶跃式)
                'cost_vol_thresholds': [10.0, 49.0], 'cost_vol_factors': [1.0, 0.85, 0.85],
                'time_vol_thresholds': [10.0, 49.0], 'time_vol_factors': [1.0, 0.75, 0.75]
            },
            'A.98.Z.Z.Z.813': {
                'cat': '结构构件', 'edp': 'drift', 'cost': 1483,
                'medians': [0.007, 0.010, 0.016, 0.023], 'betas': [0.4, 0.4, 0.4, 0.4],
                'loss_ratios': [0.1, 0.2, 0.5, 1.0], 'repair_factors': [1.20, 1.15, 1.07, 3.57],
                'repair_times': [2.6, 6.2, 9.4, 27.8],
                # GB/T 38591-2020 表C.10/C.12: ≤10→1.0, 11~49→0.85/0.75, ≥50→0.85/0.75 (阶跃式)
                'cost_vol_thresholds': [10.0, 49.0], 'cost_vol_factors': [1.0, 0.85, 0.85],
                'time_vol_thresholds': [10.0, 49.0], 'time_vol_factors': [1.0, 0.75, 0.75]
            },
            'B.98.Z.Z.Z.161': {
                'cat': '位移敏感型非结构构件', 'edp': 'drift', 'cost': 150,
                'medians': [0.005, 0.010, 0.021], 'betas': [0.4, 0.3, 0.2],
                'loss_ratios': [0.22, 0.50, 1.00], 'repair_factors': [1.15, 1.21, 1.27],
                'repair_times': [0.015, 0.029, 0.056],
                # GB/T 38591-2020 表E.7: 1→1.0, 2~9→0.89/0.89, ≥10→0.89/0.89 (阶跃式)
                'cost_vol_thresholds': [1.0, 9.0], 'cost_vol_factors': [1.0, 0.89, 0.89],
                'time_vol_thresholds': [1.0, 9.0], 'time_vol_factors': [1.0, 0.89, 0.89]
            },
            'B.01.B.B.A.001': {
                'cat': '加速度敏感型非结构构件', 'edp': 'accel', 'cost': 225,
                'medians': [1.17, 1.58, 1.82], 'betas': [0.25, 0.25, 0.25],
                'loss_ratios': [0.10, 0.50, 1.00], 'repair_factors': [1.93, 1.49, 1.31],
                'repair_times': [0.015, 0.116, 0.240],
                # GB/T 38591-2020 表E.7(吊顶类): 1→1.0, 2~9→0.96/0.96, ≥10→0.96/0.96 (阶跃式)
                'cost_vol_thresholds': [1.0, 9.0], 'cost_vol_factors': [1.0, 0.96, 0.96],
                'time_vol_thresholds': [1.0, 9.0], 'time_vol_factors': [1.0, 0.96, 0.96]
            },
            'B.01.A.A.A.001': {
                'cat': '位移敏感型非结构构件', 'edp': 'drift', 'cost': 1520,
                'medians': [0.0338, 0.0383], 'betas': [0.4, 0.4],
                'loss_ratios': [0.0, 1.00], 'repair_factors': [0.0, 1.67],
                'repair_times': [0.0, 0.216],
                # GB/T 38591-2020 表E.7(幕墙/外墙类): 1→1.0, 2~9→0.89/0.89, ≥10→0.89/0.89 (阶跃式)
                'cost_vol_thresholds': [1.0, 9.0], 'cost_vol_factors': [1.0, 0.89, 0.89],
                'time_vol_thresholds': [1.0, 9.0], 'time_vol_factors': [1.0, 0.89, 0.89]
            }
        }
        
        edp_raw = pd.read_excel(xls, '结构响应')
        
        # 严格对齐国标 GB/T 38591-2020 且修复源数据错行Bug：读取后转置 (.T)，使矩阵形状为纯物理楼层 (11行波, 3列层)
        
        # =====================================================================
        # 终极“盲盒”数据重组：根据 pandas 真实索引彻底避开表头字符，强行对齐矩阵
        # =====================================================================
        
        # 1. 提取【层间位移角】(Drift)：
        # df.iloc 的 2、3、4 行，完美提取出物理的 1、2、3 层位移角
        self.edp_drift = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.drifts_x = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.drifts_y = edp_raw.iloc[[2, 3, 4], 23:34].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        
        # 2. 提取【楼面加速度】(Accel) - 【乱序重组】：
        # 物理1层(2.67)在 iloc[3], 物理2层(3.45)在 iloc[4], 物理3层(4.19最大)在 iloc[1]
        # 巧妙使用列表 [3, 4, 1] 瞬间完成乱序矩阵的物理复原，并彻底避开 iloc[0] 的字符报错！
        self.edp_accel = edp_raw.iloc[[3, 4, 1], 12:23].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.accels_x = edp_raw.iloc[[3, 4, 1], 12:23].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.accels_y = edp_raw.iloc[[3, 4, 1], 34:45].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        # =====================================================================

    def expand_edp_matrix(self, edp_matrix):
        """EDP联合对数正态扩充引擎"""
        edp_matrix = np.clip(edp_matrix, a_min=1e-6, a_max=None)
        log_edp = np.log(edp_matrix)
        
        # 国标要求：行代表分析次数，列代表参数。故列均值为 axis=0，协方差 rowvar=False
        mean_log = np.mean(log_edp, axis=0)
        cov_log = np.cov(log_edp, rowvar=False)
        
        eigvals, eigvecs = np.linalg.eigh(cov_log)
        eigvals[eigvals < 1e-8] = 1e-8 
        cov_log_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        sampled_log = np.random.multivariate_normal(mean_log, cov_log_pd, self.num_simulations)
        return np.exp(sampled_log)

    def calc_84th(self, arr):
        """提取 84% 保证率分位值 (GB/T 38591-2020 第9.4.1条)
        直接对全部样本取84%分位值, 不过滤零值, 避免高估
        """
        arr = np.array(arr, dtype=float)
        if len(arr) == 0:
            return 0.0
        return float(np.percentile(arr, 84))

    def get_comp_qty(self, cid, floor_num):
        """提取对应楼层真实构件的拆分方向数量"""
        s_match = self.struct_info[(self.struct_info['易损性编号'] == cid) & 
                                   (self.struct_info['起始楼层'] <= floor_num) & 
                                   (self.struct_info['终止楼层'] >= floor_num)]
        ns_match = self.nonstruct_info[(self.nonstruct_info['易损性编号'] == cid) & 
                                       (self.nonstruct_info['起始楼层'] <= floor_num) & 
                                       (self.nonstruct_info['终止楼层'] >= floor_num)]
        
        qty_x, qty_y, qty_none = 0, 0, 0
        for _, row in pd.concat([s_match, ns_match]).iterrows():
            # 取消max混合，强制把X、Y和无方向的构件分离提取出来
            qty_x += row.get('X方向易损性数据', 0)
            qty_y += row.get('Y方向易损性数据', 0)
            qty_none += row.get('无方向易损性数据', 0)
            
        return qty_x, qty_y, qty_none

    def get_volume_discount(self, qty, thresholds, factors):
        """根据真实工程量计算折减系数 (GB/T 38591-2020 阶跃式, 不插值)
        thresholds=[t1, t2] 将数量分为三档: ≤t1 / (t1, t2] / >t2
        factors=[f1, f2, f3] 对应三档折减系数
        """
        if qty <= thresholds[0]:
            return factors[0]
        elif qty <= thresholds[1]:
            return factors[1]
        else:
            return factors[2]

    def expected_cost(self, cinfo, edp_val, qty, vol_disc):
        """费用期望积分法"""
        if edp_val <= 1e-6 or qty <= 0: return 0.0
        exceed_probs = norm.cdf(np.log(edp_val / np.array(cinfo['medians'])) / np.array(cinfo['betas']))
        probs = np.zeros(len(cinfo['medians']) + 1)
        probs[0] = 1.0 - exceed_probs[0]
        for j in range(len(exceed_probs) - 1):
            probs[j+1] = exceed_probs[j] - exceed_probs[j+1]
        probs[-1] = exceed_probs[-1]
        
        cost = 0
        for ds_idx in range(len(cinfo['medians'])):
            cost += probs[ds_idx+1] * cinfo['loss_ratios'][ds_idx] * cinfo['cost'] * qty * cinfo['repair_factors'][ds_idx] * vol_disc
        return cost

    def evaluate(self):
        np.random.seed(42)

        self.bldg_comp_qtys = {}
        for cid in self.fragility_db.keys():
            tot = 0
            for f_name in self.floor_info.index:
                f_num = int(''.join(filter(str.isdigit, str(f_name))))
                qx, qy, qn = self.get_comp_qty(cid, f_num)
                tot += (qx + qy + qn)
            self.bldg_comp_qtys[cid] = tot

        
        f_c_str = {0:[], 1:[], 2:[]}
        f_c_dsp = {0:[], 1:[], 2:[]}
        f_c_acc = {0:[], 1:[], 2:[]}
        g_str_total, g_dsp_total, g_acc_total = [], [], []
        # GB/T 38591-2020 附录A.1: 输入参数宜通过蒙特卡洛法模拟不少于1000次
        # 记录每次模拟三类费用之和, 用于正确计算总费用84%分位值
        g_cost_total_per_sim = []

        # ===============================================
        # 模块 1：费用的蒙特卡洛评估 (1000次, 符合GB/T 38591-2020附录A.1)
        # ===============================================
        # 先扩充EDP矩阵 (1000次模拟)
        sim_drifts_x_cost = self.expand_edp_matrix(self.drifts_x)
        sim_drifts_y_cost = self.expand_edp_matrix(self.drifts_y)
        sim_accels_x_cost = self.expand_edp_matrix(self.accels_x) / 9.8
        sim_accels_y_cost = self.expand_edp_matrix(self.accels_y) / 9.8

        for w in range(self.num_simulations):
            g_s, g_d, g_a = 0, 0, 0
            for f in range(self.num_floors):
                floor_name = self.floor_info.index[f]
                floor_num = int(''.join(filter(str.isdigit, str(floor_name))))
                floor_influence = float(self.floor_info.loc[floor_name, '楼层影响系数']) if '楼层影响系数' in self.floor_info.columns else 1.0

                dx = sim_drifts_x_cost[w, f]
                dy = sim_drifts_y_cost[w, f]
                ax = sim_accels_x_cost[w, f]
                ay = sim_accels_y_cost[w, f]

                c_str, c_dsp, c_acc = 0, 0, 0

                for cid, cinfo in self.fragility_db.items():
                    qty_x, qty_y, qty_none = self.get_comp_qty(cid, floor_num)
                    total_qty = qty_x + qty_y + qty_none
                    if total_qty <= 0: continue

                    c_vol_disc = self.get_volume_discount(total_qty, cinfo['cost_vol_thresholds'], cinfo['cost_vol_factors'])

                    if cid.startswith('A') or cid == 'B.98.Z.Z.Z.161':
                        # X方向构件只受 X向地震破环
                        cost_x = self.expected_cost(cinfo, dx, qty_x, c_vol_disc)
                        # Y方向构件只受 Y向地震破坏
                        cost_y = self.expected_cost(cinfo, dy, qty_y, c_vol_disc)
                        # 无方向构件取两个方向破坏的包络值
                        cost_none = self.expected_cost(cinfo, max(dx, dy), qty_none, c_vol_disc)

                        total_dmg_cost = cost_x + cost_y + cost_none
                        if cid.startswith('A'): c_str += total_dmg_cost
                        else: c_dsp += total_dmg_cost

                    elif cid == 'B.01.A.A.A.001':
                        c_dsp += self.expected_cost(cinfo, np.sqrt(dx**2 + dy**2), total_qty, c_vol_disc)
                    elif cid == 'B.01.B.B.A.001':
                        c_acc += self.expected_cost(cinfo, np.sqrt(ax**2 + ay**2), total_qty, c_vol_disc)

                c_str *= floor_influence
                c_dsp *= floor_influence
                c_acc *= floor_influence

                f_c_str[f].append(c_str)
                f_c_dsp[f].append(c_dsp)
                f_c_acc[f].append(c_acc)

                g_s += c_str
                g_d += c_dsp
                g_a += c_acc

            g_str_total.append(g_s)
            g_dsp_total.append(g_d)
            g_acc_total.append(g_a)
            # 记录本次模拟三类费用之和, 用于正确计算总费用84%分位值
            g_cost_total_per_sim.append(g_s + g_d + g_a)

        # ===============================================
        # 模块 2 & 3：修复时间与人员伤亡的蒙特卡洛评估
        # ===============================================
        sim_drifts = self.expand_edp_matrix(self.edp_drift)
        sim_accels = self.expand_edp_matrix(self.edp_accel)
        sim_drifts_x = self.expand_edp_matrix(self.drifts_x)
        sim_drifts_y = self.expand_edp_matrix(self.drifts_y)
        sim_accels_x = self.expand_edp_matrix(self.accels_x) / 9.8 
        sim_accels_y = self.expand_edp_matrix(self.accels_y) / 9.8 
        
        results_time_s1 = {f: [] for f in range(self.num_floors)}
        results_time_s2 = {f: [] for f in range(self.num_floors)}
        results_injuries, results_deaths = [], []

        for sim_idx in range(self.num_simulations):
            sim_inj, sim_death = 0, 0
            
            for f_idx, floor in enumerate(self.floor_info.index):
                floor_num = int(''.join(filter(str.isdigit, str(floor))))
                floor_area = self.floor_areas[f_idx]
                pop_density = self.floor_info.loc[floor, '楼层人口密度']
                # 【新增这一行】：读取楼层影响系数，如果没有填则默认为 1.0
                floor_influence = float(self.floor_info.loc[floor, '楼层影响系数']) if '楼层影响系数' in self.floor_info.columns else 1.0

                floor_drift = sim_drifts[sim_idx, f_idx]
                floor_accel = sim_accels[sim_idx, f_idx] / 9.8
                
                max_ds_floor = 0
                for cid, cinfo in self.fragility_db.items():
                    qty_x, qty_y, qty_none = self.get_comp_qty(cid, floor_num)
                    total_qty_inj = qty_x + qty_y + qty_none
                    if total_qty_inj <= 0: continue
                    
                    edp_val = floor_drift if cinfo['edp'] == 'drift' else floor_accel
                    exceed_probs = norm.cdf(np.log(edp_val / np.array(cinfo['medians'])) / np.array(cinfo['betas']))
                    probs = np.zeros(len(cinfo['medians']) + 1)
                    probs[0] = 1.0 - exceed_probs[0]
                    for j in range(len(exceed_probs) - 1):
                        probs[j+1] = exceed_probs[j] - exceed_probs[j+1]
                    probs[-1] = exceed_probs[-1]
                    probs = np.clip(probs, 0, 1)
                    probs /= probs.sum()
                    
                    ds = np.random.choice(len(probs), p=probs)
                    if ds > 0:
                        max_ds_floor = max(max_ds_floor, ds)
                
                floor_pop = floor_area * pop_density
                # GB/T 38591-2020 公式(14)(15): 人员伤亡按面积和楼层影响系数计算
                # M_H = Σ_r (r_hr × Σ_k (S_r × ζ_k × A_{r,k}))
                # 用楼层面积×楼层影响系数作为加权面积, 替代原按人口计算
                weighted_area = floor_area * floor_influence
                if max_ds_floor >= 4:      # 对应表格 Ⅴ 级
                    sim_inj += weighted_area * (1/140)
                    sim_death += weighted_area * (1/800)
                elif max_ds_floor == 3:    # 对应表格 Ⅳ 级
                    sim_inj += weighted_area * (1/8000)
                    sim_death += weighted_area * (1/80000)
                elif max_ds_floor == 2:    # 对应表格 Ⅲ 级
                    sim_inj += weighted_area * (1/20000)
                    # 死亡率为0，无需累加
                elif max_ds_floor == 1:    # 完美对应表格 Ⅱ 级！
                    sim_inj += weighted_area * (1/80000)
                    # 死亡率为0，无需累加
                
                # 如果 max_ds_floor == 0 (对应表格 Ⅰ 级)，伤亡率为0，直接跳过

                dx, dy = sim_drifts_x[sim_idx, f_idx], sim_drifts_y[sim_idx, f_idx]
                ax, ay = sim_accels_x[sim_idx, f_idx], sim_accels_y[sim_idx, f_idx]
                
                # GB/T 38591-2020 表1: 共8个工种 W1~W8
                q_w1, q_w2, q_w3, q_w4, q_w5, q_w6, q_w7, q_w8 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

                def calc_time_for_qty(qty_sub, edp_sub, component_info, vol_disc_factor):
                    if qty_sub <= 0 or edp_sub <= 1e-6: return 0.0
                    e_probs = norm.cdf(np.log(edp_sub / np.array(component_info['medians'])) / np.array(component_info['betas']))
                    p_t = np.zeros(len(component_info['medians']) + 1)
                    p_t[0] = 1.0 - e_probs[0]
                    for j in range(len(e_probs) - 1): p_t[j+1] = e_probs[j] - e_probs[j+1]
                    p_t[-1] = e_probs[-1]
                    p_t = np.clip(p_t, 0, 1)
                    p_t /= p_t.sum()

                    sampled_ds = np.random.choice(len(p_t), p=p_t)
                    if sampled_ds > 0:
                        return component_info['repair_times'][sampled_ds - 1] * qty_sub * vol_disc_factor
                    return 0.0

                for cid, cinfo in self.fragility_db.items():
                    qty_x, qty_y, qty_none = self.get_comp_qty(cid, floor_num)
                    total_qty_time = qty_x + qty_y + qty_none
                    if total_qty_time <= 0: continue

                    t_vol_disc = self.get_volume_discount(total_qty_time, cinfo['time_vol_thresholds'], cinfo['time_vol_factors'])
                    work_hours = 0.0

                    if cid.startswith('A') or cid == 'B.98.Z.Z.Z.161':
                        # X方向构件只因为X向变形而产生修复时间，Y向同理，完美分离！
                        work_hours += calc_time_for_qty(qty_x, dx, cinfo, t_vol_disc)
                        work_hours += calc_time_for_qty(qty_y, dy, cinfo, t_vol_disc)
                        work_hours += calc_time_for_qty(qty_none, max(dx, dy), cinfo, t_vol_disc)
                    elif cinfo['edp'] == 'drift':
                        work_hours += calc_time_for_qty(total_qty_time, max(dx, dy), cinfo, t_vol_disc)
                    else:
                        work_hours += calc_time_for_qty(total_qty_time, max(ax, ay), cinfo, t_vol_disc)

                    # GB/T 38591-2020 表1 构件到工种映射:
                    # W1=结构构件, W2=内部构件, W3=承重墙, W4=幕墙, W5=非承重墙, W6=吊顶, W7=设备, W8=管道
                    if cid.startswith('A'):
                        q_w1 += work_hours          # 结构构件 → W1
                    elif cid == 'B.98.Z.Z.Z.161':
                        q_w5 += work_hours          # 填充墙(非承重墙) → W5
                    elif cid == 'B.01.A.A.A.001':
                        q_w4 += work_hours          # 幕墙构件 → W4
                    elif cid == 'B.01.B.B.A.001':
                        q_w6 += work_hours          # 吊顶构件 → W6

                # GB/T 38591-2020 公式(6): Q = Σ(Q×n) × ζ_T(h) × λ_T(k)
                # 楼层影响系数应乘在工作量Q上, 而非除在工人数N上
                lambda_Tk = 1.0  # 修复分组并行系数, 默认1.0(无分组)
                q_w1 *= floor_influence * lambda_Tk
                q_w2 *= floor_influence * lambda_Tk
                q_w3 *= floor_influence * lambda_Tk
                q_w4 *= floor_influence * lambda_Tk
                q_w5 *= floor_influence * lambda_Tk
                q_w6 *= floor_influence * lambda_Tk
                q_w7 *= floor_influence * lambda_Tk
                q_w8 *= floor_influence * lambda_Tk

                # GB/T 38591-2020 公式(9): N_k,max = 0.026 × A_g,k (不除楼层影响系数)
                n_max = 0.026 * floor_area

                # GB/T 38591-2020 表1: 各工种密度系数
                # W1=2/100m², W2=2/100m², W3=1/100m², W4=1/100m², W5=1/100m², W6=3/100m², W7=2/100m², W8=2/100m²
                n_w1 = max(1e-6, min(n_max, (2.0 / 100) * floor_area))
                n_w2 = max(1e-6, min(n_max, (2.0 / 100) * floor_area))
                n_w3 = max(1e-6, min(n_max, (1.0 / 100) * floor_area))
                n_w4 = max(1e-6, min(n_max, (1.0 / 100) * floor_area))
                n_w5 = max(1e-6, min(n_max, (1.0 / 100) * floor_area))
                n_w6 = max(1e-6, min(n_max, (3.0 / 100) * floor_area))
                n_w7 = max(1e-6, min(n_max, (2.0 / 100) * floor_area))
                n_w8 = max(1e-6, min(n_max, (2.0 / 100) * floor_area))

                t_w1 = q_w1 / n_w1 if q_w1 > 0 else 0.0
                t_w2 = q_w2 / n_w2 if q_w2 > 0 else 0.0
                t_w3 = q_w3 / n_w3 if q_w3 > 0 else 0.0
                t_w4 = q_w4 / n_w4 if q_w4 > 0 else 0.0
                t_w5 = q_w5 / n_w5 if q_w5 > 0 else 0.0
                t_w6 = q_w6 / n_w6 if q_w6 > 0 else 0.0
                t_w7 = q_w7 / n_w7 if q_w7 > 0 else 0.0
                t_w8 = q_w8 / n_w8 if q_w8 > 0 else 0.0

                # GB/T 38591-2020 公式(11)(12):
                # T_S1 = max(T_W1, T_W2)
                # T_S2 = max(T_W3, T_W4+T_W5, T_W6, T_W7, T_W8)
                time_s1 = max(t_w1, t_w2)
                time_s2 = max(t_w3, t_w4 + t_w5, t_w6, t_w7, t_w8)

                results_time_s1[f_idx].append(time_s1)
                results_time_s2[f_idx].append(time_s2)

            results_injuries.append(sim_inj)
            results_deaths.append(sim_death)

        # ===============================================
        # 结果汇总计算
        # ===============================================
        final_res = {'floor_cost_details': [], 'floor_time_details': []}

        # GB/T 38591-2020 表C.13/E.6: 修复时间层数调整系数 n_S
        # 12层以上→1.10, 7~12层→1.08, 4~6层→1.05, 1~3层→1.00
        num_floors = self.num_floors
        if num_floors >= 12:
            n_S = 1.10
        elif num_floors >= 7:
            n_S = 1.08
        elif num_floors >= 4:
            n_S = 1.05
        else:
            n_S = 1.00

        for f, floor in enumerate(self.floor_info.index):
            fc_str = (self.calc_84th(f_c_str[f]) / self.bldg_total_cost) * 100
            fc_dsp = (self.calc_84th(f_c_dsp[f]) / self.bldg_total_cost) * 100
            fc_acc = (self.calc_84th(f_c_acc[f]) / self.bldg_total_cost) * 100
            fc_tot = fc_str + fc_dsp + fc_acc
            final_res['floor_cost_details'].append((floor, fc_tot, fc_acc, fc_dsp, fc_str))

            ft_s1 = self.calc_84th(results_time_s1[f])
            ft_s2 = self.calc_84th(results_time_s2[f])
            # GB/T 38591-2020 公式(13): T_tot = (T_S1 + T_S2) × n_S
            ft_tot = (ft_s1 + ft_s2) * n_S
            final_res['floor_time_details'].append((floor, ft_tot, ft_s1, ft_s2))

        final_res['cost_struct'] = (self.calc_84th(g_str_total) / self.bldg_total_cost) * 100
        final_res['cost_disp'] = (self.calc_84th(g_dsp_total) / self.bldg_total_cost) * 100
        final_res['cost_accel'] = (self.calc_84th(g_acc_total) / self.bldg_total_cost) * 100
        # GB/T 38591-2020: 总费用84%分位值 = P84(每次模拟三类费用之和), 而非三类84%分位值之和
        final_res['cost_total'] = (self.calc_84th(g_cost_total_per_sim) / self.bldg_total_cost) * 100
        
        max_time_floor_detail = max(final_res['floor_time_details'], key=lambda x: x[1])
        final_res['time_total'] = max_time_floor_detail[1]
        final_res['time_s1_total'] = max_time_floor_detail[2]
        final_res['time_s2_total'] = max_time_floor_detail[3]

        # GB/T 38591-2020 公式(17)(18): γ = M / Σ(ζ_m × A_m), 按加权面积归一化
        if '楼层影响系数' in self.floor_info.columns:
            total_weighted_area = (self.floor_info['楼层面积（m^2）'] * self.floor_info['楼层影响系数']).sum()
        else:
            total_weighted_area = self.floor_info['楼层面积（m^2）'].sum()
        final_res['injury_rate'] = (self.calc_84th(results_injuries) / total_weighted_area) if total_weighted_area > 0 else 0
        final_res['death_rate'] = (self.calc_84th(results_deaths) / total_weighted_area) if total_weighted_area > 0 else 0

        return final_res

    def generate_report(self):
        try:
            res = self.evaluate()
        except Exception as e:
            print(f"数据处理过程出错，请检查输入表格格式是否正确：{e}")
            return
            
        print("="*80)
        print("二、建筑抗震韧性评级 - 综合汇总报告 (构件方向完全拆解终极版)")
        print("="*80)
        
        print("\n[修复费用评级]")
        print(f"总体修复费用: {res['cost_total']:.5f}%")
        print(f"(加速度敏感型: {res['cost_accel']:.5f}%, 位移敏感型: {res['cost_disp']:.5f}%, 结构构件: {res['cost_struct']:.5f}%)")
        
        print("\n各楼层修复类型对修复费用的贡献")
        print("-" * 75)
        print(f"{'楼层':<4} | {'总计费用(%)':<12} | {'加速度敏感型(%)':<15} | {'位移敏感型(%)':<15} | {'结构构件(%)'}")
        for row in res['floor_cost_details']:
            print(f"{row[0]:<6} | {row[1]:<14.5f} | {row[2]:<17.5f} | {row[3]:<17.5f} | {row[4]:.5f}")
            
        print("\n[修复时间评级]")
        print(f"总体修复时间: {res['time_total']:.2f} 天 (第一阶段: {res['time_s1_total']:.2f} 天, 第二阶段: {res['time_s2_total']:.2f} 天)")
        
        print("\n各楼层阶段性修复时间")
        print("-" * 60)
        print(f"{'楼层':<4} | {'修复时间':<10} | {'第一阶段(天)':<14} | {'第二阶段(天)'}")
        for row in res['floor_time_details']:
            print(f"{row[0]:<6} | {row[1]:<12.2f} | {row[2]:<16.2f} | {row[3]:.2f}")

        print("\n[人员损失评级]")
        print(f"最大受伤率: {res['injury_rate']:.5%} | 死亡率: {res['death_rate']:.5%}")

        # ==========================================
        # 星级评定逻辑 (对标 GB/T 38591-2020 第9.4节 + 表5/6/7)
        # ==========================================
        # 标准规定: 星级对应地震水准, 通过该水准阈值即获对应星级
        #   多遇地震 → 一星, 设防地震 → 二星, 罕遇地震 → 三星
        # 每个指标在该水准通过→对应星级, 不通过→无星级
        # 综合等级 = 三个指标中的最低等级 (木桶效应)
        print(f"\n[当前评估地震水准]: {self.seismic_level}")

        star_map = {"三星": 3, "二星": 2, "一星": 1, "无星级": 0}
        reverse_map = {3: "三星", 2: "二星", 1: "一星", 0: "无星级"}

        if '多遇' in self.seismic_level or '小震' in self.seismic_level:
            # 多遇地震 → 一星 (表5/6/7 第一行阈值)
            target_star = 1
            cost_pass = res['cost_total'] <= 5.0
            time_pass = res['time_total'] <= 7.0
            casualty_pass = (res['injury_rate'] <= 1.0e-4 and res['death_rate'] <= 1.0e-5)
        elif '设防' in self.seismic_level or '中震' in self.seismic_level:
            # 设防地震 → 二星 (表5/6/7 第二行阈值)
            target_star = 2
            cost_pass = res['cost_total'] <= 10.0
            time_pass = res['time_total'] <= 30.0
            casualty_pass = (res['injury_rate'] <= 1.0e-3 and res['death_rate'] <= 1.0e-4)
        elif '罕遇' in self.seismic_level or '大震' in self.seismic_level or '极罕遇' in self.seismic_level:
            # 罕遇地震 → 三星 (表5/6/7 第三行阈值)
            target_star = 3
            cost_pass = res['cost_total'] <= 10.0
            time_pass = res['time_total'] <= 30.0
            casualty_pass = (res['injury_rate'] <= 1.0e-3 and res['death_rate'] <= 1.0e-4)
        else:
            # 防错机制：如果表里填了奇奇怪怪的词
            target_star = 0
            cost_pass = time_pass = casualty_pass = False

        star_cost = reverse_map[target_star] if cost_pass else "无星级"
        star_time = reverse_map[target_star] if time_pass else "无星级"
        star_casualty = reverse_map[target_star] if casualty_pass else "无星级"

        # 综合星级判定：木桶效应，取三个指标中的最低等级
        min_star_val = min(star_map[star_cost], star_map[star_time], star_map[star_casualty])
        overall_star = reverse_map[min_star_val]

        # ==========================================
        # 打印最终评级
        # ==========================================
        print("\n[最终评级]")
        print(f"人员损失等级：\t{star_casualty}")
        print(f"修复时间等级：\t{star_time}")
        print(f"修复费用等级：\t{star_cost}")
        print(f"抗震韧性等级评价：\t{overall_star}")
        print("="*80)

if __name__ == "__main__":
    evaluator = TrueResilienceEvaluator()
    evaluator.generate_report()