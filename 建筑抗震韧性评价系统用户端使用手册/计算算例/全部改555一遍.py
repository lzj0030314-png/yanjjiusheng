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
                'cost_vol_thresholds': [10.0, 50.0], 'cost_vol_factors': [1.0, 0.85],       
                'time_vol_thresholds': [10.0, 50.0], 'time_vol_factors': [1.0, 0.75]        
            },
            'A.98.Z.Z.Z.813': {
                'cat': '结构构件', 'edp': 'drift', 'cost': 1483, 
                'medians': [0.007, 0.010, 0.016, 0.023], 'betas': [0.4, 0.4, 0.4, 0.4],
                'loss_ratios': [0.1, 0.2, 0.5, 1.0], 'repair_factors': [1.20, 1.07, 1.15, 3.57],
                'repair_times': [2.6, 6.2, 9.4, 27.8],
                'cost_vol_thresholds': [10.0, 50.0], 'cost_vol_factors': [1.0, 0.85],       
                'time_vol_thresholds': [10.0, 50.0], 'time_vol_factors': [1.0, 0.75]        
            },
            'B.98.Z.Z.Z.161': {
                'cat': '位移敏感型非结构构件', 'edp': 'drift', 'cost': 150, 
                'medians': [0.005, 0.010, 0.021], 'betas': [0.4, 0.3, 0.2],
                'loss_ratios': [0.22, 0.50, 1.00], 'repair_factors': [1.15, 1.21, 1.27],
                'repair_times': [0.015, 0.029, 0.056],
                'cost_vol_thresholds': [10.0, 50.0], 'cost_vol_factors': [1.0, 0.89],       
                'time_vol_thresholds': [10.0, 50.0], 'time_vol_factors': [1.0, 0.90]        
            },
            'B.01.B.B.A.001': {
                'cat': '加速度敏感型非结构构件', 'edp': 'accel', 'cost': 225, 
                'medians': [1.17, 1.58, 1.82], 'betas': [0.25, 0.25, 0.25],
                'loss_ratios': [0.10, 0.50, 1.00], 'repair_factors': [1.93, 1.49, 1.31],
                'repair_times': [0.015, 0.116, 0.240],
                'cost_vol_thresholds': [10.0, 50.0], 'cost_vol_factors': [1.0, 0.96],       
                'time_vol_thresholds': [10.0, 50.0], 'time_vol_factors': [1.0, 0.90]        
            },
            'B.01.A.A.A.001': {
                'cat': '位移敏感型非结构构件', 'edp': 'drift', 'cost': 1520, 
                'medians': [0.0338, 0.0383], 'betas': [0.4, 0.4],
                'loss_ratios': [0.0, 1.00], 'repair_factors': [0.0, 1.67],
                'repair_times': [0.0, 0.216],
                'cost_vol_thresholds': [10.0, 50.0], 'cost_vol_factors': [1.0, 0.90],       
                'time_vol_thresholds': [10.0, 50.0], 'time_vol_factors': [1.0, 0.90]        
            }
        }
        
        edp_raw = pd.read_excel(xls, '结构响应')
        
        # 严格对齐国标 GB/T 38591-2020：读取后转置 (.T)，使矩阵形状为 (11行波, 4列层)
        self.edp_drift = edp_raw.iloc[1:5, 1:12].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.edp_accel = edp_raw.iloc[1:5, 12:23].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        
        self.drifts_x = edp_raw.iloc[1:5, 1:12].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.accels_x = edp_raw.iloc[1:5, 12:23].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.drifts_y = edp_raw.iloc[1:5, 23:34].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.accels_y = edp_raw.iloc[1:5, 34:45].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T

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
        """提取 84% 保证率极值"""
        arr = np.array(arr)
        valid = arr[arr > 1e-6] 
        if len(valid) == 0: return 0.0
        return float(np.exp(np.mean(np.log(valid)) + np.std(np.log(valid))))

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
        """根据真实工程量动态插值计算折减系数"""
        if qty <= thresholds[0]: return factors[0]  
        elif qty >= thresholds[1]: return factors[1]  
        else:
            ratio = (qty - thresholds[0]) / (thresholds[1] - thresholds[0])
            return factors[0] - ratio * (factors[0] - factors[1])

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
        
        f_c_str = {0:[], 1:[], 2:[]}
        f_c_dsp = {0:[], 1:[], 2:[]}
        f_c_acc = {0:[], 1:[], 2:[]}
        g_str_total, g_dsp_total, g_acc_total = [], [], []

        # ===============================================
        # 模块 1：费用的期望积分评估
        # ===============================================
        for w in range(11):
            g_s, g_d, g_a = 0, 0, 0
            for f in range(self.num_floors):
                floor_name = self.floor_info.index[f]
                floor_num = int(''.join(filter(str.isdigit, str(floor_name))))
                floor_influence = float(self.floor_info.loc[floor_name, '楼层影响系数']) if '楼层影响系数' in self.floor_info.columns else 1.0
                
                dx = self.drifts_x[w, floor_num]
                dy = self.drifts_y[w, floor_num]
                ax = self.accels_x[w, floor_num] / 9.8   
                ay = self.accels_y[w, floor_num] / 9.8 
                
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

                floor_drift = sim_drifts[sim_idx, floor_num]
                floor_accel = sim_accels[sim_idx, floor_num] / 9.8 
                
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
                if max_ds_floor >= 4:      # 对应表格 Ⅴ 级
                    sim_inj += floor_pop * (1/140)
                    sim_death += floor_pop * (1/800)
                elif max_ds_floor == 3:    # 对应表格 Ⅳ 级
                    sim_inj += floor_pop * (1/8000)
                    sim_death += floor_pop * (1/80000)
                elif max_ds_floor == 2:    # 对应表格 Ⅲ 级
                    sim_inj += floor_pop * (1/20000)
                    # 死亡率为0，无需累加
                elif max_ds_floor == 1:    # 完美对应表格 Ⅱ 级！
                    sim_inj += floor_pop * (1/80000)
                    # 死亡率为0，无需累加
                
                # 如果 max_ds_floor == 0 (对应表格 Ⅰ 级)，伤亡率为0，直接跳过

                dx, dy = sim_drifts_x[sim_idx, floor_num], sim_drifts_y[sim_idx, floor_num]
                ax, ay = sim_accels_x[sim_idx, floor_num], sim_accels_y[sim_idx, floor_num]
                
                q_w1, q_w3, q_w4, q_w5 = 0.0, 0.0, 0.0, 0.0
                
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
                        
                    if cid.startswith('A'): q_w1 += work_hours
                    elif cid == 'B.01.A.A.A.001': q_w3 += work_hours
                    elif cid == 'B.98.Z.Z.Z.161': q_w4 += work_hours
                    elif cid == 'B.01.B.B.A.001': q_w5 += work_hours

                # 用楼层影响系数折减这层楼能容纳的最大工人数
                n_max = (0.026 * floor_area) / floor_influence
                
                # 同样，各专业队的最大派工人数也受到影响系数的严格限制
                n_w1 = max(1e-6, min(n_max, ((2.0 / 100) * floor_area) / floor_influence))
                n_w3 = max(1e-6, min(n_max, ((1.0 / 100) * floor_area) / floor_influence))
                n_w4 = max(1e-6, min(n_max, ((1.0 / 100) * floor_area) / floor_influence))
                n_w5 = max(1e-6, min(n_max, ((1.0 / 100) * floor_area) / floor_influence))
                
                
                t_w1 = q_w1 / n_w1
                t_w3 = q_w3 / n_w3
                t_w4 = q_w4 / n_w4
                t_w5 = q_w5 / n_w5
                
                time_s1 = t_w1                                
                time_s2 = max(t_w3, t_w4 + t_w5)              
                
                results_time_s1[f_idx].append(time_s1)
                results_time_s2[f_idx].append(time_s2)

            results_injuries.append(sim_inj)
            results_deaths.append(sim_death)

        # ===============================================
        # 结果汇总计算
        # ===============================================
        final_res = {'floor_cost_details': [], 'floor_time_details': []}
        
        for f, floor in enumerate(self.floor_info.index):
            fc_str = (self.calc_84th(f_c_str[f]) / self.bldg_total_cost) * 100
            fc_dsp = (self.calc_84th(f_c_dsp[f]) / self.bldg_total_cost) * 100
            fc_acc = (self.calc_84th(f_c_acc[f]) / self.bldg_total_cost) * 100
            fc_tot = fc_str + fc_dsp + fc_acc
            final_res['floor_cost_details'].append((floor, fc_tot, fc_acc, fc_dsp, fc_str))
            
            ft_s1 = self.calc_84th(results_time_s1[f])
            ft_s2 = self.calc_84th(results_time_s2[f])
            ft_tot = ft_s1 + ft_s2
            final_res['floor_time_details'].append((floor, ft_tot, ft_s1, ft_s2))

        final_res['cost_struct'] = (self.calc_84th(g_str_total) / self.bldg_total_cost) * 100
        final_res['cost_disp'] = (self.calc_84th(g_dsp_total) / self.bldg_total_cost) * 100
        final_res['cost_accel'] = (self.calc_84th(g_acc_total) / self.bldg_total_cost) * 100
        final_res['cost_total'] = final_res['cost_struct'] + final_res['cost_disp'] + final_res['cost_accel']
        
        max_time_floor_detail = max(final_res['floor_time_details'], key=lambda x: x[1])
        final_res['time_total'] = max_time_floor_detail[1]
        final_res['time_s1_total'] = max_time_floor_detail[2]
        final_res['time_s2_total'] = max_time_floor_detail[3]

        total_population = (self.floor_info['楼层面积（m^2）'] * self.floor_info['楼层人口密度']).sum()
        final_res['injury_rate'] = (self.calc_84th(results_injuries) / total_population)  if total_population > 0 else 0
        final_res['death_rate'] = (self.calc_84th(results_deaths) / total_population)  if total_population > 0 else 0

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
        # 终极动态星级评定逻辑 (完全对标 GB/T 38591-2020)
        # ==========================================
        print(f"\n[当前评估地震水准]: {self.seismic_level}")
        
        if '设防' in self.seismic_level:
            # 【设防地震】最高只能评一星
            star_cost = "一星" if res['cost_total'] <= 10.0 else "无星级"
            star_time = "一星" if res['time_total'] <= 30.0 else "无星级"
            star_casualty = "一星" if (res['injury_rate'] <= 1.0e-3 and res['death_rate'] <= 1.0e-4) else "无星级"
            
        elif '罕遇' in self.seismic_level:
            # 【罕遇地震】可以评一到三星
            # 1. 费用评级
            if res['cost_total'] <= 5.0: star_cost = "三星"
            elif res['cost_total'] <= 10.0: star_cost = "二星"
            else: star_cost = "无星级"
            
            # 2. 时间评级
            if res['time_total'] <= 7.0: star_time = "三星"
            elif res['time_total'] <= 30.0: star_time = "二星"
            else: star_time = "无星级"
            
            # 3. 伤亡评级
            if res['injury_rate'] <= 1.0e-4 and res['death_rate'] <= 1.0e-5: 
                star_casualty = "三星"
            elif res['injury_rate'] <= 1.0e-3 and res['death_rate'] <= 1.0e-4: 
                star_casualty = "二星"
            else: 
                star_casualty = "无星级"
            
        else:
            # 防错机制：如果表里填了奇奇怪怪的词
            star_cost, star_time, star_casualty = "无星级", "无星级", "无星级"

        # 【综合星级判定：木桶效应】
        # 巧妙利用字典将星级转化为数字，取三个指标中的最小值！
        star_map = {"三星": 3, "二星": 2, "一星": 1, "无星级": 0}
        min_star_val = min(star_map[star_cost], star_map[star_time], star_map[star_casualty])
        
        # 将数字再转回文字
        reverse_map = {3: "三星", 2: "二星", 1: "一星", 0: "无星级"}
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