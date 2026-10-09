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
        
        # 完整挂载全参数易损性数据库（包含数量上下限与费用/时间折减系数）
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
        
        # 兼容处理表格中的 '--' 和 'NULL'，提取 1,2,3 层 (即 Pandas 的 2,3,4 行)
        self.edp_drift = edp_raw.iloc[2:5, 1:12].replace(['--', 'NULL'], 0).astype(float).values
        self.edp_accel = edp_raw.iloc[2:5, 12:23].replace(['--', 'NULL'], 0).astype(float).values
        
        self.drifts_x = edp_raw.iloc[2:5, 1:12].replace(['--', 'NULL'], 0).astype(float).values
        self.accels_x = edp_raw.iloc[2:5, 12:23].replace(['--', 'NULL'], 0).astype(float).values
        self.drifts_y = edp_raw.iloc[2:5, 23:34].replace(['--', 'NULL'], 0).astype(float).values
        self.accels_y = edp_raw.iloc[2:5, 34:45].replace(['--', 'NULL'], 0).astype(float).values

    def expand_edp_matrix(self, edp_matrix):
        """EDP联合对数正态扩充 (蒙特卡洛引擎)"""
        edp_matrix = np.clip(edp_matrix, a_min=1e-6, a_max=None)
        log_edp = np.log(edp_matrix)
        mean_log = np.mean(log_edp, axis=1)
        cov_log = np.cov(log_edp)
        
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
        """提取对应楼层的真实构件数量"""
        s_match = self.struct_info[(self.struct_info['易损性编号'] == cid) & 
                                   (self.struct_info['起始楼层'] <= floor_num) & 
                                   (self.struct_info['终止楼层'] >= floor_num)]
        ns_match = self.nonstruct_info[(self.nonstruct_info['易损性编号'] == cid) & 
                                       (self.nonstruct_info['起始楼层'] <= floor_num) & 
                                       (self.nonstruct_info['终止楼层'] >= floor_num)]
        total_qty = 0
        for _, row in pd.concat([s_match, ns_match]).iterrows():
            total_qty += max(row['X方向易损性数据'], row['Y方向易损性数据'], row['无方向易损性数据'])
        return total_qty

    def get_volume_discount(self, qty, thresholds, factors):
        """根据真实工程量动态插值计算折减系数"""
        if qty <= thresholds[0]:
            return factors[0]  
        elif qty >= thresholds[1]:
            return factors[1]  
        else:
            ratio = (qty - thresholds[0]) / (thresholds[1] - thresholds[0])
            return factors[0] - ratio * (factors[0] - factors[1])

    def expected_cost(self, cinfo, edp_val, qty, vol_disc):
        """费用期望积分法"""
        if edp_val <= 1e-6: return 0.0
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
        # 模块 1：费用的期望积分评估 (应用了费用动态折减)
        # ===============================================
        for w in range(11):
            g_s, g_d, g_a = 0, 0, 0
            for f in range(self.num_floors):
                dx = self.drifts_x[f][w]
                dy = self.drifts_y[f][w]
                ax = self.accels_x[f][w] / 9.8   
                ay = self.accels_y[f][w] / 9.8 
                
                c_str, c_dsp, c_acc = 0, 0, 0
                floor_name = self.floor_info.index[f]
                floor_num = int(''.join(filter(str.isdigit, str(floor_name))))
                
                for cid, cinfo in self.fragility_db.items():
                    qty = self.get_comp_qty(cid, floor_num)
                    if qty <= 0: continue
                    
                    # 调用插值计算获得专属费用折减系数
                    c_vol_disc = self.get_volume_discount(qty, cinfo['cost_vol_thresholds'], cinfo['cost_vol_factors'])
                    
                    if cid.startswith('A'): 
                        c_str += self.expected_cost(cinfo, dx, qty, c_vol_disc)
                        c_str += self.expected_cost(cinfo, dy, qty, c_vol_disc)
                    elif cid == 'B.98.Z.Z.Z.161': 
                        c_dsp += self.expected_cost(cinfo, dx, qty, c_vol_disc)
                        c_dsp += self.expected_cost(cinfo, dy, qty, c_vol_disc)
                    elif cid == 'B.01.A.A.A.001': 
                        c_dsp += self.expected_cost(cinfo, np.sqrt(dx**2 + dy**2), qty, c_vol_disc)
                    elif cid == 'B.01.B.B.A.001': 
                        c_acc += self.expected_cost(cinfo, np.sqrt(ax**2 + ay**2), qty, c_vol_disc)
                        
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
        # 模块 2 & 3：修复时间与人员伤亡的蒙特卡洛全概率评估
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
                
                floor_drift = sim_drifts[sim_idx, f_idx]
                floor_accel = sim_accels[sim_idx, f_idx] / 9.8 
                
                max_ds_floor = 0
                for cid, cinfo in self.fragility_db.items():
                    qty_inj = self.get_comp_qty(cid, floor_num)
                    if qty_inj <= 0: continue
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
                if max_ds_floor >= 4:
                    sim_inj += floor_pop * (1/140)
                    sim_death += floor_pop * (1/800)
                elif max_ds_floor == 3:
                    sim_inj += floor_pop * (1/8000)
                    sim_death += floor_pop * (1/80000)
                elif max_ds_floor == 2:
                    sim_inj += floor_pop * (1/20000)

                # 计算修复时间 (应用了时间动态折减)
                dx, dy = sim_drifts_x[sim_idx, f_idx], sim_drifts_y[sim_idx, f_idx]
                ax, ay = sim_accels_x[sim_idx, f_idx], sim_accels_y[sim_idx, f_idx]
                
                q_w1, q_w3, q_w4, q_w5 = 0.0, 0.0, 0.0, 0.0
                
                for cid, cinfo in self.fragility_db.items():
                    qty_time = self.get_comp_qty(cid, floor_num)
                    if qty_time <= 0: continue
                    
                    if cid.startswith('A') or cid == 'B.98.Z.Z.Z.161': 
                        edps = [dx, dy]
                    elif cinfo['edp'] == 'drift':
                        edps = [max(dx, dy)]
                    else:
                        edps = [max(ax, ay)]
                        
                    for edp_val_time in edps:
                        if edp_val_time <= 1e-6: continue
                        exceed_probs_t = norm.cdf(np.log(edp_val_time / np.array(cinfo['medians'])) / np.array(cinfo['betas']))
                        probs_t = np.zeros(len(cinfo['medians']) + 1)
                        probs_t[0] = 1.0 - exceed_probs_t[0]
                        for j in range(len(exceed_probs_t) - 1):
                            probs_t[j+1] = exceed_probs_t[j] - exceed_probs_t[j+1]
                        probs_t[-1] = exceed_probs_t[-1]
                        probs_t = np.clip(probs_t, 0, 1)
                        probs_t /= probs_t.sum()
                        
                        ds_t = np.random.choice(len(probs_t), p=probs_t)
                        if ds_t > 0:
                            ds_idx_t = ds_t - 1
                            
                            # 调用插值计算获得专属时间折减系数
                            t_vol_disc = self.get_volume_discount(qty_time, cinfo['time_vol_thresholds'], cinfo['time_vol_factors'])
                            
                            work_hours = cinfo['repair_times'][ds_idx_t] * qty_time * t_vol_disc
                            
                            if cinfo['cat'] == '结构构件':
                                q_w1 += work_hours
                            elif cid == 'B.01.A.A.A.001':
                                q_w3 += work_hours
                            elif cid == 'B.98.Z.Z.Z.161':
                                q_w4 += work_hours
                            elif cid == 'B.01.B.B.A.001':
                                q_w5 += work_hours

                n_max = 0.026 * floor_area
                n_w1 = max(1e-6, min(n_max, (2.0 / 100) * floor_area))
                n_w3 = max(1e-6, min(n_max, (1.0 / 100) * floor_area))
                n_w4 = max(1e-6, min(n_max, (1.0 / 100) * floor_area))
                n_w5 = max(1e-6, min(n_max, (1.0 / 100) * floor_area))
                
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
        final_res['injury_rate'] = (self.calc_84th(results_injuries) / total_population) * 100 if total_population > 0 else 0
        final_res['death_rate'] = (self.calc_84th(results_deaths) / total_population) * 100 if total_population > 0 else 0

        return final_res

    def generate_report(self):
        try:
            res = self.evaluate()
        except Exception as e:
            print(f"数据处理过程出错，请检查输入表格格式是否正确：{e}")
            return
            
        print("="*80)
        print("二、建筑抗震韧性评级 - 综合汇总报告 (全参动态插值终极版)")
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
        print(f"最大受伤率: {res['injury_rate']:.5f}% | 死亡率: {res['death_rate']:.5f}%")

        star_cost = "一星" if res['cost_total'] <= 10.0 else "无星级"
        star_time = "一星" if res['time_total'] <= 30.0 else "无星级"
        star_casualty = "一星" if (res['injury_rate']/100 <= 0.1 and res['death_rate']/100 <= 0.001) else "无星级"
        overall_star = "一星" if all(s == "一星" for s in [star_cost, star_time, star_casualty]) else "无星级"

        print("\n[最终评级]")
        print(f"人员损失等级：\t{star_casualty}")
        print(f"修复时间等级：\t{star_time}")
        print(f"修复费用等级：\t{star_cost}")
        print(f"抗震韧性等级评价：\t{overall_star}")
        print("="*80)

if __name__ == "__main__":
    evaluator = TrueResilienceEvaluator()
    evaluator.generate_report()