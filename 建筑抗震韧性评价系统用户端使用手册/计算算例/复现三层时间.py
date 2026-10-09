import os
import numpy as np
import pandas as pd
from scipy.stats import norm

class TimeResilienceEvaluator:
    def __init__(self, file_name='中震三层数据导出版.xls'):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.data_dir, file_name)
        self.num_simulations = 1000  
        self.load_data()
        
    def load_data(self):
        print(f"正在加载真实数据文件: {self.file_path} ...")
        xls = pd.ExcelFile(self.file_path)
        
        # 楼层面积提取
        self.floor_info = pd.read_excel(xls, '建筑信息2')
        self.floor_info.set_index('楼层（层）', inplace=True)
        self.floor_areas = self.floor_info['楼层面积（m^2）'].values
        self.num_floors = len(self.floor_info) # 补回此行，获取总楼层数
        
        # 提取 11 条波的原始结构响应 EDP
        edp_raw = pd.read_excel(xls, '结构响应')
        self.drifts_x = edp_raw.iloc[2:5, 1:12].replace('--', 0).astype(float).values
        self.accels_x = edp_raw.iloc[2:5, 12:23].replace('--', 0).astype(float).values
        self.drifts_y = edp_raw.iloc[2:5, 23:34].replace('--', 0).astype(float).values
        self.accels_y = edp_raw.iloc[2:5, 34:45].replace('--', 0).astype(float).values

        # 核心易损性数据库 (仅保留对时间计算有用的 medians, betas, repair_times)
        self.fragility_db = {
            'A.98.Z.Z.Z.814': {'cat': '结构构件', 'edp': 'drift', 'medians': [0.005, 0.010, 0.020, 0.030], 'betas': [0.4, 0.4, 0.4, 0.4], 'repair_times': [3.8, 5.6, 11.3, 25.0]},
            'A.98.Z.Z.Z.813': {'cat': '结构构件', 'edp': 'drift', 'medians': [0.007, 0.010, 0.016, 0.023], 'betas': [0.4, 0.4, 0.4, 0.4], 'repair_times': [2.6, 6.2, 9.4, 27.8]},
            'B.98.Z.Z.Z.161': {'cat': '位移敏感型非结构构件', 'edp': 'drift', 'medians': [0.005, 0.010, 0.021], 'betas': [0.4, 0.3, 0.2], 'repair_times': [0.015, 0.029, 0.056]},
            'B.01.B.B.A.001': {'cat': '加速度敏感型非结构构件', 'edp': 'accel', 'medians': [1.17, 1.58, 1.82], 'betas': [0.25, 0.25, 0.25], 'repair_times': [0.015, 0.116, 0.240]},
            'B.01.A.A.A.001': {'cat': '位移敏感型非结构构件', 'edp': 'drift', 'medians': [0.0338, 0.0383], 'betas': [0.4, 0.4], 'repair_times': [0.0, 0.216]}
        }

    def expand_edp_matrix(self, edp_matrix):
        """EDP 联合对数正态分布扩充"""
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
        """对数正态拟合提取 84% 保证率极值"""
        arr = np.array(arr)
        valid = arr[arr > 1e-6] 
        if len(valid) == 0: return 0.0
        return float(np.exp(np.mean(np.log(valid)) + np.std(np.log(valid))))

    def evaluate_time(self):
        np.random.seed(42)  # 固定种子保证结果唯一真实性
        
        # 扩充 EDP 矩阵
        sim_drifts_x = self.expand_edp_matrix(self.drifts_x)
        sim_drifts_y = self.expand_edp_matrix(self.drifts_y)
        sim_accels_x = self.expand_edp_matrix(self.accels_x) / 9.8 
        sim_accels_y = self.expand_edp_matrix(self.accels_y) / 9.8 
        
        results_time_s1 = {f: [] for f in range(self.num_floors)}
        results_time_s2 = {f: [] for f in range(self.num_floors)}

        # 蒙特卡洛全概率抽样
        for sim_idx in range(self.num_simulations):
            for f_idx, floor in enumerate(self.floor_info.index):
                dx, dy = sim_drifts_x[sim_idx, f_idx], sim_drifts_y[sim_idx, f_idx]
                ax, ay = sim_accels_x[sim_idx, f_idx], sim_accels_y[sim_idx, f_idx]
                
                # 初始化标准修复工作类别工时 (W1, W3, W4, W5)
                q_w1, q_w3, q_w4, q_w5 = 0.0, 0.0, 0.0, 0.0
                
                for cid, cinfo in self.fragility_db.items():
                    qty = 2 if cid.startswith('A') else (44 if 'A.A.A.001' in cid else 37)
                    
                    # 多维 EDP 匹配测算
                    if cid.startswith('A') or cid == 'B.98.Z.Z.Z.161': 
                        edps = [dx, dy]
                    elif cinfo['edp'] == 'drift':
                        edps = [max(dx, dy)]
                    else:
                        edps = [max(ax, ay)]
                        
                    for edp_val in edps:
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
                            ds_idx = ds - 1
                            # 时间体积折减系数 ζ_T 
                            t_vol_disc = 0.75 if (cinfo['cat'] == '结构构件' and qty >= 50) else (0.90 if (cinfo['cat'] != '结构构件' and qty >= 10) else 1.0)
                            work_hours = cinfo['repair_times'][ds_idx] * qty * t_vol_disc
                            
                            # 工时精准归集
                            if cinfo['cat'] == '结构构件':
                                q_w1 += work_hours
                            elif cid == 'B.01.A.A.A.001': # 玻璃幕墙 (围护 W3)
                                q_w3 += work_hours
                            elif cid == 'B.98.Z.Z.Z.161': # 填充墙 (隔断 W4)
                                q_w4 += work_hours
                            elif cid == 'B.01.B.B.A.001': # 吊顶 (附属 W5)
                                q_w5 += work_hours

                # ========================================================
                # 遵循纯规范公式 (7) 和 (9) 分配可用劳动力 (按单层面积计算)
                # ========================================================
                floor_area = self.floor_areas[f_idx]
                n_max = 0.026 * floor_area
                
                # 各工种实际分配人数
                n_w1 = max(1e-6, min(n_max, (2.0 / 100) * floor_area))
                n_w3 = max(1e-6, min(n_max, (1.0 / 100) * floor_area))
                n_w4 = max(1e-6, min(n_max, (1.0 / 100) * floor_area))
                n_w5 = max(1e-6, min(n_max, (1.0 / 100) * floor_area))
                
                # 公式 (10) 计算独立工序单阶段耗时
                t_w1 = q_w1 / n_w1
                t_w3 = q_w3 / n_w3
                t_w4 = q_w4 / n_w4
                t_w5 = q_w5 / n_w5
                
                # 规范公式 (11) (12)：串联及并联时间总成
                time_s1 = t_w1                                
                time_s2 = max(t_w3, t_w4 + t_w5)              
                
                results_time_s1[f_idx].append(time_s1)
                results_time_s2[f_idx].append(time_s2)

        # ====== 结果汇总 ======
        final_res = {'floor_time_details': []}
        for f_idx, floor in enumerate(self.floor_info.index):
            # 取84%保证率
            ft_s1 = self.calc_84th(results_time_s1[f_idx])
            ft_s2 = self.calc_84th(results_time_s2[f_idx])
            ft_tot = ft_s1 + ft_s2
            
            final_res['floor_time_details'].append((floor, ft_tot, ft_s1, ft_s2))

        # 取各层最大值作为整体修复时间
        max_time_floor_detail = max(final_res['floor_time_details'], key=lambda x: x[1])
        final_res['time_total'] = max_time_floor_detail[1]
        final_res['time_s1_total'] = max_time_floor_detail[2]
        final_res['time_s2_total'] = max_time_floor_detail[3]

        return final_res

    def generate_report(self):
        try:
            res = self.evaluate_time()
        except Exception as e:
            print(f"数据处理过程出错：{e}")
            return
            
        print("="*80)
        print("建筑抗震韧性评级 - 修复时间专项计算核心模块")
        print("="*80)
        
        print("\n[修复时间评级]")
        print(f"总体修复时间: {res['time_total']:.2f} 天 (第一阶段: {res['time_s1_total']:.2f} 天, 第二阶段: {res['time_s2_total']:.2f} 天)")
        
        print("\n各楼层阶段性修复时间")
        print("-" * 60)
        print(f"{'楼层':<4} | {'修复时间':<10} | {'第一阶段(天)':<14} | {'第二阶段(天)'}")
        for row in res['floor_time_details']:
            print(f"{row[0]:<6} | {row[1]:<12.2f} | {row[2]:<16.2f} | {row[3]:.2f}")
        print("="*80)

if __name__ == "__main__":
    evaluator = TimeResilienceEvaluator()
    evaluator.generate_report()