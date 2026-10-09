import os
import numpy as np
import pandas as pd
from scipy.stats import norm

class TrueCasualtyEvaluator:
    def __init__(self, file_name='中震三层数据导出版.xls'):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.data_dir, file_name)
        
        self.num_simulations = 1000  
        self.load_data()
        
    def load_data(self):
        print(f"正在加载真实数据文件: {self.file_path} ...")
        xls = pd.ExcelFile(self.file_path)
        
        # 2. 楼层信息
        self.floor_info = pd.read_excel(xls, '建筑信息2')
        self.floor_info.set_index('楼层（层）', inplace=True)
        
        # 3. 构件易损性与数量 (仅保留类别、edp、medians、betas)
        self.struct_info = pd.read_excel(xls, '结构构件信息')
        self.nonstruct_info = pd.read_excel(xls, '非结构构件信息')
        
        self.fragility_db = {
            'A.98.Z.Z.Z.814': {'cat': '结构构件', 'edp': 'drift', 'medians': [0.005, 0.010, 0.020, 0.030], 'betas': [0.4, 0.4, 0.4, 0.4]},
            'A.98.Z.Z.Z.813': {'cat': '结构构件', 'edp': 'drift', 'medians': [0.007, 0.010, 0.016, 0.023], 'betas': [0.4, 0.4, 0.4, 0.4]},
            'B.98.Z.Z.Z.161': {'cat': '位移敏感型非结构构件', 'edp': 'drift', 'medians': [0.005, 0.010, 0.021], 'betas': [0.4, 0.3, 0.2]},
            'B.01.B.B.A.001': {'cat': '加速度敏感型非结构构件', 'edp': 'accel', 'medians': [1.17, 1.58, 1.82], 'betas': [0.25, 0.25, 0.25]},
            'B.01.A.A.A.001': {'cat': '位移敏感型非结构构件', 'edp': 'drift', 'medians': [0.0338, 0.0383], 'betas': [0.4, 0.4]}
        }
        
        # 4. 提取结构响应 (EDP) - 严格退回仅使用单向(X向)与你原代码保持一致
        edp_raw = pd.read_excel(xls, '结构响应')
        self.edp_drift = edp_raw.iloc[2:5, 1:12].replace('--', 0).astype(float).values
        self.edp_accel = edp_raw.iloc[2:5, 12:23].replace('--', 0).astype(float).values

    def expand_edp_matrix(self, edp_matrix):
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
        arr = np.array(arr)
        valid = arr[arr > 1e-6] 
        if len(valid) == 0: return 0.0
        return float(np.exp(np.mean(np.log(valid)) + np.std(np.log(valid))))

    def get_comp_qty(self, cid, floor_idx):
        s_match = self.struct_info[(self.struct_info['易损性编号'] == cid) & 
                                   (self.struct_info['起始楼层'] <= floor_idx) & 
                                   (self.struct_info['终止楼层'] >= floor_idx)]
        ns_match = self.nonstruct_info[(self.nonstruct_info['易损性编号'] == cid) & 
                                       (self.nonstruct_info['起始楼层'] <= floor_idx) & 
                                       (self.nonstruct_info['终止楼层'] >= floor_idx)]
        total_qty = 0
        for _, row in pd.concat([s_match, ns_match]).iterrows():
            total_qty += max(row['X方向易损性数据'], row['Y方向易损性数据'], row['无方向易损性数据'])
        return total_qty

    def evaluate(self):
        np.random.seed(42)
        
        sim_drifts = self.expand_edp_matrix(self.edp_drift)
        sim_accels = self.expand_edp_matrix(self.edp_accel)
        
        results_injuries, results_deaths = [], []

        for sim_idx in range(self.num_simulations):
            sim_inj, sim_death = 0, 0
            
            for f_idx, floor in enumerate(self.floor_info.index):
                # 提取单次随机响应（严格使用你原版的单向遍历）
                floor_drift = sim_drifts[sim_idx, f_idx]
                floor_accel = sim_accels[sim_idx, f_idx] / 9.8 
                
                floor_area = self.floor_info.loc[floor, '楼层面积（m^2）']
                pop_density = self.floor_info.loc[floor, '楼层人口密度']
                
                max_ds_floor = 0
                
                for cid, cinfo in self.fragility_db.items():
                    qty = self.get_comp_qty(cid, floor)
                    if qty <= 0: continue
                    
                    # 取对应项 EDP
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

            results_injuries.append(sim_inj)
            results_deaths.append(sim_death)

        # ================= 真实数据的 84% 保证率统计汇总 =================
        final_res = {}
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
        print("二、建筑抗震韧性评级 - 人员损失专项计算模块 (真实数学推演版)")
        print("="*80)

        print("\n[人员损失评级]")
        print(f"最大受伤率: {res['injury_rate']:.5f}% | 死亡率: {res['death_rate']:.5f}%")

        star_casualty = "一星" if (res['injury_rate']/100 <= 0.1 and res['death_rate']/100 <= 0.001) else "无星级"

        print("\n[最终评级]")
        print(f"人员损失等级：\t{star_casualty}")
        print("="*80)

if __name__ == "__main__":
    evaluator = TrueCasualtyEvaluator()
    evaluator.generate_report()