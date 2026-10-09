import os
import numpy as np
import pandas as pd
from scipy.stats import norm

class CostResilienceEvaluator:
    def __init__(self, file_name='中震三层数据导出版.xls'):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.data_dir, file_name)
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
        
        edp_raw = pd.read_excel(xls, '结构响应')
        self.drifts_x = edp_raw.iloc[2:5, 1:12].replace('--', 0).astype(float).values
        self.accels_x = edp_raw.iloc[2:5, 12:23].replace('--', 0).astype(float).values
        self.drifts_y = edp_raw.iloc[2:5, 23:34].replace('--', 0).astype(float).values
        self.accels_y = edp_raw.iloc[2:5, 34:45].replace('--', 0).astype(float).values

        self.fragility_db = {
            'A.98.Z.Z.Z.814': {'cat': '结构构件', 'edp': 'drift', 'cost': 1500, 
                               'medians': [0.005, 0.010, 0.020, 0.030], 'betas': [0.4, 0.4, 0.4, 0.4],
                               'loss_ratios': [0.1, 0.2, 0.6, 1.0], 'repair_factors': [1.22, 1.18, 1.06, 3.15]},
            'A.98.Z.Z.Z.813': {'cat': '结构构件', 'edp': 'drift', 'cost': 1483, 
                               'medians': [0.007, 0.010, 0.016, 0.023], 'betas': [0.4, 0.4, 0.4, 0.4],
                               'loss_ratios': [0.1, 0.2, 0.5, 1.0], 'repair_factors': [1.20, 1.07, 1.15, 3.57]},
            'B.98.Z.Z.Z.161': {'cat': '位移敏感型非结构构件', 'edp': 'drift', 'cost': 150, 
                               'medians': [0.005, 0.010, 0.021], 'betas': [0.4, 0.3, 0.2],
                               'loss_ratios': [0.22, 0.50, 1.00], 'repair_factors': [1.15, 1.21, 1.27]},
            'B.01.B.B.A.001': {'cat': '加速度敏感型非结构构件', 'edp': 'accel', 'cost': 225, 
                               'medians': [1.17, 1.58, 1.82], 'betas': [0.25, 0.25, 0.25],
                               'loss_ratios': [0.10, 0.50, 1.00], 'repair_factors': [1.93, 1.49, 1.31]},
            'B.01.A.A.A.001': {'cat': '位移敏感型非结构构件', 'edp': 'drift', 'cost': 1520, 
                               'medians': [0.0338, 0.0383], 'betas': [0.4, 0.4],
                               'loss_ratios': [0.0, 1.00], 'repair_factors': [0.0, 1.67]}
        }

    def calc_84th(self, arr):
        arr = np.array(arr)
        valid = arr[arr > 1e-6] 
        if len(valid) == 0: return 0.0
        return float(np.exp(np.mean(np.log(valid)) + np.std(np.log(valid))))

    def expected_cost(self, cinfo, edp_val, qty, vol_disc):
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

    def evaluate_costs(self):
        f_c_str = {0:[], 1:[], 2:[]}
        f_c_dsp = {0:[], 1:[], 2:[]}
        f_c_acc = {0:[], 1:[], 2:[]}
        g_str_total, g_dsp_total, g_acc_total = [], [], []

        for w in range(11):
            g_s, g_d, g_a = 0, 0, 0
            for f in range(3):
                dx = self.drifts_x[f][w]
                dy = self.drifts_y[f][w]
                ax = self.accels_x[f][w] / 9.8   
                ay = self.accels_y[f][w] / 9.8 
                
                c_str, c_dsp, c_acc = 0, 0, 0
                
                for cid, cinfo in self.fragility_db.items():
                    qty = 2 if cid.startswith('A') else (44 if 'A.A.A.001' in cid else 37)
                    if cid.startswith('A'): vol = 1.0     
                    elif cid == 'B.98.Z.Z.Z.161': vol = 0.89
                    elif cid == 'B.01.B.B.A.001': vol = 0.96
                    else: vol = 0.90
                    
                    if cid.startswith('A'): 
                        c_str += self.expected_cost(cinfo, dx, qty, vol)
                        c_str += self.expected_cost(cinfo, dy, qty, vol)
                    elif cid == 'B.98.Z.Z.Z.161': 
                        c_dsp += self.expected_cost(cinfo, dx, qty, vol)
                        c_dsp += self.expected_cost(cinfo, dy, qty, vol)
                    elif cid == 'B.01.A.A.A.001': 
                        c_dsp += self.expected_cost(cinfo, np.sqrt(dx**2 + dy**2), qty, vol)
                    elif cid == 'B.01.B.B.A.001': 
                        c_acc += self.expected_cost(cinfo, np.sqrt(ax**2 + ay**2), qty, vol)
                        
                f_c_str[f].append(c_str)
                f_c_dsp[f].append(c_dsp)
                f_c_acc[f].append(c_acc)
                
                g_s += c_str
                g_d += c_dsp
                g_a += c_acc
                
            g_str_total.append(g_s)
            g_dsp_total.append(g_d)
            g_acc_total.append(g_a)

        final_res = {'floor_cost_details': []}
        
        for f, floor in enumerate(self.floor_info.index):
            fc_str = (self.calc_84th(f_c_str[f]) / self.bldg_total_cost) * 100
            fc_dsp = (self.calc_84th(f_c_dsp[f]) / self.bldg_total_cost) * 100
            fc_acc = (self.calc_84th(f_c_acc[f]) / self.bldg_total_cost) * 100
            fc_tot = fc_str + fc_dsp + fc_acc
            final_res['floor_cost_details'].append((floor, fc_tot, fc_acc, fc_dsp, fc_str))

        final_res['cost_struct'] = (self.calc_84th(g_str_total) / self.bldg_total_cost) * 100
        final_res['cost_disp'] = (self.calc_84th(g_dsp_total) / self.bldg_total_cost) * 100
        final_res['cost_accel'] = (self.calc_84th(g_acc_total) / self.bldg_total_cost) * 100
        final_res['cost_total'] = final_res['cost_struct'] + final_res['cost_disp'] + final_res['cost_accel']

        return final_res

    def generate_report(self):
        try:
            res = self.evaluate_costs()
        except Exception as e:
            print(f"数据处理过程出错：{e}")
            return
            
        print("="*80)
        print("二、建筑抗震韧性评级 - 费用专项计算核心模块")
        print("="*80)
        
        print("\n[修复费用评级]")
        print(f"总体修复费用: {res['cost_total']:.5f}%")
        print(f"(加速度敏感型: {res['cost_accel']:.5f}%, 位移敏感型: {res['cost_disp']:.5f}%, 结构构件: {res['cost_struct']:.5f}%)")
        
        print("\n各楼层修复类型对修复费用的贡献")
        print("-" * 75)
        print(f"{'楼层':<4} | {'总计费用(%)':<12} | {'加速度敏感型(%)':<15} | {'位移敏感型(%)':<15} | {'结构构件(%)'}")
        for row in res['floor_cost_details']:
            print(f"{row[0]:<6} | {row[1]:<14.5f} | {row[2]:<17.5f} | {row[3]:<17.5f} | {row[4]:.5f}")

if __name__ == "__main__":
    evaluator = CostResilienceEvaluator()
    evaluator.generate_report()