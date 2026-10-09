import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# ==============================================================================
# 全量蒙特卡洛模拟 + 全链条数据提取审计日志（严格对应所有表格参数）
# ==============================================================================

class TrueResilienceEvaluator:
    
    def __init__(self, file_name='中震三层数据导出版.xls'):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))  
        self.file_path = os.path.join(self.data_dir, file_name)     
        self.num_simulations = 1000                                 
        self.load_data()                                            
        
    def load_data(self):
        print("="*80)
        print(f"🚀 正在启动系统，加载并执行全链条数据核对:\n{self.file_path}")
        print("="*80)
        xls = pd.ExcelFile(self.file_path)                          
        
        # 1. 建筑基本参数收集
        bldg_info1 = pd.read_excel(xls, '建筑信息1').iloc[0]        
        self.total_area = float(bldg_info1.get('建筑总面积（平方米）', 111.50)) 
        self.unit_cost = float(bldg_info1.get('单位造价（元/平方米）', 3000.0))  
        self.bldg_total_cost = self.total_area * self.unit_cost       
        
        print(f"\n[🔍 数据核对] 提取自【建筑信息1】表：")
        print(f"    ├─ 建筑总面积: {self.total_area} 平方米")
        print(f"    └─ 单位造价: {self.unit_cost} 元/平方米")

        # 2. 楼层几何及各层系数信息收集 (包含相关影响系数)
        self.floor_info = pd.read_excel(xls, '建筑信息2')             
        self.floor_info.set_index('楼层（层）', inplace=True)         
        self.num_floors = len(self.floor_info)                      
        self.floor_areas = self.floor_info['楼层面积（m^2）'].values      
        
        print(f"\n[🔍 数据核对] 提取自【建筑信息2】表 (逐层核心参数及系数)：")
        for f_name in self.floor_info.index:
            f_area = self.floor_info.loc[f_name, '楼层面积（m^2）']
            f_pop = self.floor_info.loc[f_name, '楼层人口密度']
            f_inf = float(self.floor_info.loc[f_name, '楼层影响系数']) if '楼层影响系数' in self.floor_info.columns else 1.0
            print(f"    ├─ 楼层 [{f_name}]: 面积 = {f_area} m^2 | 人口密度 = {f_pop} | 楼层影响系数 = {f_inf}")
        print(f"    └─ 评估有效物理总层数: {self.num_floors} 层")

        # 3. 构件布置清单收集
        self.struct_info = pd.read_excel(xls, '结构构件信息')           
        self.nonstruct_info = pd.read_excel(xls, '非结构构件信息')        
        
        print(f"\n[🔍 数据核对] 提取自【结构/非结构构件信息】表：")
        print(f"    ├─ 结构构件解析有效记录数: {len(self.struct_info)} 条")
        print(f"    └─ 非结构构件解析有效记录数: {len(self.nonstruct_info)} 条")

        # 4. 地震水准提取
        df_seismic = pd.read_excel(xls, '地震信息1')                  
        self.seismic_level = str(df_seismic['地震水准'].iloc[0]).strip() 
        
        print(f"\n[🔍 数据核对] 提取自【地震信息1】表：")
        print(f"    └─ 评估标定地震水准: {self.seismic_level}")

        # 5. 全参数易损性数据库挂载
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
        
        # 6. 工程需求参数响应矩阵 (EDP) 切片提取
        edp_raw = pd.read_excel(xls, '结构响应')                  
        
        self.edp_drift = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.drifts_x = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T 
        self.drifts_y = edp_raw.iloc[[2, 3, 4], 23:34].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T 
        
        print(f"\n[🔍 数据核对] 提取自【结构响应】表 (层间位移角)：")
        print(f"    ├─ Pandas提取行索引: [2, 3, 4] (对应表格物理1, 2, 3层)")
        print(f"    ├─ 构建完成 X 向位移角矩阵，形状: {self.drifts_x.shape} (行=11条地震波, 列=3个楼层)")
        print(f"    └─ 验算: 第1条地震波对应的 1, 2, 3 楼位移角为: {np.round(self.drifts_x[0, :], 5)}")

        self.edp_accel = edp_raw.iloc[[2, 3, 4, 1], 12:23].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.accels_x = edp_raw.iloc[[2, 3, 4, 1], 12:23].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T 
        self.accels_y = edp_raw.iloc[[2, 3, 4, 1], 34:45].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        
        print(f"\n[🔍 数据核对] 提取自【结构响应】表 (楼面加速度)：")
        print(f"    ├─ Pandas提取行索引: [2, 3, 4, 1] (对应表格物理1, 2, 3层 + 屋顶)")
        print(f"    ├─ 构建完成 X 向加速度矩阵，形状: {self.accels_x.shape} (行=11条地震波, 列=4个特征层)")
        print(f"    └─ 验算: 第1条地震波对应的 1, 2, 3, 顶楼 加速度为: {np.round(self.accels_x[0, :], 5)}")

        # 7. 全楼易损构件总库存统计
        self.bldg_comp_qtys = {}                                    
        for cid in self.fragility_db.keys():                        
            tot = 0                                                 
            for f_name in self.floor_info.index:                    
                f_num = int(''.join(filter(str.isdigit, str(f_name)))) 
                qx, qy, qn = self.get_comp_qty(cid, f_num)          
                tot += (qx + qy + qn)                               
            self.bldg_comp_qtys[cid] = tot                          
            
        print(f"\n[🔍 数据核对] 内部统计：全楼易损件总量大盘点 (用于匹配批量折扣率)：")
        for cid, qty in self.bldg_comp_qtys.items():
            # 智能判断单位：以 A 开头的通常是结构件（个/根），B 开头的是非结构件（面积/长度）
            unit = "个" if cid.startswith('A') else "平方米"
            print(f"    ├─ 构件 [{cid}] 全楼绝对装配总量: {qty} {unit}")
        print("="*80 + "\n")

    def expand_edp_matrix(self, edp_matrix):
        edp_matrix = np.clip(edp_matrix, a_min=1e-6, a_max=None)    
        log_edp = np.log(edp_matrix)                                
        mean_log = np.mean(log_edp, axis=0)                         
        cov_log = np.cov(log_edp, rowvar=False)                     
        eigvals, eigvecs = np.linalg.eigh(cov_log)                  
        eigvals[eigvals < 1e-8] = 1e-8                              
        cov_log_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T         
        sampled_log = np.random.multivariate_normal(mean_log, cov_log_pd, self.num_simulations) 
        return np.exp(sampled_log)                                  

    def get_comp_qty(self, cid, floor_num):
        s_match = self.struct_info[(self.struct_info['易损性编号'] == cid) & 
                                   (self.struct_info['起始楼层'] <= floor_num) & 
                                   (self.struct_info['终止楼层'] >= floor_num)]
        ns_match = self.nonstruct_info[(self.nonstruct_info['易损性编号'] == cid) & 
                                       (self.nonstruct_info['起始楼层'] <= floor_num) & 
                                       (self.nonstruct_info['终止楼层'] >= floor_num)]
        qty_x, qty_y, qty_none = 0, 0, 0                            
        for _, row in pd.concat([s_match, ns_match]).iterrows():    
            qty_x += row.get('X方向易损性数据', 0)                  
            qty_y += row.get('Y方向易损性数据', 0)                  
            qty_none += row.get('无方向易损性数据', 0)              
        return qty_x, qty_y, qty_none                               

    def get_volume_discount(self, qty, thresholds, factors):
        if qty <= thresholds[0]: return factors[0]                  
        elif qty >= thresholds[1]: return factors[1]                
        else:
            ratio = (qty - thresholds[0]) / (thresholds[1] - thresholds[0])
            return factors[0] - ratio * (factors[0] - factors[1])

    def evaluate(self):
        np.random.seed(42)                                          
        
        sim_drifts = self.expand_edp_matrix(self.edp_drift)
        sim_accels = self.expand_edp_matrix(self.edp_accel)
        sim_drifts_x = self.expand_edp_matrix(self.drifts_x)
        sim_drifts_y = self.expand_edp_matrix(self.drifts_y)
        sim_accels_x = self.expand_edp_matrix(self.accels_x) / 9.8  
        sim_accels_y = self.expand_edp_matrix(self.accels_y) / 9.8 
        
        results_c_str = {f: [] for f in range(self.num_floors)}     
        results_c_dsp = {f: [] for f in range(self.num_floors)}     
        results_c_acc = {f: [] for f in range(self.num_floors)}     
        results_g_str, results_g_dsp, results_g_acc = [], [], []    
        results_g_tot = []                                          
        
        results_time_s1 = {f: [] for f in range(self.num_floors)}   
        results_time_s2 = {f: [] for f in range(self.num_floors)}   
        results_injuries, results_deaths = [], []                   

        print("-> 正在执行 1000 次全蒙特卡洛模拟 (时间、伤亡及【修复费用】联合多项随机抽样) ...")
        for sim_idx in range(self.num_simulations):                 
            sim_inj, sim_death = 0, 0                               
            sim_g_str_c, sim_g_dsp_c, sim_g_acc_c = 0.0, 0.0, 0.0   
            
            f_cost_struct_sim = {f: 0.0 for f in range(self.num_floors)} 
            f_cost_dsp_sim = {f: 0.0 for f in range(self.num_floors)}    
            f_cost_acc_sim = {f: 0.0 for f in range(self.num_floors)}    

            for f_idx, floor in enumerate(self.floor_info.index):   
                floor_num = int(''.join(filter(str.isdigit, str(floor)))) 
                floor_area = self.floor_areas[f_idx]                
                pop_density = self.floor_info.loc[floor, '楼层人口密度'] 
                floor_influence = float(self.floor_info.loc[floor, '楼层影响系数']) if '楼层影响系数' in self.floor_info.columns else 1.0

                dx = sim_drifts_x[sim_idx, f_idx]                   
                dy = sim_drifts_y[sim_idx, f_idx]                   
                ax = sim_accels_x[sim_idx, f_idx]                   
                ay = sim_accels_y[sim_idx, f_idx]                   
                
                f_c_str, f_c_dsp, f_c_acc = 0.0, 0.0, 0.0           
                q_w1, q_w3, q_w4, q_w5 = 0.0, 0.0, 0.0, 0.0         
                max_ds_floor = 0                                    
                
                for cid, cinfo in self.fragility_db.items():
                    qty_x, qty_y, qty_none = self.get_comp_qty(cid, floor_num) 
                    total_qty_all = qty_x + qty_y + qty_none        
                    if total_qty_all <= 0: continue                 
                    
                    c_vol_disc = self.get_volume_discount(self.bldg_comp_qtys[cid], cinfo['cost_vol_thresholds'], cinfo['cost_vol_factors'])
                    t_vol_disc = self.get_volume_discount(self.bldg_comp_qtys[cid], cinfo['time_vol_thresholds'], cinfo['time_vol_factors'])
                    
                    def mc_sample_cost_and_time(qty_sub, edp_sub, comp_info, c_disc, t_disc):
                        if qty_sub <= 0 or edp_sub <= 1e-6: return 0.0, 0.0, 0  
                        n_ds = len(comp_info['medians'])                        
                        e_probs = norm.cdf(np.log(edp_sub / np.array(comp_info['medians'])) / np.array(comp_info['betas']))
                        
                        probs = np.zeros(n_ds + 1)
                        probs[0] = 1.0 - e_probs[0]                             
                        for j in range(n_ds - 1): probs[j+1] = e_probs[j] - e_probs[j+1] 
                        probs[-1] = e_probs[-1]                                 
                        probs = np.clip(probs, 0, 1)                            
                        probs /= probs.sum()                                    
                        
                        ds_counts = np.random.multinomial(int(round(qty_sub)), probs)
                        
                        batch_cost, batch_time = 0.0, 0.0                       
                        max_ds_local = 0                                        
                        
                        for ds_idx in range(1, n_ds + 1):                       
                            count = ds_counts[ds_idx]                           
                            if count > 0:
                                max_ds_local = max(max_ds_local, ds_idx)        
                                batch_cost += count * comp_info['loss_ratios'][ds_idx-1] * comp_info['cost'] * comp_info['repair_factors'][ds_idx-1] * c_disc
                                batch_time += count * comp_info['repair_times'][ds_idx-1] * t_disc
                                
                        return batch_cost, batch_time, max_ds_local             

                    cost_x, time_x, ds_x = 0.0, 0.0, 0                          
                    cost_y, time_y, ds_y = 0.0, 0.0, 0                          
                    cost_n, time_n, ds_n = 0.0, 0.0, 0                          

                    if cid.startswith('A') or cid == 'B.98.Z.Z.Z.161': 
                        if qty_x > 0: cost_x, time_x, ds_x = mc_sample_cost_and_time(qty_x, dx, cinfo, c_vol_disc, t_vol_disc)
                        if qty_y > 0: cost_y, time_y, ds_y = mc_sample_cost_and_time(qty_y, dy, cinfo, c_vol_disc, t_vol_disc)
                        if qty_none > 0: cost_n, time_n, ds_n = mc_sample_cost_and_time(qty_none, max(dx, dy), cinfo, c_vol_disc, t_vol_disc) 
                    elif cid == 'B.01.A.A.A.001': 
                        if total_qty_all > 0: cost_n, time_n, ds_n = mc_sample_cost_and_time(total_qty_all, np.sqrt(dx**2 + dy**2), cinfo, c_vol_disc, t_vol_disc)
                    elif cid == 'B.01.B.B.A.001': 
                        if total_qty_all > 0: cost_n, time_n, ds_n = mc_sample_cost_and_time(total_qty_all, np.sqrt(ax**2 + ay**2), cinfo, c_vol_disc, t_vol_disc)

                    tot_cost_comp = cost_x + cost_y + cost_n                    
                    tot_time_comp = time_x + time_y + time_n                    
                    max_ds_comp = max(ds_x, ds_y, ds_n)                         
                    max_ds_floor = max(max_ds_floor, max_ds_comp)               

                    if cid.startswith('A'):
                        f_c_str += tot_cost_comp                                
                        q_w1 += tot_time_comp                                   
                    else:
                        if cinfo['edp'] == 'accel':
                            f_c_acc += tot_cost_comp                            
                            q_w5 += tot_time_comp                               
                        else:
                            f_c_dsp += tot_cost_comp                            
                            if cid == 'B.98.Z.Z.Z.161': q_w4 += tot_time_comp   
                            elif cid == 'B.01.A.A.A.001': q_w3 += tot_time_comp 

                f_c_str *= floor_influence
                f_c_dsp *= floor_influence
                f_c_acc *= floor_influence

                f_cost_struct_sim[f_idx] = f_c_str
                f_cost_dsp_sim[f_idx] = f_c_dsp
                f_cost_acc_sim[f_idx] = f_c_acc

                sim_g_str_c += f_c_str
                sim_g_dsp_c += f_c_dsp
                sim_g_acc_c += f_c_acc

                floor_pop = floor_area * pop_density                            
                if max_ds_floor >= 4:                                           
                    sim_inj += floor_pop * (1/140)
                    sim_death += floor_pop * (1/800)
                elif max_ds_floor == 3:                                         
                    sim_inj += floor_pop * (1/8000)
                    sim_death += floor_pop * (1/80000)
                elif max_ds_floor == 2:                                         
                    sim_inj += floor_pop * (1/20000)
                elif max_ds_floor == 1:                                         
                    sim_inj += floor_pop * (1/80000)

                n_max = (0.026 * floor_area) / floor_influence
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

            for f in range(self.num_floors):
                results_c_str[f].append(f_cost_struct_sim[f])
                results_c_dsp[f].append(f_cost_dsp_sim[f])
                results_c_acc[f].append(f_cost_acc_sim[f])

            results_g_str.append(sim_g_str_c)
            results_g_dsp.append(sim_g_dsp_c)
            results_g_acc.append(sim_g_acc_c)
            results_g_tot.append(sim_g_str_c + sim_g_dsp_c + sim_g_acc_c) 
            
            results_injuries.append(sim_inj)                              
            results_deaths.append(sim_death)                              

        print("-> 1000 次全蒙特卡洛引擎仿真结束，正在提取均值期望数据 ...")
        final_res = {'floor_cost_details': [], 'floor_time_details': []}
        
        for f, floor in enumerate(self.floor_info.index):
            fc_str = (np.mean(results_c_str[f]) / self.bldg_total_cost) * 100
            fc_dsp = (np.mean(results_c_dsp[f]) / self.bldg_total_cost) * 100
            fc_acc = (np.mean(results_c_acc[f]) / self.bldg_total_cost) * 100
            fc_tot = fc_str + fc_dsp + fc_acc
            final_res['floor_cost_details'].append((floor, fc_tot, fc_acc, fc_dsp, fc_str))
            
            ft_s1 = np.mean(results_time_s1[f])
            ft_s2 = np.mean(results_time_s2[f])
            ft_tot = ft_s1 + ft_s2
            final_res['floor_time_details'].append((floor, ft_tot, ft_s1, ft_s2))

        final_res['cost_struct'] = (np.mean(results_g_str) / self.bldg_total_cost) * 100
        final_res['cost_disp'] = (np.mean(results_g_dsp) / self.bldg_total_cost) * 100
        final_res['cost_accel'] = (np.mean(results_g_acc) / self.bldg_total_cost) * 100
        final_res['cost_total'] = (np.mean(results_g_tot) / self.bldg_total_cost) * 100
        
        max_time_floor_detail = max(final_res['floor_time_details'], key=lambda x: x[1])
        final_res['time_total'] = max_time_floor_detail[1]
        final_res['time_s1_total'] = max_time_floor_detail[2]
        final_res['time_s2_total'] = max_time_floor_detail[3]

        total_population = (self.floor_info['楼层面积（m^2）'] * self.floor_info['楼层人口密度']).sum()
        final_res['injury_rate'] = (np.mean(results_injuries) / total_population)  if total_population > 0 else 0
        final_res['death_rate'] = (np.mean(results_deaths) / total_population)  if total_population > 0 else 0

        return final_res                                            

    def generate_report(self):
        try:
            res = self.evaluate()
        except Exception as e:
            print(f"致命异常：数据崩溃，请审查输入表单！详情：{e}")
            return
            
        print("="*80)
        print("二、建筑抗震韧性评级 - 综合汇总报告 (1000次全蒙特卡洛均值期望版)")
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

        print(f"\n[当前评估地震水准]: {self.seismic_level}")
        
        if '设防' in self.seismic_level:
            star_cost = "一星" if res['cost_total'] <= 10.0 else "无星级"
            star_time = "一星" if res['time_total'] <= 30.0 else "无星级"
            star_casualty = "一星" if (res['injury_rate'] <= 1.0e-3 and res['death_rate'] <= 1.0e-4) else "无星级"
            
        elif '罕遇' in self.seismic_level:
            if res['cost_total'] <= 5.0: star_cost = "三星"
            elif res['cost_total'] <= 10.0: star_cost = "二星"
            else: star_cost = "无星级"
            
            if res['time_total'] <= 7.0: star_time = "三星"
            elif res['time_total'] <= 30.0: star_time = "二星"
            else: star_time = "无星级"
            
            if res['injury_rate'] <= 1.0e-4 and res['death_rate'] <= 1.0e-5:
                star_casualty = "三星"
            elif res['injury_rate'] <= 1.0e-3 and res['death_rate'] <= 1.0e-4:
                star_casualty = "二星"
            else: 
                star_casualty = "无星级"
            
        else:
            star_cost, star_time, star_casualty = "无星级", "无星级", "无星级"

        star_map = {"三星": 3, "二星": 2, "一星": 1, "无星级": 0}
        min_star_val = min(star_map[star_cost], star_map[star_time], star_map[star_casualty])
        
        reverse_map = {3: "三星", 2: "二星", 1: "一星", 0: "无星级"}
        overall_star = reverse_map[min_star_val]

        print("\n[最终评级]")
        print(f"人员损失等级：\t{star_casualty}")
        print(f"修复时间等级：\t{star_time}")
        print(f"修复费用等级：\t{star_cost}")
        print(f"抗震韧性等级评价：\t{overall_star}")
        print("="*80)

if __name__ == "__main__":
    evaluator = TrueResilienceEvaluator()
    evaluator.generate_report()