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
        if '楼层（层）' in self.floor_info.columns:
            self.floor_info.set_index('楼层（层）', inplace=True)
            self.floor_areas = self.floor_info['楼层面积（m^2）'].values
        else:
            self.floor_info.set_index('楼层', inplace=True)
            self.floor_areas = self.floor_info['楼层面积(m²)'].values
        self.num_floors = len(self.floor_info)
        
        self.struct_info = pd.read_excel(xls, '结构构件信息')
        self.nonstruct_info = pd.read_excel(xls, '非结构构件信息')
        df_seismic = pd.read_excel(xls, '地震信息1')
        self.seismic_level = str(df_seismic['地震水准'].iloc[0]).strip()

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
        self.edp_drift = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.drifts_x = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.drifts_y = edp_raw.iloc[[2, 3, 4], 23:34].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.edp_accel = edp_raw.iloc[[3, 4, 1], 12:23].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.accels_x = edp_raw.iloc[[3, 4, 1], 12:23].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.accels_y = edp_raw.iloc[[3, 4, 1], 34:45].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T

    def expand_edp_matrix(self, edp_matrix):
        """👉 终极修复：带有‘零值过滤护盾’的矩阵扩充引擎，防止 log(1e-6) 毒染分布"""
        # 找出完全有效的地震波（剔除那些因为报错或极小导致 < 1e-5 的无效行）
        valid_mask = np.all(edp_matrix > 1e-5, axis=1)
        valid_edp = edp_matrix[valid_mask]
        
        if len(valid_edp) < 2:
            # 容错：如果有效波太少，不进行扩充，直接对原波进行复用
            idx = np.random.choice(len(edp_matrix), self.num_simulations)
            return edp_matrix[idx]
            
        # 仅对真实、干净的数据进行对数联合正态拟合
        log_edp = np.log(valid_edp)
        mean_log = np.mean(log_edp, axis=0)
        cov_log = np.cov(log_edp, rowvar=False)
        
        eigvals, eigvecs = np.linalg.eigh(cov_log)
        eigvals[eigvals < 1e-8] = 1e-8 
        cov_log_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        sampled_log = np.random.multivariate_normal(mean_log, cov_log_pd, self.num_simulations)
        return np.exp(sampled_log)

    # ============ 标准对齐：损伤状态概率 / 楼层破坏等级 / 伤亡率 / 84%保证率 ============
    def _ds_probs(self, cinfo, edp_val):
        """标准5.2/5.3：构件处于各损伤状态(DS)的概率分布。返回长度=len(medians)+1。"""
        if edp_val <= 1e-6:
            return None
        m = np.array(cinfo['medians'], dtype=float)
        b = np.array(cinfo['betas'], dtype=float)
        e_probs = norm.cdf(np.log(edp_val / m) / b)
        probs = np.zeros(len(m) + 1)
        probs[0] = 1.0 - e_probs[0]
        for j in range(len(e_probs) - 1):
            probs[j + 1] = e_probs[j] - e_probs[j + 1]
        probs[-1] = e_probs[-1]
        probs = np.clip(probs, 0, 1)
        s = probs.sum()
        return probs / s if s > 0 else probs

    def _add_bucket(self, bucket, cinfo, qty, edp_val):
        """按工程量把构件各DS概率累加到占比桶。"""
        if qty <= 0:
            return bucket
        pr = self._ds_probs(cinfo, edp_val)
        if pr is None:
            return bucket
        L = max(len(bucket), len(pr))
        if len(bucket) < L:
            bucket = np.concatenate([bucket, np.zeros(L - len(bucket))])
        if len(pr) < L:
            pr = np.concatenate([pr, np.zeros(L - len(pr))])
        return bucket + pr * qty

    def _grade_structure(self, bucket):
        """标准8.2.2表3：结构构件楼层破坏等级(I-V)。bucket为按工程量加权后的各DS工程量。"""
        total = bucket.sum()
        if total <= 0:
            return 1
        p = bucket / total
        p2, p3, p4 = p[1], p[2], p[3]
        p5 = p[4] if len(p) > 4 else 0.0
        if p5 > 1e-9 or p4 > 0.10 or p3 > 0.20 or p2 > 0.50:
            return 5
        if p4 > 1e-9 or p3 > 0.10 or p2 > 0.20:
            return 4
        if p3 > 1e-9 or p2 > 0.10:
            return 3
        if p2 > 1e-9:
            return 2
        return 1

    def _grade_nonstruct(self, bucket):
        """标准8.2.2表3：可致伤亡非结构构件楼层破坏等级(I-V)。"""
        total = bucket.sum()
        if total <= 0:
            return 1
        p = bucket / total
        d1 = p[1] if len(p) > 1 else 0.0
        d2 = p[2] if len(p) > 2 else 0.0
        d3 = p[3] if len(p) > 3 else 0.0
        damaged = d2 + d3  # DS>=2 占比
        if d2 > 0.50 or d3 > 0.10:
            return 5
        if d3 <= 0.10 and damaged > 0.50 and d2 <= 0.50:
            return 4
        if damaged > 1e-9:
            return 3
        if d1 <= 0.10:
            return 1
        if d1 <= 0.30:
            return 2
        return 1

    def _casualty_rates(self, grade):
        """标准8.2.3/表4：楼层破坏等级 -> 名义受伤率、名义死亡率。"""
        rates = {
            1: (1.0 / 80000, 0.0),
            2: (1.0 / 20000, 0.0),
            3: (1.0 / 8000, 1.0 / 80000),
            4: (1.0 / 140, 1.0 / 800),
            5: (1.0 / 140, 1.0 / 800),
        }
        return rates.get(grade, (0.0, 0.0))

    def _p84(self, arr):
        """标准9.4.1/A.5.4：对蒙特卡洛样本取对数正态分布 84% 保证率拟合值。"""
        a = np.asarray(arr, dtype=float)
        a = a[a > 0]
        if len(a) < 2:
            return float(a.mean()) if len(a) > 0 else 0.0
        log_a = np.log(a)
        return float(np.exp(log_a.mean() + log_a.std()))

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
        
        self.bldg_comp_qtys = {}
        for cid in self.fragility_db.keys():
            tot = 0
            for f_name in self.floor_info.index:
                f_num = int(''.join(filter(str.isdigit, str(f_name))))
                qx, qy, qn = self.get_comp_qty(cid, f_num)
                tot += (qx + qy + qn)
            self.bldg_comp_qtys[cid] = tot

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
        
        results_time_s1 = {f: [] for f in range(self.num_floors)}
        results_time_s2 = {f: [] for f in range(self.num_floors)}
        results_injuries, results_deaths = [], []

        for sim_idx in range(self.num_simulations):
            sim_inj, sim_death = 0, 0
            sim_g_str_c, sim_g_dsp_c, sim_g_acc_c = 0.0, 0.0, 0.0
            
            for f_idx, floor in enumerate(self.floor_info.index):
                floor_num = int(''.join(filter(str.isdigit, str(floor))))
                floor_area = self.floor_areas[f_idx]
                pop_density = self.floor_info.loc[floor, '楼层人口密度'] if '楼层人口密度' in self.floor_info.columns else self.floor_info.loc[floor, '楼层人口密度(人/m²)']
                floor_influence = float(self.floor_info.loc[floor, '楼层影响系数']) if '楼层影响系数' in self.floor_info.columns else 1.0
                
                dx, dy = sim_drifts_x[sim_idx, f_idx], sim_drifts_y[sim_idx, f_idx]
                ax, ay = sim_accels_x[sim_idx, f_idx], sim_accels_y[sim_idx, f_idx]
                floor_drift = sim_drifts[sim_idx, f_idx]
                floor_accel = sim_accels[sim_idx, f_idx] / 9.8
                
                f_c_str, f_c_dsp, f_c_acc = 0.0, 0.0, 0.0
                q_w1, q_w3, q_w4, q_w5 = 0.0, 0.0, 0.0, 0.0
                max_ds_floor = 0
                # 👉 标准8.2.2：按工程量累计各DS占比
                struct_bucket = np.zeros(5)
                nonstruct_bucket = np.zeros(4)
                
                for cid, cinfo in self.fragility_db.items():
                    qty_x, qty_y, qty_none = self.get_comp_qty(cid, floor_num)
                    total_qty_all = qty_x + qty_y + qty_none
                    if total_qty_all <= 0: continue
                    
                    c_vol = self.get_volume_discount(self.bldg_comp_qtys[cid], cinfo['cost_vol_thresholds'], cinfo['cost_vol_factors'])
                    t_vol = self.get_volume_discount(self.bldg_comp_qtys[cid], cinfo['time_vol_thresholds'], cinfo['time_vol_factors'])
                    
                    def get_expected_metrics(qty, edp_val):
                        if qty <= 0 or edp_val <= 1e-6: return 0.0, 0.0, 0
                        e_probs = norm.cdf(np.log(edp_val / np.array(cinfo['medians'])) / np.array(cinfo['betas']))
                        probs = np.zeros(len(cinfo['medians']) + 1)
                        probs[0] = 1.0 - e_probs[0]
                        for j in range(len(e_probs) - 1): probs[j+1] = e_probs[j] - e_probs[j+1]
                        probs[-1] = e_probs[-1]
                        probs = np.clip(probs, 0, 1)
                        probs /= probs.sum()
                        
                        exp_c, exp_t = 0.0, 0.0
                        for ds_idx in range(len(cinfo['medians'])):
                            exp_c += probs[ds_idx+1] * cinfo['loss_ratios'][ds_idx] * cinfo['cost'] * qty * cinfo['repair_factors'][ds_idx] * c_vol
                            exp_t += probs[ds_idx+1] * cinfo['repair_times'][ds_idx] * qty * t_vol
                            
                        ds = np.random.choice(len(probs), p=probs)
                        return exp_c, exp_t, ds

                    c_x, t_x, ds_x = 0.0, 0.0, 0
                    c_y, t_y, ds_y = 0.0, 0.0, 0
                    c_n, t_n, ds_n = 0.0, 0.0, 0
                    
                    if cid.startswith('A') or cid == 'B.98.Z.Z.Z.161': 
                        c_x, t_x, ds_x = get_expected_metrics(qty_x, dx)
                        c_y, t_y, ds_y = get_expected_metrics(qty_y, dy)
                        c_n, t_n, ds_n = get_expected_metrics(qty_none, max(dx, dy))
                        
                        comp_c = c_x + c_y + c_n
                        if cid.startswith('A'): f_c_str += comp_c
                        else: f_c_dsp += comp_c
                        
                        if cid.startswith('A'): q_w1 += (t_x + t_y + t_n)
                        else: q_w4 += (t_x + t_y + t_n)
                        max_ds_floor = max(max_ds_floor, ds_x, ds_y, ds_n)
                        
                    elif cid == 'B.01.A.A.A.001': 
                        c_n, t_n, ds_n = get_expected_metrics(total_qty_all, max(dx, dy))
                        f_c_dsp += c_n
                        q_w3 += t_n
                        max_ds_floor = max(max_ds_floor, ds_n)
                    elif cid == 'B.01.B.B.A.001': 
                        c_n, t_n, ds_n = get_expected_metrics(total_qty_all, max(ax, ay))
                        f_c_acc += c_n
                        q_w5 += t_n
                        max_ds_floor = max(max_ds_floor, ds_n)

                # 👉 标准8.2.2：按工程量累计结构/非结构构件各DS概率分布
                for _cid, _cinfo in self.fragility_db.items():
                    _qx, _qy, _qn = self.get_comp_qty(_cid, floor_num)
                    if _cinfo['cat'] == '结构构件':
                        struct_bucket = self._add_bucket(struct_bucket, _cinfo, _qx, dx)
                        struct_bucket = self._add_bucket(struct_bucket, _cinfo, _qy, dy)
                        struct_bucket = self._add_bucket(struct_bucket, _cinfo, _qn, max(dx, dy))
                    else:
                        if _cinfo['edp'] == 'accel':
                            nonstruct_bucket = self._add_bucket(nonstruct_bucket, _cinfo, _qn, max(ax, ay))
                        else:
                            nonstruct_bucket = self._add_bucket(nonstruct_bucket, _cinfo, _qx, dx)
                            nonstruct_bucket = self._add_bucket(nonstruct_bucket, _cinfo, _qy, dy)
                            nonstruct_bucket = self._add_bucket(nonstruct_bucket, _cinfo, _qn, max(dx, dy))

                floor_pop = floor_area * pop_density

                # 👉 标准8.2.2/表3/表4：按构件损伤状态占比判定楼层破坏等级，再映射名义伤亡率
                floor_grade = max(self._grade_structure(struct_bucket), self._grade_nonstruct(nonstruct_bucket))
                inj_rate, death_rate = self._casualty_rates(floor_grade)
                sim_inj += floor_pop * inj_rate
                sim_death += floor_pop * death_rate

                f_c_str *= floor_influence
                f_c_dsp *= floor_influence
                f_c_acc *= floor_influence
                results_c_str[f_idx].append(f_c_str)
                results_c_dsp[f_idx].append(f_c_dsp)
                results_c_acc[f_idx].append(f_c_acc)
                
                sim_g_str_c += f_c_str
                sim_g_dsp_c += f_c_dsp
                sim_g_acc_c += f_c_acc

                # 👉 标准式(9)：单层工人最大容量 N_max = 0.026 * A_f（不含楼层影响系数）
                n_max = 0.026 * floor_area
                n_w1 = max(1e-6, min(n_max, (2.0 / 100) * floor_area))
                n_w3 = max(1e-6, min(n_max, (1.0 / 100) * floor_area))
                n_w4 = max(1e-6, min(n_max, (1.0 / 100) * floor_area))
                n_w5 = max(1e-6, min(n_max, (1.0 / 100) * floor_area))
                
                # 👉 标准7.2：阶段1=结构(同时)；阶段2=其余非结构，管线→隔断→吊顶依次累加
                results_time_s1[f_idx].append(q_w1 / n_w1)
                results_time_s2[f_idx].append(q_w3 / n_w3 + q_w4 / n_w4 + q_w5 / n_w5)

            results_g_str.append(sim_g_str_c)
            results_g_dsp.append(sim_g_dsp_c)
            results_g_acc.append(sim_g_acc_c)
            results_injuries.append(sim_inj)
            results_deaths.append(sim_death)

        # ===============================================
        # 模块 4：汇总（严格基于标准 1000 次样本平滑测算）
        # ===============================================
        final_res = {'floor_cost_details': [], 'floor_time_details': []}
        
        for f, floor in enumerate(self.floor_info.index):
            # 👉 标准修正：费用/时间/伤亡三项指标统一采用对数正态 84% 保证率拟合值（A.5.4）
            fc_str = (self._p84(results_c_str[f]) / self.bldg_total_cost) * 100
            fc_dsp = (self._p84(results_c_dsp[f]) / self.bldg_total_cost) * 100
            fc_acc = (self._p84(results_c_acc[f]) / self.bldg_total_cost) * 100
            fc_tot = fc_str + fc_dsp + fc_acc
            final_res['floor_cost_details'].append((floor, fc_tot, fc_acc, fc_dsp, fc_str))
            
            ft_s1 = self._p84(results_time_s1[f])
            ft_s2 = self._p84(results_time_s2[f])
            ft_tot = ft_s1 + ft_s2
            final_res['floor_time_details'].append((floor, ft_tot, ft_s1, ft_s2))

        final_res['cost_struct'] = (self._p84(results_g_str) / self.bldg_total_cost) * 100
        final_res['cost_disp'] = (self._p84(results_g_dsp) / self.bldg_total_cost) * 100
        final_res['cost_accel'] = (self._p84(results_g_acc) / self.bldg_total_cost) * 100
        final_res['cost_total'] = final_res['cost_struct'] + final_res['cost_disp'] + final_res['cost_accel']
        
        max_time_floor_detail = max(final_res['floor_time_details'], key=lambda x: x[1])
        final_res['time_total'] = max_time_floor_detail[1]
        final_res['time_s1_total'] = max_time_floor_detail[2]
        final_res['time_s2_total'] = max_time_floor_detail[3]

        if '楼层人口密度' in self.floor_info.columns:
            total_population = (self.floor_info['楼层面积（m^2）'] * self.floor_info['楼层人口密度']).sum()
        else:
            total_population = (self.floor_info['楼层面积(m²)'] * self.floor_info['楼层人口密度(人/m²)']).sum()
            
        final_res['injury_rate'] = (self._p84(results_injuries) / total_population) if total_population > 0 else 0
        final_res['death_rate'] = (self._p84(results_deaths) / total_population) if total_population > 0 else 0

        return final_res

    def generate_report(self):
        try:
            res = self.evaluate()
        except Exception as e:
            print(f"数据处理过程出错，请检查输入表格格式是否正确：{e}")
            return
            
        print("="*80)
        print("二、建筑抗震韧性评级 - 综合汇总报告 (全对齐终极版)")
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
            if res['injury_rate'] <= 1.0e-4 and res['death_rate'] <= 1.0e-5: star_casualty = "三星"
            elif res['injury_rate'] <= 1.0e-3 and res['death_rate'] <= 1.0e-4: star_casualty = "二星"
            else: star_casualty = "无星级"
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
    evaluator = TrueResilienceEvaluator('中震三层数据导出版.xls')
    evaluator.generate_report()
