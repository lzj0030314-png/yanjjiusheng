import os                             # 导入操作系统接口模块，用于处理文件物理路径
import numpy as np                    # 导入科学计算核心库，用于矩阵、向量运算及蒙特卡洛随机数生成
import pandas as pd                   # 导入数据分析库，用于高速读取和清洗 Excel 结构表格
from scipy.stats import norm          # 导入科学统计库中的正态分布模块，用于计算超越概率与累积分布函数(CDF)

# ==============================================================================
# 依据图 A.1 流程及 A.5.1~A.5.3 强制要求重构的【全量纯蒙特卡洛模拟版】
# 特性保留：1. 编号对齐（激活 161 隔墙费用） 2. X/Y 方向精准解耦（还原构件真实受力）
# ==============================================================================

class TrueResilienceEvaluator:
    """建筑抗震韧性全蒙特卡洛模拟评估主类"""
    
    def __init__(self, file_name='中震三层数据导出版.xls'):
        """初始化构造函数，设定蒙特卡洛次数并按国标流程启动各项初始化"""
        self.data_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的绝对目录
        self.file_path = os.path.join(self.data_dir, file_name)     # 拼接出目标 Excel 文件的完整路径
        self.num_simulations = 1000                                 # 国标规范硬性规定：蒙特卡洛模拟次数不应少于 1000 次
        
        self.load_data()                                            # 执行数据加载与模型挂载流水线
        
    def load_data(self):
        """流程：【建筑信息收集】->【构件种类和数量统计】->【原始工程需求参数矩阵生成】"""
        print(f"正在加载真实数据文件: {self.file_path} ...")           # 控制台打印加载日志
        xls = pd.ExcelFile(self.file_path)                          # 将 Excel 文件整体加载进内存对象
        
        # 1. 建筑基本参数收集
        bldg_info1 = pd.read_excel(xls, '建筑信息1').iloc[0]        # 读取第一张建筑信息表的第一行数据
        self.total_area = float(bldg_info1.get('建筑总面积（平方米）', 111.50)) # 提取大楼总建筑面积
        self.unit_cost = float(bldg_info1.get('单位造价（元/平方米）', 3000.0))  # 提取大楼重置单位造价
        self.bldg_total_cost = self.total_area * self.unit_cost       # 计算建筑总造价基准，作为计算最后损失百分比的分母
        
        # 2. 楼层几何及人口分布信息收集
        self.floor_info = pd.read_excel(xls, '建筑信息2')             # 读取楼层详细参数表
        self.floor_info.set_index('楼层（层）', inplace=True)         # 将“楼层（层）”一列设为 Pandas 数据框的检索索引
        self.num_floors = len(self.floor_info)                      # 统计物理大楼的总体层数
        self.floor_areas = self.floor_info['楼层面积（m^2）'].values      # 将各层面积转化为 Numpy 数组方便后续高速计算
        
        # 3. 构件布置清单收集
        self.struct_info = pd.read_excel(xls, '结构构件信息')           # 读取结构构件(柱、墙等)的逐层布置清册
        self.nonstruct_info = pd.read_excel(xls, '非结构构件信息')        # 读取非结构构件(吊顶、幕墙等)的逐层布置清册

        # 4. 地震水准提取
        df_seismic = pd.read_excel(xls, '地震信息1')                  # 读取包含地震评估烈度的配置表
        self.seismic_level = str(df_seismic['地震水准'].iloc[0]).strip() # 提取水准名称并去除首尾空格

        # 5. 全参数易损性数据库挂载（🌟 重点保留了 B.98.Z.Z.Z.161 的编号对齐，成功激活隔墙费用）
        self.fragility_db = {
            'A.98.Z.Z.Z.814': {
                'cat': '结构构件', 'edp': 'drift', 'cost': 1500, 
                'medians': [0.005, 0.010, 0.020, 0.030], 'betas': [0.4, 0.4, 0.4, 0.4],
                'loss_ratios': [0.1, 0.2, 0.6, 1.0], 'repair_factors': [1.22, 1.18, 1.06, 3.15],
                'repair_times': [3.8, 5.6, 11.3, 25.0],
                # GB/T 38591-2020 表C.10/C.12: 三档阶跃式 ≤10 / 11~49 / ≥50
                'cost_vol_thresholds': [10.0, 49.0], 'cost_vol_factors': [1.0, 0.85, 0.85],
                'time_vol_thresholds': [10.0, 49.0], 'time_vol_factors': [1.0, 0.75, 0.75]
            },
            'A.98.Z.Z.Z.813': {
                'cat': '结构构件', 'edp': 'drift', 'cost': 1483, 
                'medians': [0.007, 0.010, 0.016, 0.023], 'betas': [0.4, 0.4, 0.4, 0.4],
                'loss_ratios': [0.1, 0.2, 0.5, 1.0], 'repair_factors': [1.20, 1.15, 1.07, 3.57],
                'repair_times': [2.6, 6.2, 9.4, 27.8],
                # GB/T 38591-2020 表C.10/C.12: 三档阶跃式
                'cost_vol_thresholds': [10.0, 49.0], 'cost_vol_factors': [1.0, 0.85, 0.85],
                'time_vol_thresholds': [10.0, 49.0], 'time_vol_factors': [1.0, 0.75, 0.75]
            },
            'B.98.Z.Z.Z.161': { # 填充墙(非承重墙)：位移敏感型非结构
                'cat': '位移敏感型非结构构件', 'edp': 'drift', 'cost': 150, 
                'medians': [0.005, 0.010, 0.021], 'betas': [0.4, 0.3, 0.2],
                'loss_ratios': [0.22, 0.50, 1.00], 'repair_factors': [1.15, 1.21, 1.27],
                'repair_times': [0.015, 0.029, 0.056],
                # GB/T 38591-2020 表E.7: 三档阶跃式 1件→1.0, 2~9件→0.89, ≥10件→0.89
                'cost_vol_thresholds': [1.0, 9.0], 'cost_vol_factors': [1.0, 0.89, 0.89],
                'time_vol_thresholds': [1.0, 9.0], 'time_vol_factors': [1.0, 0.89, 0.89]
            },
            'B.01.B.B.A.001': { # 吊顶：加速度敏感型
                'cat': '加速度敏感型非结构构件', 'edp': 'accel', 'cost': 225, 
                'medians': [1.17, 1.58, 1.82], 'betas': [0.25, 0.25, 0.25],
                'loss_ratios': [0.10, 0.50, 1.00], 'repair_factors': [1.93, 1.49, 1.31],
                'repair_times': [0.015, 0.116, 0.240],
                # GB/T 38591-2020 表E.7(吊顶类): 三档阶跃式 1→1.0, 2~9→0.96, ≥10→0.96
                'cost_vol_thresholds': [1.0, 9.0], 'cost_vol_factors': [1.0, 0.96, 0.96],
                'time_vol_thresholds': [1.0, 9.0], 'time_vol_factors': [1.0, 0.96, 0.96]
            },
            'B.01.A.A.A.001': { # 幕墙：位移敏感型
                'cat': '位移敏感型非结构构件', 'edp': 'drift', 'cost': 1520, 
                'medians': [0.0338, 0.0383], 'betas': [0.4, 0.4],
                'loss_ratios': [0.0, 1.00], 'repair_factors': [0.0, 1.67],
                'repair_times': [0.0, 0.216],
                # GB/T 38591-2020 表E.7(幕墙/外墙类): 三档阶跃式 1→1.0, 2~9→0.89, ≥10→0.89
                'cost_vol_thresholds': [1.0, 9.0], 'cost_vol_factors': [1.0, 0.89, 0.89],
                'time_vol_thresholds': [1.0, 9.0], 'time_vol_factors': [1.0, 0.89, 0.89]
            }
        }
        
        # 6. 提取原始工程需求参数响应矩阵 (EDP 矩阵)
        # ⚠️ 行索引必须精确锚定：位移角使用 [1,2,3楼]=[行2,3,4]；加速度取 [1,2,3楼, 屋顶] = [2,3,4,1]
        edp_raw = pd.read_excel(xls, '结构响应')                      # 读取结构响应包络表
        # 位移角(drift)：形状 (11次模拟, 3层)
        self.edp_drift = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.drifts_x = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.drifts_y = edp_raw.iloc[[2, 3, 4], 23:34].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        # 加速度(accel)：形状 (11次模拟, 4层含屋顶)，与楼层数量对齐时只取前num_floors行
        self.edp_accel = edp_raw.iloc[[2, 3, 4, 1], 12:23].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.accels_x = edp_raw.iloc[[2, 3, 4, 1], 12:23].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.accels_y = edp_raw.iloc[[2, 3, 4, 1], 34:45].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T


        # 提前进行全楼构件大盘点，查明整栋楼每种型号的总库存，供评估“批量折减系数”使用
        self.bldg_comp_qtys = {}                                    # 初始化大楼总构件存放字典
        for cid in self.fragility_db.keys():                        # 遍历易损库中每一个型号
            tot = 0                                                 # 计数器归零
            for f_name in self.floor_info.index:                    # 乘坐电梯逐层扫描
                f_num = int(''.join(filter(str.isdigit, str(f_name)))) # 解析获取纯数字的楼层号
                qx, qy, qn = self.get_comp_qty(cid, f_num)          # 取出本层装配的该构件数量
                tot += (qx + qy + qn)                               # 加进库存总计
            self.bldg_comp_qtys[cid] = tot                          # 锁定该构件的全楼保有量

    def expand_edp_matrix(self, edp_matrix):
        """【工程需求参数矩阵扩充】（将 11 条地震波响应扩充至 1000 条蒙特卡洛全集样本）"""
        edp_matrix = np.clip(edp_matrix, a_min=1e-6, a_max=None)    # 数值托底护盾：防止数值为0导致求自然对数崩溃
        log_edp = np.log(edp_matrix)                                # 将物理空间变量强制转换为对数正态空间分布
        
        # 行(axis=0)代表11条波的不同实验结果，计算对数空间的样本均值中心
        mean_log = np.mean(log_edp, axis=0)                         
        # 计算楼层间的对数协方差矩阵（捕获地震摇晃时的层间物理耦合关系）
        cov_log = np.cov(log_edp, rowvar=False)                     
        
        eigvals, eigvecs = np.linalg.eigh(cov_log)                  # 通过特征值分解提纯并诊断协方差矩阵
        eigvals[eigvals < 1e-8] = 1e-8                              # 数学干预：强行修正由于样本量太少引发的不满秩（负数特征值）
        cov_log_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T         # 重新组装出严格的半正定协方差矩阵供底层数学引擎使用
        
        # 利用修缮好的协方差参数，在多元正态分布中抛出 1000 次具有联动规律的“多维骰子”
        sampled_log = np.random.multivariate_normal(mean_log, cov_log_pd, self.num_simulations) 
        return np.exp(sampled_log)                                  # 指数反推，将这 1000 次模拟数据弹回真实的物理空间

    def calc_84th(self, arr):
        """【统计处理】84% 保证率分位值 (GB/T 38591-2020 §9.4.1)
        对全部样本直接取经验84分位，不过滤零值，避免系统性高估。
        """
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0: return 0.0
        return float(np.percentile(arr, 84))

    def get_comp_qty(self, cid, floor_num):
        """提取指定楼层中真实分布的构件数量（并拆分方向性）"""
        # 利用 pandas 条件检索锁定结构构件表中的对应楼层目标行
        s_match = self.struct_info[(self.struct_info['易损性编号'] == cid) & 
                                   (self.struct_info['起始楼层'] <= floor_num) & 
                                   (self.struct_info['终止楼层'] >= floor_num)]
        # 同理锁定非结构构件表
        ns_match = self.nonstruct_info[(self.nonstruct_info['易损性编号'] == cid) & 
                                       (self.nonstruct_info['起始楼层'] <= floor_num) & 
                                       (self.nonstruct_info['终止楼层'] >= floor_num)]
        
        qty_x, qty_y, qty_none = 0, 0, 0                            # 初始化计数器
        for _, row in pd.concat([s_match, ns_match]).iterrows():    # 把两张查出的表拼在一起逐行消化
            # 🌟 特性保留：彻底抛弃粗暴混合，精准提取属于 X方向、Y方向 以及 无方向 的构件数
            qty_x += row.get('X方向易损性数据', 0)                  
            qty_y += row.get('Y方向易损性数据', 0)                  
            qty_none += row.get('无方向易损性数据', 0)              
            
        return qty_x, qty_y, qty_none                               # 将拆解完毕的数据呈递给受力引擎

    def get_volume_discount(self, qty, thresholds, factors):
        """批量修缮工程量折减系数 (GB/T 38591-2020 表C.10/C.12/E.7)
        阶跃式取值(不插值): thresholds=[t1, t2] 将数量分为三档
        ≤t1 → factors[0];  (t1, t2] → factors[1];  >t2 → factors[2]
        """
        if qty <= thresholds[0]:
            return factors[0]
        elif qty <= thresholds[1]:
            return factors[1]
        else:
            return factors[2]

    # ==========================================================================
    # 表3 辅助：按DS占比判定楼层破坏等级 0..4 (I..V)
    # 严格依据 GB/T 38591-2020 §8.2.2 表3：
    #   - 判定逻辑：自最严重等级Ⅴ向Ⅰ"阈值上限≤AND"逐级下探，命中即停；
    #   - 仅"Ⅴ级"行原文使用三个并列"或"关系，其余行内逗号均读作"且"。
    # ==========================================================================
    def _struct_floor_level(self, p):
        """表3 结构构件：p = DS0..DS4 的占比数组 (长度5, p[0]=DS0 … p[4]=DS4)"""
        ds2, ds3, ds4 = float(p[2]), float(p[3]), float(p[4])
        # Ⅴ (严重破坏) 行内 OR 三条件：
        #   "DS2>50%且DS3>20%" / "DS2>50%且DS4>10%" / "DS3>20%且DS4>10%"
        if (ds2 > 0.50 and ds3 > 0.20) or (ds2 > 0.50 and ds4 > 0.10) or (ds3 > 0.20 and ds4 > 0.10):
            return 4
        # Ⅳ (中等破坏)：DS2≤50% 且 DS3≤20% 且 DS4≤10%  (行内AND，上限阈值法)
        if ds2 <= 0.50 and ds3 <= 0.20 and ds4 <= 0.10:
            # 若DS2>20% 或 DS3>10% 或 DS4>0，则属中等破坏(不再是轻微)
            if ds2 > 0.20 or ds3 > 0.10 or ds4 > 0.0:
                return 3
        # Ⅲ (轻度破坏)：DS2≤20% 且 DS3≤10% 且 DS4=0
        if ds2 <= 0.20 and ds3 <= 0.10 and ds4 <= 0.0:
            if ds2 > 0.10 or ds3 > 0.0:
                return 2
        # Ⅱ (轻微破坏)：DS2≤10% 且 DS3=0 且 DS4=0 且 存在DS2
        if ds2 <= 0.10 and ds3 <= 0.0 and ds4 <= 0.0 and ds2 > 0.0:
            return 1
        # Ⅰ (完好)：DS0=100%
        return 0

    def _nonstruct_floor_level(self, p):
        """表3 可致伤亡非结构构件：p = DS0..DS3 占比 (长度4, p[0]=DS0 … p[3]=DS3)"""
        d1, d2, d3 = float(p[1]), float(p[2]), float(p[3])
        damaged = d1 + d2 + d3
        # Ⅴ：行内 OR — "(DS2>50%且DS3>10%)" / "DS2>50%" / "DS3>10%"
        if (d2 > 0.50 and d3 > 0.10) or d2 > 0.50 or d3 > 0.10:
            return 4
        # Ⅳ：震损构件占比≤50%+ 上限50%? → 表Ⅳ行：>50% 且 DS2≤50% 且 DS3≤10%
        if damaged > 0.50 and d2 <= 0.50 and d3 <= 0.10:
            return 3
        # Ⅲ：DS1≤50% 且 DS2≤10% 且 DS3=0（震损≤50%）
        if d1 <= 0.50 and d2 <= 0.10 and d3 <= 0.0:
            # 轻度破坏：只要出现DS2 (>0但≤10%) 或 DS1 30%~50% 档
            if d2 > 0.0 or d1 > 0.30:
                return 2
        # Ⅱ：DS1≤30% 且 DS2=0 且 DS3=0 (且DS1>10%，否则落入Ⅰ)
        if d2 <= 0.0 and d3 <= 0.0 and d1 <= 0.30 and d1 > 0.10:
            return 1
        # Ⅰ：DS1≤10% 且 不出现DS>1构件
        if d1 <= 0.10 and d2 <= 0.0 and d3 <= 0.0:
            return 0
        # 兜底：凡不进入Ⅰ/Ⅱ，但又未达Ⅳ/Ⅴ的非致命DS1 30%~50%带DS2的情形 → Ⅲ
        return 2

    # 表4 名义伤亡率 (破坏等级 0..V → (受伤率, 死亡率))
    _CASUALTY_RATE = {
        0: (0.0,       0.0),
        1: (1/80000,   0.0),
        2: (1/20000,   0.0),
        3: (1/8000,    1/80000),
        4: (1/140,     1/800),
    }

    def evaluate(self):
        """
        中枢控制台：1000 次蒙特卡洛模拟 — GB/T 38591-2020 纯标准版
        ┌───────────────────────────────────────────────────────────────────┐
        │ F1 修复时间:  N_max = 0.026·A_f（式9不除ζ）；Q×ζ（式6, ζ作用在工时）│
        │ F2 伤亡:      Σ(A·ζ·rate) / Σ(A·ζ)   （式17/18 加权面积归一化）  │
        │ F3 费用&工时: 解析期望值 ΣP(DS)·指标（§A.5.2 仅EDP为输入随机量）  │
        │ F4 时间汇总:  每次MC内先max各层T得到建筑T → 再P84(建筑T)（A.5.4）│
        │ 保留:        8工种S1/S2、n_S、表3 DS占比、阶跃式折减、84%不筛选0 │
        └───────────────────────────────────────────────────────────────────┘
        """
        np.random.seed(42)

        # EDP扩充
        sim_drifts_x = self.expand_edp_matrix(self.drifts_x)
        sim_drifts_y = self.expand_edp_matrix(self.drifts_y)
        sim_accels_x = self.expand_edp_matrix(self.accels_x) / 9.8
        sim_accels_y = self.expand_edp_matrix(self.accels_y) / 9.8

        # 表C.13: n_S 修复时间层数调整系数
        nf = self.num_floors
        if nf >= 12:   n_S = 1.10
        elif nf >= 7:  n_S = 1.08
        elif nf >= 4:  n_S = 1.05
        else:          n_S = 1.00

        # 表1: 8工种密度系数 (人/100m²)
        W_DENSITY = {
            'W1': 2.0, 'W2': 2.0, 'W3': 1.0, 'W4': 1.0,
            'W5': 1.0, 'W6': 3.0, 'W7': 2.0, 'W8': 2.0,
        }
        NMAX_COEF = 0.026  # 式(9)

        # 结果容器
        results_c_str = {f: [] for f in range(nf)}
        results_c_dsp = {f: [] for f in range(nf)}
        results_c_acc = {f: [] for f in range(nf)}
        results_g_str, results_g_dsp, results_g_acc = [], [], []
        results_g_tot = []
        results_time_s1 = {f: [] for f in range(nf)}
        results_time_s2 = {f: [] for f in range(nf)}
        results_time_bldg = []     # F4: 每次MC内建筑T = max各层T，再P84
        results_bldg_s1 = []       # 建筑口径：每次MC max各层S1（与T_tot同统计序）
        results_bldg_s2 = []       # 建筑口径：每次MC max各层S2（与T_tot同统计序）
        results_injuries, results_deaths = [], []

        # F2 伤亡严格按 GB/T 38591-2020 式(14)(15)(17)(18) 逐字逐句：
        #   式(14) M_H = Σ r_{Hk} · ζ_k · A_k · λ_k        （ζ_k = 室内人员密度，λ_k = 楼层影响系数）
        #   式(17) γ_H = P84(M_H) / Σ ζ_m · A_m · λ_m      （分母 = 全楼总人数，"伤亡人数占全部人数比例"）
        #   式(15) M_D、式(18) γ_D 同理
        if '楼层影响系数' in self.floor_info.columns:
            flr_lam = self.floor_info['楼层影响系数'].astype(float).values
        else:
            flr_lam = np.ones(len(self.floor_info))
        if '楼层人口密度' in self.floor_info.columns:
            flr_zet = self.floor_info['楼层人口密度'].astype(float).values
        else:
            # 表2 建筑室内人员密度 — 办公用房 ζ=0.5 人/m² 作为缺省
            flr_zet = np.full(len(self.floor_info), 0.5, dtype=float)
        flr_areas = self.floor_info['楼层面积（m^2）'].astype(float).values
        # 式17/18 分母：Σ (人口密度 × 面积 × 楼层影响系数) = 人数加权和
        total_weighted = float((flr_zet * flr_areas * flr_lam).sum())
        # 保留一个"未乘人数版"的面积加权分母备用（防极端缺列时除以0）
        total_weighted_area = float((flr_areas * flr_lam).sum())

        print("-> 正在执行 1000 次蒙特卡洛模拟 (纯标准解析期望版) ...")
        for sim_idx in range(self.num_simulations):
            sim_inj, sim_death = 0.0, 0.0
            sim_g_s, sim_g_d, sim_g_a = 0.0, 0.0, 0.0
            sim_floor_times = {}   # 每次MC中各层总修复时间 (s1+s2)*nS

            for f_idx, floor in enumerate(self.floor_info.index):
                floor_num = int(''.join(filter(str.isdigit, str(floor))))
                floor_area = float(self.floor_areas[f_idx])
                floor_influence = float(self.floor_info.loc[floor, '楼层影响系数']) if '楼层影响系数' in self.floor_info.columns else 1.0
                # 式14/15：ζ_k = 室内人员密度（人/m²）；默认值同表2 办公用房 0.5
                pop_density = float(self.floor_info.loc[floor, '楼层人口密度']) if '楼层人口密度' in self.floor_info.columns else 0.5

                dx = sim_drifts_x[sim_idx, f_idx]
                dy = sim_drifts_y[sim_idx, f_idx]
                ax = sim_accels_x[sim_idx, f_idx]
                ay = sim_accels_y[sim_idx, f_idx]

                f_c_str, f_c_dsp, f_c_acc = 0.0, 0.0, 0.0
                Q = {k: 0.0 for k in ['W1','W2','W3','W4','W5','W6','W7','W8']}

                # DS计数：期望值累加（解析分布 → qty × P(DS)）
                struct_ds = np.zeros(5, dtype=float)
                struct_total = 0.0
                nonstruct_ds = np.zeros(4, dtype=float)
                nonstruct_total = 0.0

                for cid, cinfo in self.fragility_db.items():
                    qty_x, qty_y, qty_none = self.get_comp_qty(cid, floor_num)
                    total_qty = qty_x + qty_y + qty_none
                    if total_qty <= 0:
                        continue

                    # 折减系数：同层内同类数量（§6.2.2 / §7.2.2）
                    c_vol = self.get_volume_discount(total_qty, cinfo['cost_vol_thresholds'], cinfo['cost_vol_factors'])
                    t_vol = self.get_volume_discount(total_qty, cinfo['time_vol_thresholds'], cinfo['time_vol_factors'])

                    is_struct = (cinfo['cat'] == '结构构件')

                    def _analytic(qty_sub, edp_sub):
                        """F3: 解析期望值。返回 (期望费用, 期望工时, 期望DS数组)。"""
                        if qty_sub <= 0 or edp_sub <= 1e-6:
                            return 0.0, 0.0, None
                        medians = np.asarray(cinfo['medians'], dtype=float)
                        betas   = np.asarray(cinfo['betas'],   dtype=float)
                        n_ds = len(medians)

                        e = np.ones(n_ds)
                        for j, m in enumerate(medians):
                            if m <= 0:
                                e[j] = 1.0
                            else:
                                e[j] = norm.cdf(np.log(edp_sub / m) / betas[j])

                        probs = np.zeros(n_ds + 1)
                        probs[0] = 1.0 - e[0]
                        for j in range(n_ds - 1):
                            probs[j + 1] = e[j] - e[j + 1]
                        probs[-1] = e[-1]
                        probs = np.clip(probs, 0, 1)
                        if probs.sum() > 0:
                            probs /= probs.sum()

                        exp_c, exp_t = 0.0, 0.0
                        for j in range(1, n_ds + 1):
                            exp_c += probs[j] * cinfo['loss_ratios'][j-1] * cinfo['cost'] * cinfo['repair_factors'][j-1] * c_vol * qty_sub
                            exp_t += probs[j] * cinfo['repair_times'][j-1] * t_vol * qty_sub
                        ds_exp = qty_sub * probs
                        return exp_c, exp_t, ds_exp

                    groups = []
                    if cinfo['edp'] == 'accel':
                        ex, ey, enone = ax, ay, max(ax, ay)
                    else:
                        ex, ey, enone = dx, dy, max(dx, dy)

                    if cid.startswith('A') or cid == 'B.98.Z.Z.Z.161':
                        if qty_x > 0:    groups.append((qty_x, ex))
                        if qty_y > 0:    groups.append((qty_y, ey))
                        if qty_none > 0: groups.append((qty_none, enone))
                    elif cid == 'B.01.A.A.A.001':
                        if total_qty > 0:
                            groups.append((total_qty, np.sqrt(dx**2 + dy**2)))
                    elif cid == 'B.01.B.B.A.001':
                        if total_qty > 0:
                            groups.append((total_qty, np.sqrt(ax**2 + ay**2)))

                    for qty_sub, edp_sub in groups:
                        c_sub, t_sub, ds_sub = _analytic(qty_sub, edp_sub)
                        if is_struct:
                            f_c_str += c_sub
                            Q['W1'] += t_sub
                            if ds_sub is not None and len(ds_sub) == 5:
                                struct_ds += ds_sub
                                struct_total += qty_sub
                        else:
                            if cinfo['edp'] == 'accel':
                                f_c_acc += c_sub
                                Q['W6'] += t_sub
                            else:
                                f_c_dsp += c_sub
                                if cid == 'B.98.Z.Z.Z.161':
                                    Q['W5'] += t_sub
                                elif cid == 'B.01.A.A.A.001':
                                    Q['W4'] += t_sub
                            if ds_sub is not None:
                                ns_len = len(ds_sub)
                                nonstruct_ds[:ns_len] += ds_sub
                                nonstruct_total += qty_sub

                # 费用 × ζ
                f_c_str *= floor_influence
                f_c_dsp *= floor_influence
                f_c_acc *= floor_influence

                results_c_str[f_idx].append(f_c_str)
                results_c_dsp[f_idx].append(f_c_dsp)
                results_c_acc[f_idx].append(f_c_acc)
                sim_g_s += f_c_str
                sim_g_d += f_c_dsp
                sim_g_a += f_c_acc

                # —— 伤亡：表3 DS占比 → 表4 → F2 面积×ζ 归一化 ——
                if struct_total > 0:
                    lvl_s = self._struct_floor_level(struct_ds / struct_total)
                else:
                    lvl_s = 0
                if nonstruct_total > 0:
                    lvl_n = self._nonstruct_floor_level(nonstruct_ds[:4] / nonstruct_total)
                else:
                    lvl_n = 0
                floor_level = max(lvl_s, lvl_n)
                inj_rt, dth_rt = self._CASUALTY_RATE[floor_level]

                # F2 式(14)(15) 严格逐句：
                #   M_H = Σ r_{Hk} · ζ(人口密度) · A · λ(楼层影响系数)
                #  人员：最终γ = P84(M_H) / Σ(ζ·A·λ)，即标准所述"伤亡人数占全部人数的比例"
                sim_unit = floor_area * floor_influence * pop_density
                sim_inj   += sim_unit * inj_rt
                sim_death += sim_unit * dth_rt

                # —— F1 修复时间：N不除ζ；ζ×Q ——
                for k in Q:
                    Q[k] *= floor_influence                       # Q × ζ_T(h)  式(6)

                n_max = NMAX_COEF * floor_area                   # 式(9): 不除ζ
                N = {}
                for wk, den in W_DENSITY.items():
                    N[wk] = max(1e-6, min(n_max, (den / 100.0) * floor_area))  # 式(10): 不除ζ

                T = {wk: (Q[wk] / N[wk] if Q[wk] > 0 else 0.0) for wk in W_DENSITY}

                time_s1 = max(T['W1'], T['W2'])                 # 式(11)
                time_s2 = max(T['W3'], T['W4'] + T['W5'], T['W6'], T['W7'], T['W8'])  # 式(12)

                results_time_s1[f_idx].append(time_s1)
                results_time_s2[f_idx].append(time_s2)
                sim_floor_times[f_idx] = (time_s1 + time_s2) * n_S   # 式(13)

            # —— F4：每次MC的建筑时间 = max各层时间；S1/S2也同步取建筑max（与T_tot同口径）——
            results_g_str.append(sim_g_s)
            results_g_dsp.append(sim_g_d)
            results_g_acc.append(sim_g_a)
            results_g_tot.append(sim_g_s + sim_g_d + sim_g_a)
            results_injuries.append(sim_inj)
            results_deaths.append(sim_death)

            s1_per_floor = {f: results_time_s1[f][-1] for f in range(nf)}
            s2_per_floor = {f: results_time_s2[f][-1] for f in range(nf)}
            results_bldg_s1.append(max(s1_per_floor.values()))
            results_bldg_s2.append(max(s2_per_floor.values()))
            # 式(13)：建筑总修复时间 = max_f (S1,f + S2,f) · n_S
            results_time_bldg.append(max((s1_per_floor[f] + s2_per_floor[f]) * n_S for f in range(nf)))

        # ============================================================
        # 提取 84% 保证率分位值
        # ============================================================
        print("-> 1000 次模拟结束，正在提取 84% 保证率判据 ...")
        final_res = {'floor_cost_details': [], 'floor_time_details': []}

        for f, floor in enumerate(self.floor_info.index):
            fc_str = (self.calc_84th(results_c_str[f]) / self.bldg_total_cost) * 100
            fc_dsp = (self.calc_84th(results_c_dsp[f]) / self.bldg_total_cost) * 100
            fc_acc = (self.calc_84th(results_c_acc[f]) / self.bldg_total_cost) * 100
            fc_tot = fc_str + fc_dsp + fc_acc
            final_res['floor_cost_details'].append((floor, fc_tot, fc_acc, fc_dsp, fc_str))

            # 楼层细节展示 (s1+s2)*nS
            ft_s1 = self.calc_84th(results_time_s1[f])
            ft_s2 = self.calc_84th(results_time_s2[f])
            ft_tot = (ft_s1 + ft_s2) * n_S
            final_res['floor_time_details'].append((floor, ft_tot, ft_s1, ft_s2))

        final_res['cost_struct'] = (self.calc_84th(results_g_str) / self.bldg_total_cost) * 100
        final_res['cost_disp']   = (self.calc_84th(results_g_dsp) / self.bldg_total_cost) * 100
        final_res['cost_accel']  = (self.calc_84th(results_g_acc) / self.bldg_total_cost) * 100
        final_res['cost_total']  = (self.calc_84th(results_g_tot) / self.bldg_total_cost) * 100

        # F4: 建筑修复时间、S1、S2统一采用同一统计口径：
        #     每次MC内 取max(各层) → 再对1000次取84%分位（避免楼层分位数简单相加 ≠ 建筑总分位数）
        final_res['time_total']    = self.calc_84th(results_time_bldg)
        final_res['time_s1_total'] = self.calc_84th(results_bldg_s1)
        final_res['time_s2_total'] = self.calc_84th(results_bldg_s2)

        # F2 式17/18 伤亡率 = P84(M) / Σ(ζ_人口密度 · A · λ_楼层影响) — 占全部人数的比例
        denom = total_weighted if total_weighted > 0 else total_weighted_area
        final_res['injury_rate'] = (self.calc_84th(results_injuries) / denom) if denom > 0 else 0.0
        final_res['death_rate']  = (self.calc_84th(results_deaths) / denom) if denom > 0 else 0.0

        return final_res

    def generate_report(self):
        """法官宣判模块：打印抗震等级报告书并依照国标刻下星级钢印"""
        try:
            res = self.evaluate()
        except Exception as e:
            print(f"致命异常：数据崩溃，请审查你的输入表单合规性！详情：{e}")
            return
            
        print("="*80)
        print("二、建筑抗震韧性评级 - 综合汇总报告 (1000次随机受损碰撞测试极值版)")
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
        # GB/T 38591-2020 §9.4 星级按实际计算结果独立评定（表5/6/7阈值，不与水准硬绑定）
        #   一星: κ≤5%,  T≤7d,   γH≤1e-4, γD≤1e-5
        #   二星: κ≤10%, T≤30d,  γH≤1e-3, γD≤1e-4
        #   三星: κ≤10%, T≤30d,  γH≤1e-3, γD≤1e-4
        # 每个指标先判自身能达到的最高星级，综合 = 三者木桶取最低。
        # ==========================================
        print(f"\n[当前评估地震水准]: {self.seismic_level}")

        ct, tt = res['cost_total'], res['time_total']
        ir, dr = res['injury_rate'], res['death_rate']

        def _rate_star(cond1, cond2, cond3):
            """按条件从严到松，返回该指标能达到的最高星级名称。"""
            if cond3: return "三星"
            if cond2: return "二星"
            if cond1: return "一星"
            return "无星级"

        # 修复费用 (表5): 一/二星阈值不等，三星与二星阈值一致
        star_cost = _rate_star(
            ct <= 5.0,
            ct <= 10.0,
            ct <= 10.0
        )
        # 修复时间 (表6): 一/二星阈值不等，三星与二星阈值一致
        star_time = _rate_star(
            tt <= 7.0,
            tt <= 30.0,
            tt <= 30.0
        )
        # 人员伤亡 (表7): 一/二星阈值不等，三星与二星阈值一致
        star_casualty = _rate_star(
            ir <= 1.0e-4 and dr <= 1.0e-5,
            ir <= 1.0e-3 and dr <= 1.0e-4,
            ir <= 1.0e-3 and dr <= 1.0e-4
        )

        # 木桶效应：综合星级
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