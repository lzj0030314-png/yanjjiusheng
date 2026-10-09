import os                             # 导入操作系统接口模块，用于处理文件物理路径
import numpy as np                    # 导入科学计算核心库，用于矩阵、向量运算及蒙特卡洛随机数生成
import pandas as pd                   # 导入数据分析库，用于高速读取和清洗 Excel 结构表格
from scipy.stats import norm, lognorm  # 正态分布用于易损性概率；lognorm用于GB/T 38591-2020 A.5.4的对数正态拟合

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
        # 5. 全参数易损性数据库挂载（严苛对齐 GB/T 38591 规范表格的插值边界）
        self.fragility_db = {
            'A.98.Z.Z.Z.814': {
                'cat': '结构构件', 'edp': 'drift', 'cost': 1500, 
                'medians': [0.005, 0.010, 0.020, 0.030], 'betas': [0.4, 0.4, 0.4, 0.4],
                'loss_ratios': [0.1, 0.2, 0.6, 1.0], 'repair_factors': [1.22, 1.18, 1.06, 3.15],
                'repair_times': [3.8, 5.6, 11.3, 25.0],
                # 费用折减（表 C.10）：≤10为1.0, ≥50为0.85，配合 np.interp 自动计算 11~49 插值
                'cost_vol_thresholds': [10.0, 50.0], 'cost_vol_factors': [1.0, 0.85],       
                # 工时折减（表 C.12）：≤10为1.0, ≥50为0.75，配合 np.interp 自动计算 11~49 插值
                'time_vol_thresholds': [10.0, 50.0], 'time_vol_factors': [1.0, 0.75]        
            },
            'A.98.Z.Z.Z.813': {
                'cat': '结构构件', 'edp': 'drift', 'cost': 1483, 
                'medians': [0.007, 0.010, 0.016, 0.023], 'betas': [0.4, 0.4, 0.4, 0.4],
                'loss_ratios': [0.1, 0.2, 0.5, 1.0], 'repair_factors': [1.20, 1.07, 1.15, 3.57],
                'repair_times': [2.6, 6.2, 9.4, 27.8],
                # 费用/工时折减同上
                'cost_vol_thresholds': [10.0, 50.0], 'cost_vol_factors': [1.0, 0.85],       
                'time_vol_thresholds': [10.0, 50.0], 'time_vol_factors': [1.0, 0.75]        
            },
            'B.98.Z.Z.Z.161': { 
                'cat': '位移敏感型非结构构件', 'edp': 'drift', 'cost': 150, 
                'medians': [0.005, 0.010, 0.021], 'betas': [0.4, 0.3, 0.2],
                'loss_ratios': [0.22, 0.50, 1.00], 'repair_factors': [1.15, 1.21, 1.27],
                'repair_times': [0.015, 0.029, 0.056],
                # 参照表 E.7 (填充墙类): ≤1为1.0, ≥10为0.89，配合 np.interp 自动计算 2~9 插值
                'cost_vol_thresholds': [1.0, 10.0], 'cost_vol_factors': [1.0, 0.89],       
                'time_vol_thresholds': [1.0, 10.0], 'time_vol_factors': [1.0, 0.89]        
            },
            'B.01.B.B.A.001': {
                'cat': '加速度敏感型非结构构件', 'edp': 'accel', 'cost': 225, 
                'medians': [1.17, 1.58, 1.82], 'betas': [0.25, 0.25, 0.25],
                'loss_ratios': [0.10, 0.50, 1.00], 'repair_factors': [1.93, 1.49, 1.31],
                'repair_times': [0.015, 0.116, 0.240],
                # 参照表 E.7 (吊顶类): ≤1为1.0, ≥10为0.96，配合 np.interp 自动计算 2~9 插值
                'cost_vol_thresholds': [1.0, 10.0], 'cost_vol_factors': [1.0, 0.96],       
                'time_vol_thresholds': [1.0, 10.0], 'time_vol_factors': [1.0, 0.96]        
            },
            'B.01.A.A.A.001': {
                'cat': '位移敏感型非结构构件', 'edp': 'drift', 'cost': 1520, 
                'medians': [0.0338, 0.0383], 'betas': [0.4, 0.4],
                'loss_ratios': [0.0, 1.00], 'repair_factors': [0.0, 1.67],
                'repair_times': [0.0, 0.216],
                # 参照表 E.7 (幕墙类): ≤1为1.0, ≥10为0.89，配合 np.interp 自动计算 2~9 插值
                'cost_vol_thresholds': [1.0, 10.0], 'cost_vol_factors': [1.0, 0.89],       
                'time_vol_thresholds': [1.0, 10.0], 'time_vol_factors': [1.0, 0.89]        
            }
        
        }
        
        # 6. 提取原始工程需求参数响应矩阵 (EDP 矩阵) - 【纯净读取，不做任何自动剔除】
        import numpy as np
        edp_raw = pd.read_excel(xls, '结构响应')                  
        # 单纯地把 '--' 或 'NULL' 变成空值(np.nan)，不删列也不填0，原汁原味直接提取
        # 位移角(drift)：
        self.edp_drift = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL', ''], np.nan).astype(float).values.T
        self.drifts_x = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL', ''], np.nan).astype(float).values.T 
        self.drifts_y = edp_raw.iloc[[2, 3, 4], 23:34].replace(['--', 'NULL', ''], np.nan).astype(float).values.T 
        # 加速度(accel)：
        self.edp_accel = edp_raw.iloc[[2, 3, 4, 1], 12:23].replace(['--', 'NULL', ''], np.nan).astype(float).values.T
        self.accels_x = edp_raw.iloc[[2, 3, 4, 1], 12:23].replace(['--', 'NULL', ''], np.nan).astype(float).values.T 
        self.accels_y = edp_raw.iloc[[2, 3, 4, 1], 34:45].replace(['--', 'NULL', ''], np.nan).astype(float).values.T


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
        """【工程需求参数矩阵扩充】（严格按照 GB/T 38591-2020 附录 G.2 公式 G.1 实现）"""
        # 1. 数值托底护盾，防止对数运算报错
        edp_matrix = np.clip(edp_matrix, a_min=1e-6, a_max=None)
        
        # 2. 对数正态空间转换 (G.2.3: 矩阵的值取对数)
        # 这里为了配合公式的矩阵维度，调整为 (n_params, m_waves) 格式
        Y = np.log(edp_matrix).T  
        n_params = Y.shape[0]
        # 3. 计算对数空间的均值向量 M_Y (形状为 (n_params, 1))
        mean_y = np.mean(Y, axis=1, keepdims=True)
        # 4. 计算对数空间的协方差矩阵 \Sigma_{YY}
        cov_y = np.cov(Y)
        if np.isscalar(cov_y):
            cov_y = np.array([[cov_y]])
        # 5. 特征值分解修复并进行 Cholesky 分解 (G.2.4 & G.2.5): 得到下三角矩阵 L
        eigvals, eigvecs = np.linalg.eigh(cov_y)
        eigvals[eigvals < 1e-8] = 1e-8  # 修正小样本导致的不满秩
        cov_y_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        try:
            L = np.linalg.cholesky(cov_y_pd)
        except np.linalg.LinAlgError:
            # 极端情况下的兜底
            L = np.linalg.cholesky(cov_y_pd + np.eye(n_params) * 1e-6)
        # 6. 生成独立正态分布变量 U (G.2.6: 伪随机数生成的标准正态变量，形状为 (n_params, num_simulations))
        U = np.random.randn(n_params, self.num_simulations)
        # 7. 严格执行国标核心公式 (G.1): Z = L * U + M_Y
        Z = np.dot(L, U) + mean_y
        # 8. 指数还原 (G.2.8) 并转置回 (num_simulations, n_params) 的输出格式
        simulated_data = np.exp(Z).T
        return simulated_data                                  # 指数反推，将这 1000 次模拟数据弹回真实的物理空间

    def calc_84th(self, arr):
        """
        GB/T 38591-2020 A.5.4 / 9.4.1

        对蒙特卡洛得到的指标集合进行对数正态分布拟合，
        再取具有84%保证率的拟合值。

        处理原则：
        1) 全部为0：该指标在全部模拟中均为0，直接返回0；
        2) 全部>0：严格进行对数正态拟合；
        3) 0与正值混合：标准正文没有给出“零膨胀样本”的专门拟合方法。
           为避免静默删除0值，这里采用“零点质量 + 正值部分对数正态”的工程统计处理：
           - 先保留零值比例 p0；
           - 若累计概率0.84落在零点质量内（p0 >= 0.84），84%保证率就是0；
           - 否则在正值部分按条件概率 p=(0.84-p0)/(1-p0) 求对数正态拟合分位值。

        第3种情况属于对标准“对数正态拟合”要求的数值实现补充，
        标准A.5.4本身并未明确规定零值应如何处理。
        """
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return 0.0

        if np.any(arr < 0):
            raise ValueError("抗震韧性指标不应出现负值。")

        zero_mask = (arr == 0)
        zero_ratio = float(np.mean(zero_mask))

        # 全部为0：没有必要做对数正态拟合
        if zero_ratio == 1.0:
            return 0.0

        positive = arr[~zero_mask]

        # 84%累计概率仍落在零值质量中
        if zero_ratio >= 0.84:
            return 0.0

        # 将总体84%累计概率映射到“正值条件分布”的累计概率
        conditional_p = (0.84 - zero_ratio) / (1.0 - zero_ratio)
        conditional_p = float(np.clip(conditional_p, 1e-12, 1.0 - 1e-12))

        shape, loc, scale = lognorm.fit(positive, floc=0)
        value_84 = lognorm.ppf(conditional_p, shape, loc=loc, scale=scale)

        if not np.isfinite(value_84):
            raise ValueError("对数正态分布拟合得到的84%保证率值不是有限数。")

        return float(value_84)

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
        """【经济学引擎】：调用 numpy 底层插值算法，完美还原规范表中的(两端锁定+中间平滑插值)规则"""
        return float(np.interp(qty, thresholds, factors))

    # ==========================================================================
    # 表3 辅助：按DS占比判定楼层破坏等级 0..4 (I..V)
    # 严格依据 GB/T 38591-2020 §8.2.2 表3：
    #   - 判定逻辑：自最严重等级Ⅴ向Ⅰ"阈值上限≤AND"逐级下探，命中即停；
    #   - 仅"Ⅴ级"行原文使用三个并列"或"关系，其余行内逗号均读作"且"。
    # ==========================================================================
    def _struct_floor_level(self, p):
        """【完全复刻国标表3规范逻辑】结构构件楼层破坏等级判定
        p 数组对应构件比例 p[2]=DS2, p[3]=DS3, p[4]=DS4
        """
        ds2, ds3, ds4 = float(p[2]), float(p[3]), float(p[4])
        # -------------------------------------------------------------------------
        # I 级：损伤状态不大于1级的构件数量占比等于 100%
        # （注：不大于1级即代表 DS2=0 且 DS3=0 且 DS4=0）
        # -------------------------------------------------------------------------
        if (ds2 == 0.0) and (ds3 == 0.0) and (ds4 == 0.0):
            return 1 
        # -------------------------------------------------------------------------
        # II 级：损伤状态为2级的构件不超过 10%，且不出现损伤状态大于2级的构件
        # -------------------------------------------------------------------------
        if (ds2 <= 0.10) and (ds3 == 0.0) and (ds4 == 0.0):
            return 2 
        # -------------------------------------------------------------------------
        # III 级：损伤状态为2级的构件不超过 20%，且损伤状态为3级的构件占比不超过 10%，且不出现4级
        # -------------------------------------------------------------------------
        if (ds2 <= 0.20) and (ds3 <= 0.10) and (ds4 == 0.0):
            return 3
        # ------------------------------------------------------------------------
        # IV 级：损伤状态为2级不超过 50%，3级不超过 20%，4级不超过 10%
        # -------------------------------------------------------------------------
        if (ds2 <= 0.50) and (ds3 <= 0.20) and (ds4 <= 0.10):
            return 4
        # -------------------------------------------------------------------------
        # V 级：兜底拦截。凡是逃过了 IV 级条件限制的，必然是有任一项超过了上限，直接归入最差等级。
        # （完美等效于原标准中的或逻辑：DS2>50% 或 DS3>20% 或 DS4>10%）
        # -------------------------------------------------------------------------
        return 5      
        

    def _nonstruct_floor_level(self, p):
        """【完全严格对标国标表3】可致伤亡的非结构构件楼层破坏等级判定"""
        d1, d2, d3 = float(p[1]), float(p[2]), float(p[3])
        damaged = d1 + d2 + d3
        # -------------------------------------------------------------------------
        # I 级：损伤状态为1级的构件占比不超过 10%，且不出现超过1级的构件
        # -------------------------------------------------------------------------
        if (d1 <= 0.10) and (d2 == 0.0) and (d3 == 0.0):
            return 1
        # -------------------------------------------------------------------------
        # II 级：损伤状态为1级的构件占比不超过 30%，且不出现大于1级的构件
        # -------------------------------------------------------------------------
        if (d1 <= 0.30) and (d2 == 0.0) and (d3 == 0.0):
            return 2
        # -------------------------------------------------------------------------
        # III 级：震损构件占比不超过 50%，且DS2不超过 10%，且不出现超过2级的构件
        # -------------------------------------------------------------------------
        if (damaged <= 0.50) and (d2 <= 0.10) and (d3 == 0.0):
            return 3
        # -------------------------------------------------------------------------
        # IV 级：震损构件占比超过 50%，且DS2占比不超过 50%，且DS3占比不超过 10%
        # -------------------------------------------------------------------------
        if (damaged > 0.50) and (d2 <= 0.50) and (d3 <= 0.10):
            return 4
        # -------------------------------------------------------------------------
        # V 级：DS2比例超过 50%，或DS3占比超过 10%
        # -------------------------------------------------------------------------
        if (d2 > 0.50) or (d3 > 0.10):
            return 5
        # -------------------------------------------------------------------------
        # 规范盲区兜底（工程保守原则）
        # 拦截情况：例如 damaged <= 50% 但 d2 达到了 20% (大于10%且未触发V级)
        # -------------------------------------------------------------------------
        return 4
     

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
        中枢控制台：1000 次蒙特卡洛模拟 
        """
        np.random.seed(42)

        # EDP扩充
        sim_drifts_x = self.expand_edp_matrix(self.drifts_x)
        sim_drifts_y = self.expand_edp_matrix(self.drifts_y)
        sim_accels_x = self.expand_edp_matrix(self.accels_x) / 9.8
        sim_accels_y = self.expand_edp_matrix(self.accels_y) / 9.8

        # n_S 在标准公式(13)中表示建筑楼层数，
        # 是 max 的索引上限，不是“1.00/1.05/1.08/1.10”的乘法系数。
        # 1.00/1.05/1.08/1.10 属于楼层修复时间影响系数 lambda_T(k)，
        # 应在 Q(i,k) 中体现，而不能在 T_k,S1 + T_k,S2 后再次乘一次。
        nf = self.num_floors

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
        results_time_bldg = []     
        results_bldg_s1 = []       
        results_bldg_s2 = []       
        results_injuries, results_deaths = [], []

        # ==========================================
        # 1. 提取全楼分母参数 (纯净对标标准，无楼层影响系数)
        # ==========================================
        if '楼层人口密度' in self.floor_info.columns:
            flr_zet = self.floor_info['楼层人口密度'].astype(float).values
        else:
            flr_zet = np.full(len(self.floor_info), 0.5, dtype=float)
            
        flr_areas = self.floor_info['楼层面积（m^2）'].astype(float).values
        
        # 式(17)和式(18)的分母：Σ(ζ_m · A_m)，作为评估伤亡率的绝对基数
        total_weighted = float((flr_zet * flr_areas).sum())
        total_weighted_area = float(flr_areas.sum()) 

        # 2. 名义伤亡率表
        _CASUALTY_RATE_TABLE = {
            0: (0.0,         0.0),          
            1: (1/80000.0,   0.0),          
            2: (1/20000.0,   0.0),          
            3: (1/8000.0,    1/80000.0),    
            4: (1/140.0,     1/800.0),      
        }

        print("-> 正在执行 1000 次蒙特卡洛模拟 (纯标准解析期望版) ...")
        for sim_idx in range(self.num_simulations):
            sim_inj, sim_death = 0.0, 0.0
            sim_g_s, sim_g_d, sim_g_a = 0.0, 0.0, 0.0
            sim_floor_times = {}   

            for f_idx, floor in enumerate(self.floor_info.index):
                floor_num = int(''.join(filter(str.isdigit, str(floor))))
                floor_area = float(self.floor_areas[f_idx])
                floor_influence = float(self.floor_info.loc[floor, '楼层影响系数']) if '楼层影响系数' in self.floor_info.columns else 1.0
                pop_density = float(self.floor_info.loc[floor, '楼层人口密度']) if '楼层人口密度' in self.floor_info.columns else 0.5

                dx = sim_drifts_x[sim_idx, f_idx]
                dy = sim_drifts_y[sim_idx, f_idx]
                ax = sim_accels_x[sim_idx, f_idx]
                ay = sim_accels_y[sim_idx, f_idx]

                f_c_str, f_c_dsp, f_c_acc = 0.0, 0.0, 0.0
                Q = {k: 0.0 for k in ['W1','W2','W3','W4','W5','W6','W7','W8']}

                struct_ds = np.zeros(5, dtype=float)
                struct_total = 0.0
                nonstruct_ds = np.zeros(4, dtype=float)
                nonstruct_total = 0.0

                for cid, cinfo in self.fragility_db.items():
                    qty_x, qty_y, qty_none = self.get_comp_qty(cid, floor_num)
                    total_qty = qty_x + qty_y + qty_none
                    if total_qty <= 0:
                        continue

                    c_vol = self.get_volume_discount(total_qty, cinfo['cost_vol_thresholds'], cinfo['cost_vol_factors'])
                    t_vol = self.get_volume_discount(total_qty, cinfo['time_vol_thresholds'], cinfo['time_vol_factors'])
                    is_struct = (cinfo['cat'] == '结构构件')

                    def _analytic(qty_sub, edp_sub):
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

                # 注意：floor_influence 对应修复时间中的 λ_T(k)，
                # 不是建筑修复费用公式中的系数。
                # 因此费用这里不再乘 floor_influence。

                results_c_str[f_idx].append(f_c_str)
                results_c_dsp[f_idx].append(f_c_dsp)
                results_c_acc[f_idx].append(f_c_acc)
                sim_g_s += f_c_str
                sim_g_d += f_c_dsp
                sim_g_a += f_c_acc

                # —— 伤亡：严格执行式 (14) 与 式 (15) 的双重求和逻辑 ——
                # —— 伤亡：按国标 8.2.2 确定楼层唯一破坏等级，取最大值 ——
                
                # —— 伤亡：按国标 8.2.2 确定楼层唯一破坏等级，取最大值 ——
                
                # 【修正：严格对齐国标表 4 的 5 个破坏等级】
                # 消除索引错位：1对应Ⅰ级(完好)，5对应Ⅴ级(严重破坏)
                CORRECT_CASUALTY_RATE = {
                    1: (0.0,         0.0),          # Ⅰ级
                    2: (1/80000.0,   0.0),          # Ⅱ级
                    3: (1/20000.0,   0.0),          # Ⅲ级
                    4: (1/8000.0,    1/80000.0),    # Ⅳ级
                    5: (1/140.0,     1/800.0),      # Ⅴ级
                }
                
                # 1. 判定结构构件破坏等级
                if struct_total > 0:
                    lvl_s = self._struct_floor_level(struct_ds / struct_total)
                else:
                    lvl_s = 1  # 无损坏即为 Ⅰ级
                    
                # 2. 判定非结构构件破坏等级
                if nonstruct_total > 0:
                    lvl_n = self._nonstruct_floor_level(nonstruct_ds[:4] / nonstruct_total)
                else:
                    lvl_n = 1  # 无损坏即为 Ⅰ级
                    
                # 3. 核心要求：判定结果取两者中较大的楼层破坏等级
                floor_level = max(lvl_s, lvl_n)
                
                # 4. 获取该等级对应的名义伤亡率
                r_hr, r_dr = CORRECT_CASUALTY_RATE.get(floor_level, (0.0, 0.0))
                
                # 5. 整个楼层的面积 A_k 都属于该破坏等级，直接累加
                sim_inj   += r_hr * pop_density * floor_area
                sim_death += r_dr * pop_density * floor_area
                
                # —— F1 修复时间：N不除ζ；ζ×Q ——
                for k in Q:
                    Q[k] *= floor_influence                       

                n_max = NMAX_COEF * floor_area                   
                N = {}
                for wk, den in W_DENSITY.items():
                    N[wk] = max(1e-6, min(n_max, (den / 100.0) * floor_area))  

                T = {wk: (Q[wk] / N[wk] if Q[wk] > 0 else 0.0) for wk in W_DENSITY}

                time_s1 = max(T['W1'], T['W2'])                 
                # 严格对标式(12): 管线W6、隔断W4、吊顶W5必须串联相加
                time_s2 = max(T['W3'], T['W4'] + T['W5'] + T['W6'], T['W7'], T['W8'])  

                results_time_s1[f_idx].append(time_s1)
                results_time_s2[f_idx].append(time_s2)
                # 公式(13)是在各楼层之间取最大值；
                # n_S 是楼层数的索引上限，不是乘法系数。
                sim_floor_times[f_idx] = time_s1 + time_s2

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

            # GB/T 38591-2020 式(13)：
            # 本次蒙特卡洛模拟的整栋建筑修复时间
            # = 各楼层 (S1 + S2) 中的最大值。
            results_time_bldg.append(
                max(s1_per_floor[f] + s2_per_floor[f] for f in range(nf))
            )

        print("-> 1000 次模拟结束，正在提取 84% 保证率判据 ...")
        final_res = {'floor_cost_details': [], 'floor_time_details': []}

        for f, floor in enumerate(self.floor_info.index):
            fc_str = (self.calc_84th(results_c_str[f]) / self.bldg_total_cost) * 100
            fc_dsp = (self.calc_84th(results_c_dsp[f]) / self.bldg_total_cost) * 100
            fc_acc = (self.calc_84th(results_c_acc[f]) / self.bldg_total_cost) * 100
            fc_tot = fc_str + fc_dsp + fc_acc
            final_res['floor_cost_details'].append((floor, fc_tot, fc_acc, fc_dsp, fc_str))

            ft_s1 = self.calc_84th(results_time_s1[f])
            ft_s2 = self.calc_84th(results_time_s2[f])
            ft_tot = ft_s1 + ft_s2
            final_res['floor_time_details'].append((floor, ft_tot, ft_s1, ft_s2))

        final_res['cost_struct'] = (self.calc_84th(results_g_str) / self.bldg_total_cost) * 100
        final_res['cost_disp']   = (self.calc_84th(results_g_dsp) / self.bldg_total_cost) * 100
        final_res['cost_accel']  = (self.calc_84th(results_g_acc) / self.bldg_total_cost) * 100
        final_res['cost_total']  = (self.calc_84th(results_g_tot) / self.bldg_total_cost) * 100

        # 建筑总体修复时间：严格按“每次模拟先得到 T_tot，再对 T_tot 集合做84%拟合”。
        # 不能用 max(各楼层84%值) 代替，因为 max 与84%分位拟合一般不可交换。
        final_res['time_total'] = self.calc_84th(results_time_bldg) if results_time_bldg else 0.0

        # 以下两个值仅作为报告展示，不作为标准中的 T_tot。
        # 它们表示各楼层 S1、S2 时间集合分别拟合后的最大值。
        if final_res['floor_time_details']:
            final_res['time_s1_total'] = max(row[2] for row in final_res['floor_time_details'])
            final_res['time_s2_total'] = max(row[3] for row in final_res['floor_time_details'])
        else:
            final_res['time_s1_total'] = final_res['time_s2_total'] = 0.0

        # 最终归一化伤亡率指标
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
        # GB/T 38591-2020 §9.4 星级评定模块
       
        print(f"\n[当前评估地震水准]: {self.seismic_level}")

        ct, tt = res['cost_total'], res['time_total']
        ir, dr = res['injury_rate'], res['death_rate']

        star_cost = "无星级"
        star_time = "无星级"
        star_casualty = "无星级"

        # ------------------------------------------
        # 罕遇地震水准评价 (仅可评 三星、二星)
        # ------------------------------------------
        if self.seismic_level == "罕遇地震":
            
            # 9.1 修复费用评级 (表 5)
            if ct <= 5.0:
                star_cost = "三星"
            elif 5.0 < ct <= 10.0:    # 严格照抄原件: 5% < κ ≤ 10%
                star_cost = "二星"
                
            # 9.2 修复时间评级 (表 6)
            if tt <= 7.0:
                star_time = "三星"
            elif 7.0 < tt <= 30.0:    # 严格照抄原件: 7 d < T_tot ≤ 30 d
                star_time = "二星"
                
            # 9.3 人员伤亡评级 (表 7)
            # 原件对二星无下限描述，程序利用 elif 的互斥性承接，但条件完完全全按原件抄录
            if ir <= 1.0e-4 and dr <= 1.0e-5:
                star_casualty = "三星"
            elif ir <= 1.0e-3 and dr <= 1.0e-4:
                star_casualty = "二星"

        # ------------------------------------------
        # 设防地震水准评价 (仅可评 一星)
        # ------------------------------------------
        elif self.seismic_level == "设防地震":
            
            # 9.1 修复费用评级 (表 5)
            if ct <= 10.0:            # 严格照抄原件: κ ≤ 10%
                star_cost = "一星"
                
            # 9.2 修复时间评级 (表 6)
            if tt <= 30.0:            # 严格照抄原件: T_tot ≤ 30 d
                star_time = "一星"
                
            # 9.3 人员伤亡评级 (表 7)
            if ir <= 1.0e-3 and dr <= 1.0e-4:  # 严格照抄原件: γH ≤ 1.0×10^-3, 且 γD ≤ 1.0×10^-4
                star_casualty = "一星"
            
        else:
            print("⚠️ 警告：当前地震水准不满足国标《GB/T 38591-2020》要求，无法定级！")

        # ------------------------------------------
        # 最终综合评级逻辑：三者取最低 (木桶效应)
        # ------------------------------------------
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