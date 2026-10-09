# -------------------------------------------------------------------------
# 【第一部分：准备工具箱】
# 就像盖房子前要准备好挖掘机、吊车一样，这里在导入程序需要的各种“功能包”
# -------------------------------------------------------------------------
import os                             # 导入操作系统模块：让程序能认路，找到你电脑里的文件路径（比如D盘某某文件夹）
import numpy as np                    # 导入numpy库（简称np）：这是程序的“数学大脑”，负责处理复杂的矩阵运算和生成随机数（用于蒙特卡洛摇号）
import pandas as pd                   # 导入pandas库（简称pd）：这是“超级Excel处理员”，负责一秒钟读完你几万行的表格数据
from scipy.stats import norm          # 导入scipy里的正态分布模块：用于计算概率，比如“在这次地震下，墙壁坏掉的概率有多大”

# -------------------------------------------------------------------------
# 【第二部分：定义评估流水线（类）】
# 把整个抗震评估系统封装成一个完整的“机器”，名叫 TrueResilienceEvaluator
# -------------------------------------------------------------------------
class TrueResilienceEvaluator:
    
    # -------------------------------------------------------------------------
    # 1. 机器的启动按钮（初始化函数 __init__）
    # 当你运行程序时，这里是最先执行的地方
    # -------------------------------------------------------------------------
    def __init__(self, file_name='中震三层数据导出版.xls'):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前这个Python代码文件所在的文件夹路径
        self.file_path = os.path.join(self.data_dir, file_name)     # 把文件夹路径和你的Excel文件名拼在一起，找到数据的准确位置
        self.num_simulations = 1000                                 # 设定蒙特卡洛模拟次数为1000次（国标强制要求不能少于1000次）
        self.load_data()                                            # 按下启动按钮后，立刻呼叫下一个步骤：加载数据！
        
    # -------------------------------------------------------------------------
    # 2. 数据搬运工（load_data 函数）
    # 负责把 Excel 里的数据全部搬进电脑内存里，准备好后续的计算
    # -------------------------------------------------------------------------
    def load_data(self):
        print(f"正在加载真实数据文件: {self.file_path} ...")           # 在屏幕上打印一句话，告诉你它正在努力干活
        xls = pd.ExcelFile(self.file_path)                          # 把整个Excel文件完整地塞进机器里，起名叫 xls
        
        # --- 收集大楼整体信息 ---
        bldg_info1 = pd.read_excel(xls, '建筑信息1').iloc[0]        # 翻到叫“建筑信息1”的这一页，读取第一行数据
        self.total_area = float(bldg_info1.get('建筑总面积（平方米）', 111.50)) # 从里面提取出大楼总面积（如果没填，默认按111.5平米算）
        self.unit_cost = float(bldg_info1.get('单位造价（元/平方米）', 3000.0))  # 提取大楼每平米的造价（默认3000元）
        self.bldg_total_cost = self.total_area * self.unit_cost       # 计算出整栋大楼如果完全重修需要多少钱（总面积 × 单价）
        
        # --- 收集每一层楼的信息 ---
        self.floor_info = pd.read_excel(xls, '建筑信息2')             # 翻到“建筑信息2”这一页，读出每层楼的详细数据
        self.floor_info.set_index('楼层（层）', inplace=True)         # 把“楼层（层）”这一列设为目录索引，方便程序后面按楼层找数据
        self.num_floors = len(self.floor_info)                      # 数一数表格里有几行，就知道这栋楼一共有几层了
        self.floor_areas = self.floor_info['楼层面积（m^2）'].values      # 把每一层的面积单独抽出来，排成一列备用
        
        # --- 收集大楼里装了哪些零件（构件） ---
        self.struct_info = pd.read_excel(xls, '结构构件信息')           # 读取结构构件表（梁、柱、墙在哪层，有多少个）
        self.nonstruct_info = pd.read_excel(xls, '非结构构件信息')        # 读取非结构构件表（吊顶、玻璃幕墙在哪层，有多少个）

        # --- 获取地震到底有多大 ---
        df_seismic = pd.read_excel(xls, '地震信息1')                  # 翻到“地震信息1”这一页
        self.seismic_level = str(df_seismic['地震水准'].iloc[0]).strip() # 读出你设定的地震级别（比如“设防地震”、“罕遇地震”）

        # -------------------------------------------------------------------------
        # 易损性数据库：这是国标里规定的“零件受损说明书”
        # 记录了某种零件在多大的晃动下会坏、坏了修要多少钱、修要多少天
        # （这里列举了几个核心零件作为代表，字典格式）
        # -------------------------------------------------------------------------
        self.fragility_db = {
            # 以 A.98.Z.Z.Z.814 这种型号的结构构件为例：
            'A.98.Z.Z.Z.814': {
                'cat': '结构构件', 'edp': 'drift', 'cost': 1500,     # 类别是结构件，怕的是位移(drift)，坏了基础重修费1500元
                'medians': [0.005, 0.010, 0.020, 0.030],             # 抵抗晃动的能力（达到这些数值，就开始不同程度的损坏）
                'betas': [0.4, 0.4, 0.4, 0.4],                       # 数据的不确定性波动范围
                'loss_ratios': [0.1, 0.2, 0.6, 1.0],                 # 1至4级破坏时，各需要花基础重修费的百分之多少
                'repair_factors': [1.22, 1.18, 1.06, 3.15],          # 修复费用的放大系数
                'repair_times': [3.8, 5.6, 11.3, 25.0],              # 1至4级破坏时，修好需要多少天
                # 批发价打折规则：如果你楼里坏得多（超过50个），修的单价就打0.85折
                'cost_vol_thresholds': [10.0, 50.0], 'cost_vol_factors': [1.0, 0.85],       
                'time_vol_thresholds': [10.0, 50.0], 'time_vol_factors': [1.0, 0.75]        
            },
            # （中间其他的构件说明书逻辑完全一样，省略讲解，直接载入）
            'A.98.Z.Z.Z.813': { 'cat': '结构构件', 'edp': 'drift', 'cost': 1483, 'medians': [0.007, 0.010, 0.016, 0.023], 'betas': [0.4, 0.4, 0.4, 0.4], 'loss_ratios': [0.1, 0.2, 0.5, 1.0], 'repair_factors': [1.20, 1.07, 1.15, 3.57], 'repair_times': [2.6, 6.2, 9.4, 27.8], 'cost_vol_thresholds': [10.0, 50.0], 'cost_vol_factors': [1.0, 0.85], 'time_vol_thresholds': [10.0, 50.0], 'time_vol_factors': [1.0, 0.75] },
            'B.98.Z.Z.Z.161': { 'cat': '位移敏感型非结构构件', 'edp': 'drift', 'cost': 150, 'medians': [0.005, 0.010, 0.021], 'betas': [0.4, 0.3, 0.2], 'loss_ratios': [0.22, 0.50, 1.00], 'repair_factors': [1.15, 1.21, 1.27], 'repair_times': [0.015, 0.029, 0.056], 'cost_vol_thresholds': [1.0, 10.0], 'cost_vol_factors': [1.0, 0.89], 'time_vol_thresholds': [1.0, 10.0], 'time_vol_factors': [1.0, 0.89] },
            'B.01.B.B.A.001': { 'cat': '加速度敏感型非结构构件', 'edp': 'accel', 'cost': 225, 'medians': [1.17, 1.58, 1.82], 'betas': [0.25, 0.25, 0.25], 'loss_ratios': [0.10, 0.50, 1.00], 'repair_factors': [1.93, 1.49, 1.31], 'repair_times': [0.015, 0.116, 0.240], 'cost_vol_thresholds': [1.0, 10.0], 'cost_vol_factors': [1.0, 0.96], 'time_vol_thresholds': [1.0, 10.0], 'time_vol_factors': [1.0, 0.96] },
            'B.01.A.A.A.001': { 'cat': '位移敏感型非结构构件', 'edp': 'drift', 'cost': 1520, 'medians': [0.0338, 0.0383], 'betas': [0.4, 0.4], 'loss_ratios': [0.0, 1.00], 'repair_factors': [0.0, 1.67], 'repair_times': [0.0, 0.216], 'cost_vol_thresholds': [1.0, 10.0], 'cost_vol_factors': [1.0, 0.89], 'time_vol_thresholds': [1.0, 10.0], 'time_vol_factors': [1.0, 0.89] }
        }
        
        # --- 提取 OpenSees 软件跑出来的受力数据（EDP矩阵） ---
        edp_raw = pd.read_excel(xls, '结构响应')                  
        # 提取 X 方向和 Y 方向的层间位移角（大楼被扭歪了多少）
        self.drifts_x = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL', ''], np.nan).astype(float).values.T 
        self.drifts_y = edp_raw.iloc[[2, 3, 4], 23:34].replace(['--', 'NULL', ''], np.nan).astype(float).values.T 
        # 提取 X 方向和 Y 方向的楼层加速度（大楼甩得有多狠）
        self.accels_x = edp_raw.iloc[[2, 3, 4, 1], 12:23].replace(['--', 'NULL', ''], np.nan).astype(float).values.T 
        self.accels_y = edp_raw.iloc[[2, 3, 4, 1], 34:45].replace(['--', 'NULL', ''], np.nan).astype(float).values.T

    # -------------------------------------------------------------------------
    # 3. 核心数学引擎：数据克隆术（expand_edp_matrix 函数）
    # 你的Excel里只有11条地震波数据，这个函数能通过高深的统计学原理，
    # 凭空生成1000条符合物理规律的地震数据，用于后续的蒙特卡洛抽奖！
    # -------------------------------------------------------------------------
    def expand_edp_matrix(self, edp_matrix):
        edp_matrix = np.clip(edp_matrix, a_min=1e-6, a_max=None) # 加个底线，防止有些数据是0，后面算对数会报错
        Y = np.log(edp_matrix).T                                 # 把数据取对数，方便进行正态分布计算
        n_params = Y.shape[0]                                    # 看看数据有几行（有几个参数）
        mean_y = np.mean(Y, axis=1, keepdims=True)               # 算出这11条数据的平均水平
        cov_y = np.cov(Y)                                        # 算出它们之间的关联性（协方差）
        if np.isscalar(cov_y): cov_y = np.array([[cov_y]])       # 如果只有一个数据，包一层括号防止出错
        
        # 下面几行是在修复矩阵，确保它能进行数学分解（特征值分解）
        eigvals, eigvecs = np.linalg.eigh(cov_y)
        eigvals[eigvals < 1e-8] = 1e-8  
        cov_y_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        try:
            L = np.linalg.cholesky(cov_y_pd)                     # 找到数据的“基因密码”（Cholesky分解）
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov_y_pd + np.eye(n_params) * 1e-6) # 万一失败了，加点润滑油再分解一次
            
        U = np.random.randn(n_params, self.num_simulations)      # 抛1000次标准的硬币（生成1000组标准随机数）
        Z = np.dot(L, U) + mean_y                                # 把标准随机数加上“基因密码”和“平均值”，变成真实的模拟数据
        simulated_data = np.exp(Z).T                             # 因为开头取了对数，现在用指数把它变回真实世界的物理数值
        return simulated_data                                    # 把这1000条数据还给主程序

    # -------------------------------------------------------------------------
    # 4. 后处理工具包
    # -------------------------------------------------------------------------
    # 提取84%保证率的函数（工程上为了安全，不取平均数，取排名偏高、比较恶劣的那个数）
    def calc_84th(self, arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0: return 0.0
        return float(np.percentile(arr, 84))

    # 去表格里清点某一楼层到底装了多少个这种构件的函数
    def get_comp_qty(self, cid, floor_num):
        # 去结构表里翻这层楼的数据
        s_match = self.struct_info[(self.struct_info['易损性编号'] == cid) & 
                                   (self.struct_info['起始楼层'] <= floor_num) & 
                                   (self.struct_info['终止楼层'] >= floor_num)]
        # 去非结构表里翻这层楼的数据
        ns_match = self.nonstruct_info[(self.nonstruct_info['易损性编号'] == cid) & 
                                       (self.nonstruct_info['起始楼层'] <= floor_num) & 
                                       (self.nonstruct_info['终止楼层'] >= floor_num)]
        
        qty_x, qty_y, qty_none = 0, 0, 0                            # 准备三个袋子装构件：X方向的，Y方向的，无方向的
        for _, row in pd.concat([s_match, ns_match]).iterrows():    # 把找出来的数据一行行过一遍
            qty_x += row.get('X方向易损性数据', 0)                  # 装进X袋子
            qty_y += row.get('Y方向易损性数据', 0)                  # 装进Y袋子
            qty_none += row.get('无方向易损性数据', 0)              # 装进无方向袋子
        return qty_x, qty_y, qty_none                               # 把三个袋子交回去

    # 经济学打折函数：坏的数量越多，修的单价越便宜（工程量规模效应）
    def get_volume_discount(self, qty, thresholds, factors):
        return float(np.interp(qty, thresholds, factors))

    # -------------------------------------------------------------------------
    # 5. 严格对标《国标表3》的破坏等级判定法官
    # -------------------------------------------------------------------------
    def _struct_floor_level(self, p):
        ds2, ds3, ds4 = float(p[2]), float(p[3]), float(p[4]) # 提取出2级、3级、4级损坏状态的构件比例
        
        # I 级：完全没坏（DS2, DS3, DS4全是0）
        if (ds2 == 0.0) and (ds3 == 0.0) and (ds4 == 0.0): return 1 
        # II 级：只有很少一部分（10%以下）有轻微损坏（DS2），而且没有更严重的损坏
        if (ds2 <= 0.10) and (ds3 == 0.0) and (ds4 == 0.0): return 2 
        # III 级：DS2不超过20%，且DS3不超过10%，且绝对没有DS4
        if (ds2 <= 0.20) and (ds3 <= 0.10) and (ds4 == 0.0): return 3
        # IV 级：放宽要求，DS2≤50%，DS3≤20%，DS4≤10%
        if (ds2 <= 0.50) and (ds3 <= 0.20) and (ds4 <= 0.10): return 4
        # V 级：兜底条款！只要有任何一项超标，直接宣判为最严重的5级破坏！
        return 5      
        
    def _nonstruct_floor_level(self, p):
        d1, d2, d3 = float(p[1]), float(p[2]), float(p[3])
        damaged = d1 + d2 + d3
        # I 级：损坏总数不到10%，且都是最轻的d1
        if (d1 <= 0.10) and (d2 == 0.0) and (d3 == 0.0): return 1
        # II 级：损坏总数不到30%，且都是最轻的d1
        if (d1 <= 0.30) and (d2 == 0.0) and (d3 == 0.0): return 2
        # III 级：坏了一半以内，其中d2不到10%，没有最严重的d3
        if (damaged <= 0.50) and (d2 <= 0.10) and (d3 == 0.0): return 3
        # IV 级：坏了一大半，但d2不到一半，d3不到10%
        if (damaged > 0.50) and (d2 <= 0.50) and (d3 <= 0.10): return 4
        # V 级：d2一大半，或者d3超过10%，非常危险！
        if (d2 > 0.50) or (d3 > 0.10): return 5
        return 4 # 漏网之鱼强制保守算作4级

    # -------------------------------------------------------------------------
    # 【第三部分：中枢大脑 —— 执行1000次大楼抗震受损评级运算】
    # -------------------------------------------------------------------------
    def evaluate(self):
        np.random.seed(42) # 锁定随机种子（保证无论谁在你电脑上跑，这1000次“随机”摇出来的地震都是一模一样的，方便写论文比对）

        # 调出咱们之前写好的“克隆术”，把X、Y方向的位移和加速度分别扩充成1000组数据
        sim_drifts_x = self.expand_edp_matrix(self.drifts_x)
        sim_drifts_y = self.expand_edp_matrix(self.drifts_y)
        sim_accels_x = self.expand_edp_matrix(self.accels_x) / 9.8   # 除以9.8是把单位换算成“几个g”（重力加速度）
        sim_accels_y = self.expand_edp_matrix(self.accels_y) / 9.8

        # --- 设置施工队修复楼层时的效率折扣 ---
        nf = self.num_floors
        if nf >= 12:   n_S = 1.10   # 楼太高，工人坐电梯运料慢，时间要乘以1.1
        elif nf >= 7:  n_S = 1.08
        elif nf >= 4:  n_S = 1.05
        else:          n_S = 1.00   # 3层及以下的小楼，不用打折扣

        # 各个工种（比如刷墙、修管线）每天每100平米能修多少。W1到W8代表不同工种
        W_DENSITY = {'W1': 2.0, 'W2': 2.0, 'W3': 1.0, 'W4': 1.0, 'W5': 1.0, 'W6': 3.0, 'W7': 2.0, 'W8': 2.0}
        NMAX_COEF = 0.026  # 限制最多只能派多少个工人进场（工人太多施展不开）

        # 准备一堆空的“记账本”，用来记录每层楼修了多少钱、花了多少时间、伤亡多少人
        results_c_str = {f: [] for f in range(nf)}
        results_c_dsp = {f: [] for f in range(nf)}
        results_c_acc = {f: [] for f in range(nf)}
        results_g_str, results_g_dsp, results_g_acc, results_g_tot = [], [], [], []
        results_time_s1 = {f: [] for f in range(nf)}
        results_time_s2 = {f: [] for f in range(nf)}
        results_time_bldg, results_bldg_s1, results_bldg_s2 = [], [], []     
        results_injuries, results_deaths = [], []

        # -------------------------------------------------------------------------
        # 国标严苛化：计算整栋大楼的总人口基数（作为最后算伤亡率的分母，干干净净，不掺杂任何影响系数）
        # -------------------------------------------------------------------------
        if '楼层人口密度' in self.floor_info.columns:
            flr_zet = self.floor_info['楼层人口密度'].astype(float).values
        else:
            flr_zet = np.full(len(self.floor_info), 0.5, dtype=float) # 没写的话，默认按办公楼标准：每平米0.5个人
            
        flr_areas = self.floor_info['楼层面积（m^2）'].astype(float).values
        
        # 用 每平米人口数 × 楼层面积，算出大楼总人口
        total_weighted = float((flr_zet * flr_areas).sum())
        total_weighted_area = float(flr_areas.sum()) 

        print("-> 正在执行 1000 次蒙特卡洛模拟 (纯标准解析期望版) ...")
        
        # =========================================================================
        # 🌟 核心大循环：开始在电脑里对大楼进行 1000 次地震轰炸！
        # =========================================================================
        for sim_idx in range(self.num_simulations):
            # 每次地震开始前，把伤亡和损失记账本清零
            sim_inj, sim_death = 0.0, 0.0
            sim_g_s, sim_g_d, sim_g_a = 0.0, 0.0, 0.0
            sim_floor_times = {}   

            # 从第1层开始，一层层往上检查破坏情况
            for f_idx, floor in enumerate(self.floor_info.index):
                floor_num = int(''.join(filter(str.isdigit, str(floor)))) # 提取出纯数字的楼层号
                floor_area = float(self.floor_areas[f_idx])               # 拿到本层面积
                floor_influence = float(self.floor_info.loc[floor, '楼层影响系数']) if '楼层影响系数' in self.floor_info.columns else 1.0
                pop_density = float(self.floor_info.loc[floor, '楼层人口密度']) if '楼层人口密度' in self.floor_info.columns else 0.5

                # 提取刚才克隆出来的、属于【这一次循环】的地震摇晃数据
                dx = sim_drifts_x[sim_idx, f_idx]
                dy = sim_drifts_y[sim_idx, f_idx]
                ax = sim_accels_x[sim_idx, f_idx]
                ay = sim_accels_y[sim_idx, f_idx]

                f_c_str, f_c_dsp, f_c_acc = 0.0, 0.0, 0.0 # 本层的维修费小计
                Q = {k: 0.0 for k in ['W1','W2','W3','W4','W5','W6','W7','W8']} # 本层的各工种工时小计

                # 准备记录本层结构/非结构的受损零件数量
                struct_ds = np.zeros(5, dtype=float)
                struct_total = 0.0
                nonstruct_ds = np.zeros(4, dtype=float)
                nonstruct_total = 0.0

                # 拿着上面准备好的易损字典说明书，一件一件核对楼层里的零件坏没坏
                for cid, cinfo in self.fragility_db.items():
                    qty_x, qty_y, qty_none = self.get_comp_qty(cid, floor_num)
                    total_qty = qty_x + qty_y + qty_none
                    if total_qty <= 0: continue # 这层没装这个零件，跳过

                    # 算算看这个零件坏得多了能不能打折
                    c_vol = self.get_volume_discount(total_qty, cinfo['cost_vol_thresholds'], cinfo['cost_vol_factors'])
                    t_vol = self.get_volume_discount(total_qty, cinfo['time_vol_thresholds'], cinfo['time_vol_factors'])
                    is_struct = (cinfo['cat'] == '结构构件')

                    # 这是一个内部运算小工具：根据摇晃力度（edp_sub），算出会造成多少钱（exp_c）和多少时间（exp_t）的损失
                    def _analytic(qty_sub, edp_sub):
                        if qty_sub <= 0 or edp_sub <= 1e-6: return 0.0, 0.0, None
                        medians = np.asarray(cinfo['medians'], dtype=float)
                        betas   = np.asarray(cinfo['betas'],   dtype=float)
                        n_ds = len(medians)
                        e = np.ones(n_ds)
                        for j, m in enumerate(medians):
                            if m <= 0: e[j] = 1.0
                            else: e[j] = norm.cdf(np.log(edp_sub / m) / betas[j]) # 用正态分布公式算出被震坏的概率
                        probs = np.zeros(n_ds + 1)
                        probs[0] = 1.0 - e[0]
                        for j in range(n_ds - 1): probs[j + 1] = e[j] - e[j + 1]
                        probs[-1] = e[-1]
                        probs = np.clip(probs, 0, 1)
                        if probs.sum() > 0: probs /= probs.sum()
                        
                        exp_c, exp_t = 0.0, 0.0
                        for j in range(1, n_ds + 1):
                            # 把“受损概率 × 修复单价 × 放大系数 × 批发打折 × 总数量”，算出要赔多少钱
                            exp_c += probs[j] * cinfo['loss_ratios'][j-1] * cinfo['cost'] * cinfo['repair_factors'][j-1] * c_vol * qty_sub
                            exp_t += probs[j] * cinfo['repair_times'][j-1] * t_vol * qty_sub
                        ds_exp = qty_sub * probs # 算出坏了几个
                        return exp_c, exp_t, ds_exp

                    # 根据零件怕什么（加速度还是位移），把对应的受力数据喂给它
                    groups = []
                    if cinfo['edp'] == 'accel': ex, ey, enone = ax, ay, max(ax, ay)
                    else: ex, ey, enone = dx, dy, max(dx, dy)

                    if cid.startswith('A') or cid == 'B.98.Z.Z.Z.161':
                        if qty_x > 0:    groups.append((qty_x, ex))
                        if qty_y > 0:    groups.append((qty_y, ey))
                        if qty_none > 0: groups.append((qty_none, enone))
                    elif cid == 'B.01.A.A.A.001':
                        if total_qty > 0: groups.append((total_qty, np.sqrt(dx**2 + dy**2)))
                    elif cid == 'B.01.B.B.A.001':
                        if total_qty > 0: groups.append((total_qty, np.sqrt(ax**2 + ay**2)))

                    # 把计算出来的钱、时间、坏零件数量，分别记入对应的本本里
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
                                if cid == 'B.98.Z.Z.Z.161': Q['W5'] += t_sub
                                elif cid == 'B.01.A.A.A.001': Q['W4'] += t_sub
                            if ds_sub is not None:
                                ns_len = len(ds_sub)
                                nonstruct_ds[:ns_len] += ds_sub
                                nonstruct_total += qty_sub

                # 把这层楼的维修费乘上“楼层影响系数”（楼越高的越难运料，钱花得更多）
                f_c_str *= floor_influence
                f_c_dsp *= floor_influence
                f_c_acc *= floor_influence

                results_c_str[f_idx].append(f_c_str)
                results_c_dsp[f_idx].append(f_c_dsp)
                results_c_acc[f_idx].append(f_c_acc)
                sim_g_s += f_c_str
                sim_g_d += f_c_dsp
                sim_g_a += f_c_acc

                # -------------------------------------------------------------------------
                # 🌟 伤亡统计法庭：按国标 8.2.2 确定本层唯一最严重的破坏等级
                # （修复了原本会导致数据爆表的索引错位问题）
                # -------------------------------------------------------------------------
                CORRECT_CASUALTY_RATE = {
                    1: (0.0,         0.0),          # Ⅰ级：完好无损，零伤亡
                    2: (1/80000.0,   0.0),          # Ⅱ级：轻微破坏，极低概率受伤
                    3: (1/20000.0,   0.0),          # Ⅲ级
                    4: (1/8000.0,    1/80000.0),    # Ⅳ级
                    5: (1/140.0,     1/800.0),      # Ⅴ级：严重破坏，伤亡率极高！
                }
                
                # 问结构法官：结构骨架处于几级破坏？
                if struct_total > 0: lvl_s = self._struct_floor_level(struct_ds / struct_total)
                else: lvl_s = 1  
                    
                # 问非结构法官：玻璃墙、吊顶处于几级破坏？
                if nonstruct_total > 0: lvl_n = self._nonstruct_floor_level(nonstruct_ds[:4] / nonstruct_total)
                else: lvl_n = 1  
                    
                # 木桶效应：取两个法官判定里最糟糕（数值最大）的那个，作为整层楼的最终定性！
                floor_level = max(lvl_s, lvl_n)
                
                # 翻开名义伤亡率表，查出这个定性对应的受伤和死亡系数
                r_hr, r_dr = CORRECT_CASUALTY_RATE.get(floor_level, (0.0, 0.0))
                
                # 算人头：整层楼的面积 × 人口密度 × 伤亡系数，得出本层的伤亡绝对人数，绝不加高层惩罚系数
                sim_inj   += r_hr * pop_density * floor_area
                sim_death += r_dr * pop_density * floor_area
                
                # -------------------------------------------------------------------------
                # 修复时间工期表安排
                # -------------------------------------------------------------------------
                for k in Q: Q[k] *= floor_influence # 工人工时受高楼层影响被拉长

                n_max = NMAX_COEF * floor_area # 算出本层最多能挤下多少工人同时干活                  
                N = {}
                for wk, den in W_DENSITY.items():
                    N[wk] = max(1e-6, min(n_max, (den / 100.0) * floor_area))  

                T = {wk: (Q[wk] / N[wk] if Q[wk] > 0 else 0.0) for wk in W_DENSITY} # 总工时 ÷ 工人数 = 需要多少天

                # 第一阶段：各工种可以同时开工，所以取花时间最长的那一个做木桶短板
                time_s1 = max(T['W1'], T['W2'])                 
                # 第二阶段：严格对标式(12): 隔断W4、吊顶W5、管线W6必须排着队干(串联相加)，其余可以同时干
                time_s2 = max(T['W3'], T['W4'] + T['W5'] + T['W6'], T['W7'], T['W8'])  

                results_time_s1[f_idx].append(time_s1)
                results_time_s2[f_idx].append(time_s2)
                sim_floor_times[f_idx] = (time_s1 + time_s2) * n_S   # 把两阶段相加，得出本层的总完工时间

            # --- 这一轮虚拟地震结束，保存整栋大楼的总成绩 ---
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
            results_time_bldg.append(max((s1_per_floor[f] + s2_per_floor[f]) * n_S for f in range(nf)))

        # =========================================================================
        # 1000次地震全部跑完！开始筛选和提取成绩单（求 84% 保证率）
        # =========================================================================
        print("-> 1000 次模拟结束，正在提取 84% 保证率判据 ...")
        final_res = {'floor_cost_details': [], 'floor_time_details': []}

        # 整理楼层级的明细表数据
        for f, floor in enumerate(self.floor_info.index):
            fc_str = (self.calc_84th(results_c_str[f]) / self.bldg_total_cost) * 100
            fc_dsp = (self.calc_84th(results_c_dsp[f]) / self.bldg_total_cost) * 100
            fc_acc = (self.calc_84th(results_c_acc[f]) / self.bldg_total_cost) * 100
            fc_tot = fc_str + fc_dsp + fc_acc
            final_res['floor_cost_details'].append((floor, fc_tot, fc_acc, fc_dsp, fc_str))

            ft_s1 = self.calc_84th(results_time_s1[f])
            ft_s2 = self.calc_84th(results_time_s2[f])
            ft_tot = (ft_s1 + ft_s2) * n_S
            final_res['floor_time_details'].append((floor, ft_tot, ft_s1, ft_s2))

        # 整理大楼级的修复费用占比汇总
        final_res['cost_struct'] = (self.calc_84th(results_g_str) / self.bldg_total_cost) * 100
        final_res['cost_disp']   = (self.calc_84th(results_g_dsp) / self.bldg_total_cost) * 100
        final_res['cost_accel']  = (self.calc_84th(results_g_acc) / self.bldg_total_cost) * 100
        final_res['cost_total']  = (self.calc_84th(results_g_tot) / self.bldg_total_cost) * 100

        # -------------------------------------------------------------------------
        # 大楼总工期判定：为了让总报告和楼层明细表能够精准严丝合缝对上，
        # 我们这里强制采用“各楼层修复时间中的最大值”来作为整栋楼的工期（完全对标主流评级平台）
        # -------------------------------------------------------------------------
        if final_res['floor_time_details']:
            final_res['time_total']    = max(row[1] for row in final_res['floor_time_details'])
            final_res['time_s1_total'] = max(row[2] for row in final_res['floor_time_details'])
            final_res['time_s2_total'] = max(row[3] for row in final_res['floor_time_details'])
        else:
            final_res['time_total'] = final_res['time_s1_total'] = final_res['time_s2_total'] = 0.0

        # 最后，把绝对伤亡人数 ÷ 大楼总人数，得出用于打星级的百分比（伤亡率）
        denom = total_weighted if total_weighted > 0 else total_weighted_area
        final_res['injury_rate'] = (self.calc_84th(results_injuries) / denom) if denom > 0 else 0.0
        final_res['death_rate']  = (self.calc_84th(results_deaths) / denom) if denom > 0 else 0.0

        return final_res

    # -------------------------------------------------------------------------
    # 【第四部分：最高法院大法官宣判】
    # 生成报告，并严格对照《GB/T 38591-2020》进行星级评定
    # -------------------------------------------------------------------------
    def generate_report(self):
        try:
            res = self.evaluate() # 去把上面那套极其复杂的运算跑一遍，把结果拿过来
        except Exception as e:
            print(f"致命异常：数据崩溃，请审查你的输入表单合规性！详情：{e}")
            return
            
        # 打印费用和时间的漂亮表格
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

        # -------------------------------------------------------------------------
        # 国标 9.4 星级评定模块：在这里给你的大楼挂星星
        # -------------------------------------------------------------------------
        print(f"\n[当前评估地震水准]: {self.seismic_level}")

        ct, tt = res['cost_total'], res['time_total']
        ir, dr = res['injury_rate'], res['death_rate']

        star_cost, star_time, star_casualty = "无星级", "无星级", "无星级" # 初始默认都是没星级

        # 情况 A：如果遇到百年一遇的【罕遇地震】
        if self.seismic_level == "罕遇地震":
            # 钱方面：修起来花钱不到5%给三星，不到10%给二星
            if ct <= 5.0: star_cost = "三星"
            elif 5.0 < ct <= 10.0: star_cost = "二星"
                
            # 时间方面：7天内能修好给三星，30天内修好给二星
            if tt <= 7.0: star_time = "三星"
            elif 7.0 < tt <= 30.0: star_time = "二星"
                
            # 命方面：伤亡率控制得极低极低给三星，略微放宽一点给二星
            if ir <= 1.0e-4 and dr <= 1.0e-5: star_casualty = "三星"
            elif ir <= 1.0e-3 and dr <= 1.0e-4: star_casualty = "二星"

        # 情况 B：如果遇到中等威力的【设防地震】
        elif self.seismic_level == "设防地震":
            # 中震要求很严格，只要达标统统只给一星，超标直接无星级！
            if ct <= 10.0: star_cost = "一星"
            if tt <= 30.0: star_time = "一星"
            if ir <= 1.0e-3 and dr <= 1.0e-4: star_casualty = "一星"
            
        else:
            print("⚠️ 警告：当前地震水准不满足国标《GB/T 38591-2020》要求，无法定级！")

        # -------------------------------------------------------------------------
        # 最终综合评级：残酷的木桶效应！
        # 只要三项（钱、时间、命）里有一个是最差的，总评级就被拉跨到那个最差的级别。
        # -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# 程序启动器
# -------------------------------------------------------------------------
if __name__ == "__main__":
    evaluator = TrueResilienceEvaluator()
    evaluator.generate_report()