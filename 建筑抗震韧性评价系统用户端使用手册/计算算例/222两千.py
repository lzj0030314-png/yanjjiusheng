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
        self.num_simulations = 2000                                 # 国标规范硬性规定：蒙特卡洛模拟次数不应少于 1000 次
        
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
            'B.98.Z.Z.Z.161': { # 修复了编号脱节问题，位移敏感型非结构构件（隔墙）将正式参与算账
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
        
        # 6. 提取原始工程需求参数响应矩阵 (EDP 矩阵)
        edp_raw = pd.read_excel(xls, '结构响应')                  # 读取结构响应包络表
        
        # 精准切片提取：读取对应楼层数据后转置 (.T)，使矩阵形状变为 (11次模拟行, 楼层列)
        # 6. 提取原始工程需求参数响应矩阵 (EDP 矩阵)
        edp_raw = pd.read_excel(xls, '结构响应')                  
        
        # ⚠️ 必须使用精准切片，严禁使用 1:5 连续切片！
        # 位移角(drift)提取：使用行索引 [2, 3, 4]，完美锚定 1、2、3 层真实物理位移
        self.edp_drift = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.drifts_x = edp_raw.iloc[[2, 3, 4], 1:12].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T 
        self.drifts_y = edp_raw.iloc[[2, 3, 4], 23:34].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T 
        
        # 加速度(accel)提取：使用行索引 [3, 4, 1]，越过地面输入(索引2)，完美修复前端错位，锚定 1、2、顶层真实加速度
        self.edp_accel = edp_raw.iloc[[3, 4, 1], 12:23].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T
        self.accels_x = edp_raw.iloc[[3, 4, 1], 12:23].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T 
        self.accels_y = edp_raw.iloc[[3, 4, 1], 34:45].replace(['--', 'NULL'], 0.0).fillna(0.0).astype(float).values.T


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
        """【统计处理】提取过滤零值后的 84% 保证率恶劣极值"""
        arr = np.array(arr)                                         # 将原始离散数组转化为 Numpy 加速数组
        valid = arr[arr > 1e-6]                                     # 数据筛分：剔除完好无损的零值无效实验次数
        if len(valid) == 0: return 0.0                              # 如果在灾难中始终无损，则 84% 极值就是绝对的 0
        return float(np.exp(np.mean(np.log(valid)) + np.std(np.log(valid)))) # 拟合纯有效数据的对数正态曲线，求取 84% 右偏分位

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
        """执行大规模修缮作业下的规模效应折现计算（即多修多打折）"""
        if qty <= thresholds[0]: return factors[0]                  # 若报修量极少未触及门槛，不予优惠，按原价定损
        elif qty >= thresholds[1]: return factors[1]                # 若报修量过大突破天花板，享受极致折后底价
        else:
            # 报修量处于起步与满级之间，通过一次线性插值计算出其应该享受的专属折扣梯度
            ratio = (qty - thresholds[0]) / (thresholds[1] - thresholds[0])
            return factors[0] - ratio * (factors[0] - factors[1])

    def evaluate(self):
        """
        中枢控制台：全量 1000 次蒙特卡洛引擎（费用、时间、伤亡均使用随机抽样实现）
        满足国标 A.5.1/A.5.2/A.5.3 要求的对每个构件独立采样损伤状态。
        """
        np.random.seed(42)                                          # 钉死随机发生器种子，保证代码多次运行的科学可复现性
        
        # 获取 1000 维蒙特卡洛扩展后的地震摇晃数据海
        sim_drifts = self.expand_edp_matrix(self.edp_drift)
        sim_accels = self.expand_edp_matrix(self.edp_accel)
        sim_drifts_x = self.expand_edp_matrix(self.drifts_x)
        sim_drifts_y = self.expand_edp_matrix(self.drifts_y)
        sim_accels_x = self.expand_edp_matrix(self.accels_x) / 9.8  # 加速度数据除以 9.8 实现单位转换(m/s^2 -> g)
        sim_accels_y = self.expand_edp_matrix(self.accels_y) / 9.8 
        
        # 准备各类用于接收每一次虚拟地震结算报表的记账仓库
        results_c_str = {f: [] for f in range(self.num_floors)}     # 记录每层模拟结构花费
        results_c_dsp = {f: [] for f in range(self.num_floors)}     # 记录每层模拟位移非结构花费
        results_c_acc = {f: [] for f in range(self.num_floors)}     # 记录每层模拟加速度非结构花费
        results_g_str, results_g_dsp, results_g_acc = [], [], []    # 记录整栋楼三大项开销汇总
        results_g_tot = []                                          # 记录整栋大楼历劫总重修花费
        
        results_time_s1 = {f: [] for f in range(self.num_floors)}   # 记录每层结构第一期修缮排期耗时
        results_time_s2 = {f: [] for f in range(self.num_floors)}   # 记录每层非结构第二期修缮排期耗时
        results_injuries, results_deaths = [], []                   # 记录历次大地震整楼产生的哀痛伤亡数

        # ==============================================================================
        # 🚀 启动全维度蒙特卡洛大碰撞循环（1000次）
        # ==============================================================================
        print("-> 正在执行 1000 次全蒙特卡洛模拟 (时间、伤亡及【修复费用】联合随机抽样) ...")
        for sim_idx in range(self.num_simulations):                 # 循环 1000 次，每次代表一次独立的平行宇宙地震
            sim_inj, sim_death = 0, 0                               # 每开启一次新宇宙，清空伤亡总计数板
            sim_g_str_c, sim_g_dsp_c, sim_g_acc_c = 0.0, 0.0, 0.0   # 清空大楼建安总开销总板
            sim_floor_time = {}                                     # 初始化各层完工时点追踪记录
            
            f_cost_struct_sim = {f: 0.0 for f in range(self.num_floors)} # 单次模拟中各层的结构计费临时板
            f_cost_dsp_sim = {f: 0.0 for f in range(self.num_floors)}    # 单次模拟中各层的位移非结构计费临时板
            f_cost_acc_sim = {f: 0.0 for f in range(self.num_floors)}    # 单次模拟中各层的加速度非结构计费临时板

            # 对本期虚拟地震中的每一层建筑逐一扫描定损
            for f_idx, floor in enumerate(self.floor_info.index):   
                floor_num = int(''.join(filter(str.isdigit, str(floor)))) # 将带有汉字的楼层标签洗成干净的数字
                floor_area = self.floor_areas[f_idx]                # 抽出该层的地理面积供测算工人容纳密度
                pop_density = self.floor_info.loc[floor, '楼层人口密度'] # 抽出该层每平米常规塞了多少人
                # 获取高空施工影响惩罚系数，没填的按 1.0 平地对待
                floor_influence = float(self.floor_info.loc[floor, '楼层影响系数']) if '楼层影响系数' in self.floor_info.columns else 1.0

                # 精确调取该次大地震作用于该特定楼层的破坏力（EDP）
                dx = sim_drifts_x[sim_idx, f_idx]                   # 本轮 X向 层间位移角
                dy = sim_drifts_y[sim_idx, f_idx]                   # 本轮 Y向 层间位移角
                ax = sim_accels_x[sim_idx, f_idx]                   # 本轮 X向 楼面加速度
                ay = sim_accels_y[sim_idx, f_idx]                   # 本轮 Y向 楼面加速度
                
                f_c_str, f_c_dsp, f_c_acc = 0.0, 0.0, 0.0           # 归零本层的三大类开销罚单
                q_w1, q_w3, q_w4, q_w5 = 0.0, 0.0, 0.0, 0.0         # 归零本层的四大专业施工队的修补人时总包池
                max_ds_floor = 0                                    # 初始化本楼层的最恶劣结构构件破坏等级状态针
                
                # 对本层的所有清单构件展开排雷式多项抽样检视
                for cid, cinfo in self.fragility_db.items():
                    qty_x, qty_y, qty_none = self.get_comp_qty(cid, floor_num) # 查户口：这层这个东西到底有几个，都在什么方向
                    total_qty_all = qty_x + qty_y + qty_none        # 计算构件总计受检数
                    if total_qty_all <= 0: continue                 # 如果本层压根没装这玩意，跳过省算力
                    
                    # 取出适用于全楼总修缮量的团购折后券（分别对应省钱比例与省时比例）
                    c_vol_disc = self.get_volume_discount(self.bldg_comp_qtys[cid], cinfo['cost_vol_thresholds'], cinfo['cost_vol_factors'])
                    t_vol_disc = self.get_volume_discount(self.bldg_comp_qtys[cid], cinfo['time_vol_thresholds'], cinfo['time_vol_factors'])
                    
                    def mc_sample_cost_and_time(qty_sub, edp_sub, comp_info, c_disc, t_disc):
                        """内部核函数：基于单向受力的多项分布骰子，统合计算破碎件的罚款与所需抢修时间"""
                        if qty_sub <= 0 or edp_sub <= 1e-6: return 0.0, 0.0, 0  # 没有受害者或没有杀伤力，当场判零退出
                        n_ds = len(comp_info['medians'])                        # 获取可判定的级数
                        # 计算本受力冲击冲破各道中位值防线的正态累计超越概率
                        e_probs = norm.cdf(np.log(edp_sub / np.array(comp_info['medians'])) / np.array(comp_info['betas']))
                        
                        # 把超越概率切分到互斥的区间独立概率内
                        probs = np.zeros(n_ds + 1)
                        probs[0] = 1.0 - e_probs[0]                             # 坚守住了第一道防线的幸运儿概率(完好)
                        for j in range(n_ds - 1): probs[j+1] = e_probs[j] - e_probs[j+1] # 落在各损伤段的区间概率
                        probs[-1] = e_probs[-1]                                 # 全线崩盘粉碎到底的彻底毁坏概率
                        probs = np.clip(probs, 0, 1)                            # 防范计算机浮点误差造出负概率
                        probs /= probs.sum()                                    # 闭环校验确保所有落点概率相加恒等于 1
                        
                        # 💥 【蒙特卡洛核心裁决】：抛多项分布随机骰子，让 `qty_sub` 个构件随机分配入不同的受损箱子
                        ds_counts = np.random.multinomial(int(round(qty_sub)), probs)
                        
                        batch_cost, batch_time = 0.0, 0.0                       # 本批次追责清算池
                        max_ds_local = 0                                        # 追踪本批次最惨烈的碎片属于几级
                        
                        for ds_idx in range(1, n_ds + 1):                       # 略过 0 级完好的箱子不罚钱
                            count = ds_counts[ds_idx]                           # 数一数有几个构件掉进了这级破损箱
                            if count > 0:
                                max_ds_local = max(max_ds_local, ds_idx)        # 更新最惨烈记录
                                # 费用累加 = 碎裂个数 * 该级别毁坏赔率 * 原价 * 施工难度放大因子 * 团购折惠卡
                                batch_cost += count * comp_info['loss_ratios'][ds_idx-1] * comp_info['cost'] * comp_info['repair_factors'][ds_idx-1] * c_disc
                                # 时间累加 = 碎裂个数 * 该级别修复耗日 * 时间折惠卡
                                batch_time += count * comp_info['repair_times'][ds_idx-1] * t_disc
                                
                        return batch_cost, batch_time, max_ds_local             # 发送清算总清单与损伤最惨记录

                    # 🌟 特性保留 2：实施 X/Y 方向精准受力解耦（还原真实抗震设计不盲目用最大值）
                    cost_x, time_x, ds_x = 0.0, 0.0, 0                          # X 向记账专员
                    cost_y, time_y, ds_y = 0.0, 0.0, 0                          # Y 向记账专员
                    cost_n, time_n, ds_n = 0.0, 0.0, 0                          # 无方向复合受力记账专员

                    if cid.startswith('A') or cid == 'B.98.Z.Z.Z.161': 
                        # 结构构件与隔墙严格执行各行其道受力法则
                        if qty_x > 0: cost_x, time_x, ds_x = mc_sample_cost_and_time(qty_x, dx, cinfo, c_vol_disc, t_vol_disc)
                        if qty_y > 0: cost_y, time_y, ds_y = mc_sample_cost_and_time(qty_y, dy, cinfo, c_vol_disc, t_vol_disc)
                        if qty_none > 0: cost_n, time_n, ds_n = mc_sample_cost_and_time(qty_none, max(dx, dy), cinfo, c_vol_disc, t_vol_disc) # 无方向构件取两个方向的最不利包络值来打
                    elif cid == 'B.01.A.A.A.001': 
                        # 玻璃幕墙类由于四面环绕，对其执行双向平方和开根号的双向位移组合施加冲击
                        if total_qty_all > 0: cost_n, time_n, ds_n = mc_sample_cost_and_time(total_qty_all, np.sqrt(dx**2 + dy**2), cinfo, c_vol_disc, t_vol_disc)
                    elif cid == 'B.01.B.B.A.001': 
                        # 吊顶设备类无处可藏，承受双向加速度合成的组合惯性力
                        if total_qty_all > 0: cost_n, time_n, ds_n = mc_sample_cost_and_time(total_qty_all, np.sqrt(ax**2 + ay**2), cinfo, c_vol_disc, t_vol_disc)

                    tot_cost_comp = cost_x + cost_y + cost_n                    # 汇总该类型下所有存活体的修缮账单
                    tot_time_comp = time_x + time_y + time_n                    # 汇总该类所需抢修人日
                    max_ds_comp = max(ds_x, ds_y, ds_n)                         # 寻找其中伤得最重的一个
                    max_ds_floor = max(max_ds_floor, max_ds_comp)               # 并将最惨烈纪录上报给本楼层安监局

                    # 各自对口归类收账
                    if cid.startswith('A'):
                        f_c_str += tot_cost_comp                                # 把钱放入结构总造价簿
                        q_w1 += tot_time_comp                                   # 把活派给第一包工队
                    else:
                        if cinfo['edp'] == 'accel':
                            f_c_acc += tot_cost_comp                            # 把钱放入加速度敏感设备受损总账
                            q_w5 += tot_time_comp                               # 把活派给第五包工队
                        else:
                            f_c_dsp += tot_cost_comp                            # 把钱放入位移敏感非结构总账
                            if cid == 'B.98.Z.Z.Z.161': q_w4 += tot_time_comp   # 把隔墙的活派给第四包工队
                            elif cid == 'B.01.A.A.A.001': q_w3 += tot_time_comp # 把幕墙的活派给第三包工队

                # === 单层核算结案阶段 ===
                # 楼层所处越高，运料与修复越慢且费钱，乘以楼层影响系数予以惩罚放大
                f_c_str *= floor_influence
                f_c_dsp *= floor_influence
                f_c_acc *= floor_influence

                # 单层罚款登报
                f_cost_struct_sim[f_idx] = f_c_str
                f_cost_dsp_sim[f_idx] = f_c_dsp
                f_cost_acc_sim[f_idx] = f_c_acc

                # 大楼总罚款累加
                sim_g_str_c += f_c_str
                sim_g_dsp_c += f_c_dsp
                sim_g_acc_c += f_c_acc

                # 人员伤亡判决：依循本层出现的最惨烈破坏等级状态针，依照规范死伤率结算命数
                floor_pop = floor_area * pop_density                            # 测算发生灾难时刻该楼层的活人总数
                if max_ds_floor >= 4:                                           # 若出现毁坏(V级)，死伤最为惨重
                    sim_inj += floor_pop * (1/140)
                    sim_death += floor_pop * (1/800)
                elif max_ds_floor == 3:                                         # 若出现严重破坏(IV级)
                    sim_inj += floor_pop * (1/8000)
                    sim_death += floor_pop * (1/80000)
                elif max_ds_floor == 2:                                         # 若出现中度破坏(III级)
                    sim_inj += floor_pop * (1/20000)
                elif max_ds_floor == 1:                                         # 若只发生轻微裂缝破坏(II级)
                    sim_inj += floor_pop * (1/80000)

                # 施工排期判决：依照平米数与拥挤法有限定可派驻的极限修复工人数
                n_max = (0.026 * floor_area) / floor_influence
                n_w1 = max(1e-6, min(n_max, ((2.0 / 100) * floor_area) / floor_influence)) # 结构修复队上架人头数
                n_w3 = max(1e-6, min(n_max, ((1.0 / 100) * floor_area) / floor_influence)) # 非结构幕墙队上架人头数
                n_w4 = max(1e-6, min(n_max, ((1.0 / 100) * floor_area) / floor_influence)) # 非结构隔墙队上架人头数
                n_w5 = max(1e-6, min(n_max, ((1.0 / 100) * floor_area) / floor_influence)) # 设备队上架人头数
                
                # 总工时 / 可用工人 = 各工序硬性完成所需日历天
                t_w1 = q_w1 / n_w1
                t_w3 = q_w3 / n_w3
                t_w4 = q_w4 / n_w4
                t_w5 = q_w5 / n_w5
                
                time_s1 = t_w1                                # 第一阶段必须先单独修结构，记为其独立工期
                time_s2 = max(t_w3, t_w4 + t_w5)              # 结构修完后各路非结构人马平行同时进场干活，故该期耗时取决于那个最拖后腿的队伍（最大值）
                
                results_time_s1[f_idx].append(time_s1)        # 封存一期工单
                results_time_s2[f_idx].append(time_s2)        # 封存二期工单
                sim_floor_time[f_idx] = time_s1 + time_s2     # 本层全盘完工总计天数

            # 单次大楼宏观仿真收尾清账入库
            for f in range(self.num_floors):
                results_c_str[f].append(f_cost_struct_sim[f])
                results_c_dsp[f].append(f_cost_dsp_sim[f])
                results_c_acc[f].append(f_cost_acc_sim[f])

            results_g_str.append(sim_g_str_c)
            results_g_dsp.append(sim_g_dsp_c)
            results_g_acc.append(sim_g_acc_c)
            results_g_tot.append(sim_g_str_c + sim_g_dsp_c + sim_g_acc_c) # 封存第 sim_idx 次的建安造价
            
            results_injuries.append(sim_inj)                              # 封存伤员
            results_deaths.append(sim_death)                              # 封存亡灵

        # ==============================================================================
        # 提取极值：将 1000 次波动的离散分布强制归一到规范要求的 84% 保证率统计线上
        # ==============================================================================
        print("-> 1000 次全蒙特卡洛引擎仿真结束，正在提取 84% 保证率数学判据 ...")
        final_res = {'floor_cost_details': [], 'floor_time_details': []}
        
        for f, floor in enumerate(self.floor_info.index):
            # 抽出楼层各项费用极限值，除以重置本源总价转化为直观的损耗比百分数
            fc_str = (self.calc_84th(results_c_str[f]) / self.bldg_total_cost) * 100
            fc_dsp = (self.calc_84th(results_c_dsp[f]) / self.bldg_total_cost) * 100
            fc_acc = (self.calc_84th(results_c_acc[f]) / self.bldg_total_cost) * 100
            fc_tot = fc_str + fc_dsp + fc_acc
            final_res['floor_cost_details'].append((floor, fc_tot, fc_acc, fc_dsp, fc_str))
            
            # 抽出楼层极度拖沓时长
            ft_s1 = self.calc_84th(results_time_s1[f])
            ft_s2 = self.calc_84th(results_time_s2[f])
            ft_tot = ft_s1 + ft_s2
            final_res['floor_time_details'].append((floor, ft_tot, ft_s1, ft_s2))

        # 敲定抗震三大终审主指标
        final_res['cost_struct'] = (self.calc_84th(results_g_str) / self.bldg_total_cost) * 100
        final_res['cost_disp'] = (self.calc_84th(results_g_dsp) / self.bldg_total_cost) * 100
        final_res['cost_accel'] = (self.calc_84th(results_g_acc) / self.bldg_total_cost) * 100
        final_res['cost_total'] = (self.calc_84th(results_g_tot) / self.bldg_total_cost) * 100
        
        # 将大楼交房日压在修复耗时最漫长的那层短板楼上
        max_time_floor_detail = max(final_res['floor_time_details'], key=lambda x: x[1])
        final_res['time_total'] = max_time_floor_detail[1]
        final_res['time_s1_total'] = max_time_floor_detail[2]
        final_res['time_s2_total'] = max_time_floor_detail[3]

        total_population = (self.floor_info['楼层面积（m^2）'] * self.floor_info['楼层人口密度']).sum()
        final_res['injury_rate'] = (self.calc_84th(results_injuries) / total_population)  if total_population > 0 else 0
        final_res['death_rate'] = (self.calc_84th(results_deaths) / total_population)  if total_population > 0 else 0

        return final_res                                            # 结算完毕转交打印法庭

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
        # GB/T 38591-2020 大满贯星级判罚书
        # ==========================================
        print(f"\n[当前评估地震水准]: {self.seismic_level}")
        
        if '设防' in self.seismic_level:
            # 设防地震评估门槛极严，封顶只授予一星
            star_cost = "一星" if res['cost_total'] <= 10.0 else "无星级"
            star_time = "一星" if res['time_total'] <= 30.0 else "无星级"
            star_casualty = "一星" if (res['injury_rate'] <= 1.0e-3 and res['death_rate'] <= 1.0e-4) else "无星级"
            
        elif '罕遇' in self.seismic_level:
            # 罕遇地震允许参评全系满贯三星
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

        # 无情木桶绞肉机：将星级数字化降维比较，斩断最短的那块木板作为大楼的宿命
        star_map = {"三星": 3, "二星": 2, "一星": 1, "无星级": 0}
        min_star_val = min(star_map[star_cost], star_map[star_time], star_map[star_casualty])
        
        # 升维解码，还原中文至高荣耀
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