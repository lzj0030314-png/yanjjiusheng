import os
import re
import math
import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm


class GBT38591ResilienceEvaluator:
    """GB/T 38591-2020 建筑抗震韧性评价（1000次蒙特卡洛）

    计算链：
        原始EDP -> 联合概率分布扩充 -> 构件损伤状态随机抽样
        -> 修复费用/修复时间/人员伤亡 -> 对数正态84%保证率 -> 评级

    说明：
    1. 易损性参数优先直接读取用户导出的Excel，不再把易损性参数硬编码在程序里。
    2. 对“使用上一楼层EDP=是”的构件，按Excel字段自动取上一楼层EDP。
    3. X/Y方向EDP联合扩充，保持工程需求参数相关性。
    4. 每次蒙特卡洛对损伤状态进行随机抽样，而不是直接使用损伤概率的期望值。
    5. 修复工程量折减系数按本次模拟的实际损伤数量计算。
    6. 本轮对标采用1000次样本的经验84%分位（np.percentile），因为你给出的标准软件结果
       与这种统计方式明显更接近；不删除0值。
    7. 楼层费用84%后求和，楼层时间84%后按楼层最大值取建筑总修复时间，用于复现截图汇总。
    """

    DEFAULT_FILE = "中震三层数据导出版.xls"
    N_SIM = 1000
    RANDOM_SEED = 42

    # "strict": 对建筑总体指标集合直接取经验84%分位；
    # "reference": 复现截图汇总方式：费用=各楼层84%之和，时间=max(各楼层84%)。
    REPORT_MODE = "reference"

    # 结构化构件的标准修复工作映射
    # W1结构、W2楼梯、W3围护、W4隔断、W5吊顶及附属、W6管线、W7大型设备、W8电梯
    WORK_MAP = {
        "A.98.Z.Z.Z.813": "W1",  # 柱
        "A.98.Z.Z.Z.814": "W1",  # 梁
        "B.98.Z.Z.Z.161": "W3",  # Excel构件信息中的编号，与160易损性库对应
        "B.01.A.A.A.001": "W3",  # 玻璃幕墙
        "B.01.B.B.A.001": "W5",  # 数据描述为吊顶
    }

    

    # GB/T 38591-2020 表1：单位工作的人数需求。
    # 面积型：人/100m2；构件型：人/个或人/台。
    WORK_UNIT_DEMAND = {
        "W1": ("area", 2.0 / 100.0),
        "W2": ("unit", 2.0),
        "W3": ("area", 1.0 / 100.0),
        "W4": ("area", 1.0 / 100.0),
        "W5": ("area", 1.0 / 100.0),
        "W6": ("area", 1.0 / 100.0),
        "W7": ("unit", 3.0),
        "W8": ("unit", 2.0),
    }

    # GB/T 38591-2020 表4：不同破坏等级名义受伤率、名义死亡率。
    CASUALTY_RATE = {
        1: (0.0, 0.0),
        2: (1.0 / 80000.0, 0.0),
        3: (1.0 / 20000.0, 0.0),
        4: (1.0 / 8000.0, 1.0 / 80000.0),
        5: (1.0 / 140.0, 1.0 / 800.0),
    }

    def __init__(self, file_name=DEFAULT_FILE, n_sim=N_SIM, seed=RANDOM_SEED):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = file_name if os.path.isabs(file_name) else os.path.join(self.base_dir, file_name)
        self.num_simulations = int(n_sim)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._load_excel()
        self._build_fragility_database()
        self._prepare_component_lookup()
        self._prepare_edp_database()

    # ------------------------------------------------------------------
    # Excel读取
    # ------------------------------------------------------------------
    def _load_excel(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"找不到Excel文件：{self.file_path}")

        try:
            self.xls = pd.ExcelFile(self.file_path)
        except ImportError as e:
            raise ImportError(
                "读取.xls需要xlrd。请执行：pip install xlrd>=2.0.1"
            ) from e

        required = [
            "建筑信息1", "建筑信息2", "结构构件易损性", "结构构件信息",
            "非结构构件易损性", "非结构构件信息", "地震信息1", "结构响应"
        ]
        missing = [s for s in required if s not in self.xls.sheet_names]
        if missing:
            raise ValueError(f"Excel缺少工作表：{missing}")

        b1 = pd.read_excel(self.xls, "建筑信息1").iloc[0]
        self.total_area = float(b1["建筑总面积（平方米）"])
        self.unit_cost = float(b1["单位造价（元/平方米）"])
        self.bldg_total_cost = self.total_area * self.unit_cost

        self.floor_info = pd.read_excel(self.xls, "建筑信息2").copy()
        self.floor_info["楼层（层）"] = pd.to_numeric(self.floor_info["楼层（层）"], errors="coerce").astype(int)
        self.floor_info = self.floor_info.sort_values("楼层（层）").set_index("楼层（层）")
        self.floor_numbers = self.floor_info.index.to_list()
        self.num_floors = len(self.floor_numbers)

        self.struct_info = pd.read_excel(self.xls, "结构构件信息").copy()
        self.nonstruct_info = pd.read_excel(self.xls, "非结构构件信息").copy()
        self.struct_fragility_raw = pd.read_excel(self.xls, "结构构件易损性").copy()
        self.nonstruct_fragility_raw = pd.read_excel(self.xls, "非结构构件易损性").copy()

        seismic = pd.read_excel(self.xls, "地震信息1").iloc[0]
        self.seismic_level = str(seismic.get("地震水准", "")).strip()
        self.wave_count = int(float(seismic.get("地震波数", 0))) if pd.notna(seismic.get("地震波数", np.nan)) else 0

        self.usage = str(b1.get("使用功能", ""))

        # 楼层人口密度 ζ_k；如果没有该列，按0.5兜底，但本项目Excel有该列。
        self.floor_areas = self.floor_info["楼层面积（m^2）"].astype(float).to_numpy()
        self.floor_lambda = self.floor_info["楼层影响系数"].astype(float).to_numpy()
        self.floor_density = self.floor_info["楼层人口密度"].astype(float).to_numpy()

    # ------------------------------------------------------------------
    # 易损性数据库：直接从Excel构造，不硬编码中位值、beta、损失系数、修复系数、工时
    # ------------------------------------------------------------------
    @staticmethod
    def _ffill_columns(df):
        cols = [
            "易损性编号", "构件名称", "构件造价", "构件单位", "构件类型",
            "EDP类型", "是否具有方向性", "使用上一楼层的EDP", "所属修复工作内容"
        ]
        for c in cols:
            if c in df.columns:
                df[c] = df[c].ffill()
        return df

    def _build_one_fragility(self, df, is_struct):
        df = self._ffill_columns(df.copy())
        db = {}

        for cid, g in df.groupby("易损性编号", sort=False):
            cid = str(cid).strip()
            if not cid or cid.lower() == "nan":
                continue

            states = []
            for _, r in g.iterrows():
                median = pd.to_numeric(r.get("中位值"), errors="coerce")
                beta = pd.to_numeric(r.get("对数标准差"), errors="coerce")
                loss = pd.to_numeric(r.get("损失系数"), errors="coerce")
                repair_factor = pd.to_numeric(r.get("修复系数"), errors="coerce")
                repair_time = pd.to_numeric(r.get("修复时间（人/天）"), errors="coerce")

                # 空白分隔行跳过。
                if pd.isna(median) or pd.isna(beta) or pd.isna(loss) or pd.isna(repair_factor) or pd.isna(repair_time):
                    continue

                # 某些易损性条目用 median=0,beta=0 的零损失行作为占位（本项目玻璃幕墙）。
                # 该行不构成可计算的易损性阈值，跳过；真正损伤状态从第一个有效lognormal阈值开始。
                if float(median) <= 0 or float(beta) <= 0:
                    continue

                states.append({
                    "median": float(median),
                    "beta": float(beta),
                    "loss_ratio": float(loss),
                    "repair_factor": float(repair_factor),
                    "repair_time": float(repair_time),
                    "name": str(r.get("损伤状态名称", "")),
                })

            if not states:
                continue

            first = g.iloc[0]
            cost = pd.to_numeric(first.get("构件造价"), errors="coerce")
            if pd.isna(cost):
                raise ValueError(f"易损性 {cid} 缺少构件造价。")

            edp_raw = str(first.get("EDP类型", "")).strip().lower()
            if "acceleration" in edp_raw or "加速度" in edp_raw:
                edp_type = "accel"
            elif "drift" in edp_raw or "位移" in edp_raw:
                edp_type = "drift"
            else:
                raise ValueError(f"无法识别易损性 {cid} 的EDP类型：{first.get('EDP类型')}")

            directional = str(first.get("是否具有方向性", "")).strip() == "是"
            previous_floor = str(first.get("使用上一楼层的EDP", "")).strip() == "是"

            cmin = pd.to_numeric(first.get("最小工程量折减数量（修复费用）"), errors="coerce")
            cfmin = pd.to_numeric(first.get("工程量折减系数（修复费用）"), errors="coerce")
            cmax = pd.to_numeric(first.get("最大工程量折减数量（修复费用）"), errors="coerce")
            cfmax = pd.to_numeric(first.get("工程量折减系数（修复费用）.1"), errors="coerce")

            tmin = pd.to_numeric(first.get("最小工程量折减数量（修复时间）"), errors="coerce")
            tfmin = pd.to_numeric(first.get("工程量折减系数（修复时间）"), errors="coerce")
            tmax = pd.to_numeric(first.get("最大工程量折减数量（修复时间）"), errors="coerce")
            tfmax = pd.to_numeric(first.get("工程量折减系数（修复时间）.1"), errors="coerce")

            db[cid] = {
                "id": cid,
                "name": str(first.get("构件名称", "")),
                "cost": float(cost),
                "edp_type": edp_type,
                "directional": directional,
                "previous_floor_edp": previous_floor,
                "is_struct": bool(is_struct),
                "states": states,
                "cost_discount": (self._safe_float(cmin, 1.0), self._safe_float(cmax, 50.0), self._safe_float(cfmin, 1.0), self._safe_float(cfmax, 1.0)),
                "time_discount": (self._safe_float(tmin, 1.0), self._safe_float(tmax, 50.0), self._safe_float(tfmin, 1.0), self._safe_float(tfmax, 1.0)),
            }

        return db

    @staticmethod
    def _safe_float(x, default):
        try:
            return default if pd.isna(x) else float(x)
        except Exception:
            return default

    def _build_fragility_database(self):
        sdb = self._build_one_fragility(self.struct_fragility_raw, True)
        ndb = self._build_one_fragility(self.nonstruct_fragility_raw, False)
        self.fragility_db = {**sdb, **ndb}


        # 只保留实际在构件信息表中出现的构件类型，避免空类型参与模拟。
        used_ids = set(self.struct_info["易损性编号"].dropna().astype(str).str.strip()) | set(self.nonstruct_info["易损性编号"].dropna().astype(str).str.strip())
        missing = [cid for cid in used_ids if cid not in self.fragility_db]
        if missing:
            raise ValueError(f"以下构件在信息表中存在，但在易损性库找不到：{missing}")

    # ------------------------------------------------------------------
    # 构件信息查询
    # ------------------------------------------------------------------
    def _component_rows(self, floor, cid):
        parts = []
        for table in (self.struct_info, self.nonstruct_info):
            if "易损性编号" not in table.columns:
                continue
            g = table[
                (table["易损性编号"].astype(str).str.strip() == cid)
                & (pd.to_numeric(table["起始楼层"], errors="coerce") <= floor)
                & (pd.to_numeric(table["终止楼层"], errors="coerce") >= floor)
            ]
            if not g.empty:
                parts.append(g)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def get_component_quantity(self, cid, floor):
        g = self._component_rows(floor, cid)
        if g.empty:
            return 0.0, 0.0, 0.0

        qx = pd.to_numeric(g["X方向易损性数据"], errors="coerce").fillna(0).sum()
        qy = pd.to_numeric(g["Y方向易损性数据"], errors="coerce").fillna(0).sum()
        qn = pd.to_numeric(g["无方向易损性数据"], errors="coerce").fillna(0).sum()
        return float(qx), float(qy), float(qn)

    def _prepare_component_lookup(self):
        self.component_qty = {}
        for floor in self.floor_numbers:
            self.component_qty[floor] = {}
            for cid in self.fragility_db:
                self.component_qty[floor][cid] = self.get_component_quantity(cid, floor)

    # ------------------------------------------------------------------
    # EDP读取和联合扩充
    # ------------------------------------------------------------------
    def _prepare_edp_database(self):
        df = pd.read_excel(self.xls, "结构响应")
        self.edp_raw = df

        # 列索引：0为楼层；1:12 X漂移；12:23 X加速度；23:34 Y漂移；34:45 Y加速度。
        # 结构响应中：漂移只有1~3层；加速度有0~3层。
        def block(rows, start, end):
            x = df.iloc[rows, start:end].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            return x

        self.drift_floors = [1, 2, 3]
        self.accel_floors = [0, 1, 2, 3]

        drift_rows = [2, 3, 4]  # header=0后，对应楼层1~3
        accel_rows = [1, 2, 3, 4]  # 对应楼层0~3

        self.raw_dx = block(drift_rows, 1, 12)
        self.raw_ax = block(accel_rows, 12, 23)
        self.raw_dy = block(drift_rows, 23, 34)
        self.raw_ay = block(accel_rows, 34, 45)

        if self.raw_dx.shape[1] != 11:
            raise ValueError("结构响应X方向层间位移角应包含11条地震波。")

        # 14个参数联合扩充：3层X/Y漂移 + 4个楼层位置X/Y加速度。
        # 加速度0层是地面/基底参考，用于“使用上一楼层EDP”的1层构件。
        self.edp_keys = []
        mats = []
        for i, floor in enumerate(self.drift_floors):
            self.edp_keys.append(("drift_x", floor))
            mats.append(self.raw_dx[i])
        for i, floor in enumerate(self.drift_floors):
            self.edp_keys.append(("drift_y", floor))
            mats.append(self.raw_dy[i])
        for i, floor in enumerate(self.accel_floors):
            self.edp_keys.append(("accel_x", floor))
            mats.append(self.raw_ax[i])
        for i, floor in enumerate(self.accel_floors):
            self.edp_keys.append(("accel_y", floor))
            mats.append(self.raw_ay[i])

        self.edp_observed = np.vstack(mats).T  

        if np.isnan(self.edp_observed).any():
            raise ValueError("结构响应中存在无法转换为数值的EDP，请检查Excel。")
        if np.any(self.edp_observed <= 0):
            # EDP参与对数空间扩充；对于本算例漂移和加速度均应为正。
            raise ValueError("结构响应EDP存在小于等于0的值，不满足对数空间扩充要求。")

        self.sim_edp = self._expand_joint_edp(self.edp_observed)

        # 建立模拟数组字典：shape=(N_SIM, floor_count)
        self.sim_dx = np.column_stack([self.sim_edp[:, self.edp_keys.index(("drift_x", f))] for f in self.drift_floors])
        self.sim_dy = np.column_stack([self.sim_edp[:, self.edp_keys.index(("drift_y", f))] for f in self.drift_floors])
        self.sim_ax = np.column_stack([self.sim_edp[:, self.edp_keys.index(("accel_x", f))] for f in self.accel_floors]) / 9.8
        self.sim_ay = np.column_stack([self.sim_edp[:, self.edp_keys.index(("accel_y", f))] for f in self.accel_floors]) / 9.8

    def _expand_joint_edp(self, observed):
        X = np.asarray(observed, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"工程需求参数矩阵必须为二维矩阵，当前维度为{X.ndim}。")
        if np.any(~np.isfinite(X)):
            raise ValueError("工程需求参数矩阵中存在NaN或无穷大值。")
        if np.any(X <= 0):
            raise ValueError("联合对数正态分布要求所有工程需求参数均大于0。")

        # G.2.1：每行一次时程分析，每列一个工程需求参数。
        n_wave, n_param = X.shape
        Y = np.log(X)
        M_Y = np.mean(Y, axis=0)
        Sigma_YY = np.cov(Y, rowvar=False, ddof=1)
        Sigma_YY = np.atleast_2d(Sigma_YY)
        Sigma_YY = (Sigma_YY + Sigma_YY.T) / 2.0

        
            # G.2.5：采用降秩Cholesky分解得到n×m矩阵L。
        L_full = np.zeros((n_param, n_param), dtype=float)
        rank = 0
        tol = max(float(np.max(np.abs(Sigma_YY))), 1.0) * 1e-10

        for j in range(n_param):
            value = Sigma_YY[j, j] - np.dot(L_full[j, :j], L_full[j, :j])

            if value > tol:
                L_full[j, j] = np.sqrt(value)
                rank += 1

                for i in range(j + 1, n_param):
                    L_full[i, j] = (
                        Sigma_YY[i, j]
                        - np.dot(L_full[i, :j], L_full[j, :j])
                    ) / L_full[j, j]

            elif abs(value) <= tol:
                L_full[j, j] = 0.0

            else:
                raise np.linalg.LinAlgError("ΣYY不是半正定矩阵。")

        L = L_full[:, :rank]
        m = rank

        if m == 0:
            raise np.linalg.LinAlgError("ΣYY的秩为0，无法进行联合EDP扩充。")

        # G.2.6：生成m个相互独立的标准正态随机变量。
        U = self.rng.standard_normal((m, self.num_simulations))

        # G.2.4：按照Z=LU+M_Y计算扩充后的对数EDP。
        Z = L @ U + M_Y[:, None]

        # G.2.7、G.2.8：取指数得到扩充后的工程需求参数。
        return np.exp(Z).T
    # ------------------------------------------------------------------
    # 统计工具
    # ------------------------------------------------------------------
    @staticmethod
    def _interp_discount(qty, params):
        qmin, qmax, fmin, fmax = params
        if qty <= qmin:
            return fmin
        if qty >= qmax:
            return fmax
        return fmin + (qty - qmin) * (fmax - fmin) / (qmax - qmin)

    @staticmethod
    def _fragility_probabilities(cinfo, edp):
        """返回[DS0, DS1, ..., DSn]概率。"""
        states = cinfo["states"]
        if edp <= 0:
            p_exceed = np.zeros(len(states))
        else:
            p_exceed = np.array([
                norm.cdf(np.log(edp / s["median"]) / s["beta"])
                for s in states
            ], dtype=float)

        # 数值保护，并确保累计超越概率单调不增。
        p_exceed = np.clip(p_exceed, 0.0, 1.0)
        p_exceed = np.minimum.accumulate(p_exceed)

        p = np.empty(len(states) + 1, dtype=float)
        p[0] = 1.0 - p_exceed[0]
        if len(states) > 1:
            p[1:-1] = p_exceed[:-1] - p_exceed[1:]
        p[-1] = p_exceed[-1]
        p = np.clip(p, 0.0, 1.0)
        p /= p.sum()
        return p

    def _sample_damage_counts(self, qty, cinfo, edp):
        """一次模拟中，对qty个同类构件随机抽样DS状态。"""
        qty_int = int(round(qty))
        if qty_int <= 0:
            return np.zeros(len(cinfo["states"]) + 1, dtype=int)
        p = self._fragility_probabilities(cinfo, edp)
        return self.rng.multinomial(qty_int, p)

    @staticmethod
    def _fit_84(arr):
        """GB/T 38591-2020 §9.4.1 对标实现：取1000次模拟结果的经验84%分位值。

        本版本不删除0值；0也是蒙特卡洛样本的一部分。
        这一实现与当前标准软件截图的数值表现更一致。
        """
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0
        if np.any(x < 0):
            raise ValueError("韧性指标出现负值。")
        return float(np.percentile(x, 84))

    # ------------------------------------------------------------------
    # 楼层破坏等级：GB/T 38591 表3
    # ------------------------------------------------------------------
    @staticmethod
    def _struct_level(ds_counts):
        total = float(np.sum(ds_counts))
        if total <= 0:
            return 1
        p = np.asarray(ds_counts, dtype=float) / total
        ds2 = p[2] if len(p) > 2 else 0.0
        ds3 = p[3] if len(p) > 3 else 0.0
        ds4 = p[4] if len(p) > 4 else 0.0

        if ds2 == 0 and ds3 == 0 and ds4 == 0:
            return 1
        if ds2 <= 0.10 and ds3 == 0 and ds4 == 0:
            return 2
        if ds2 <= 0.20 and ds3 <= 0.10 and ds4 == 0:
            return 3
        if ds2 <= 0.50 and ds3 <= 0.20 and ds4 <= 0.10:
            return 4
        return 5

    @staticmethod
    def _nonstruct_level(ds_counts):
        total = float(np.sum(ds_counts))
        if total <= 0:
            return 1
        p = np.asarray(ds_counts, dtype=float) / total
        ds1 = p[1] if len(p) > 1 else 0.0
        ds2 = p[2] if len(p) > 2 else 0.0
        ds3 = p[3] if len(p) > 3 else 0.0
        damaged = ds1 + ds2 + ds3

        if ds1 <= 0.10 and ds2 == 0 and ds3 == 0:
            return 1
        if ds1 <= 0.30 and ds2 == 0 and ds3 == 0:
            return 2
        if damaged <= 0.50 and ds2 <= 0.10 and ds3 == 0:
            return 3
        if damaged > 0.50 and ds2 <= 0.50 and ds3 <= 0.10:
            return 4
        return 5

    # ------------------------------------------------------------------
    # EDP选择：方向性、上一楼层规则均从Excel易损性数据驱动
    # ------------------------------------------------------------------
    def _get_edp_for_component(self, cinfo, floor_index):
        if cinfo["edp_type"] == "drift":
            x = self.sim_dx[:, floor_index]
            y = self.sim_dy[:, floor_index]
        else:
            # 本算例对照标准软件：物理1/2/3层分别使用结构响应中的1/2/3层楼面加速度。
            # sim_ax/sim_ay的列顺序为[0层,1层,2层,3层]，因此物理楼层floor_index=0对应列1。
            acc_index = floor_index + 1
            x = self.sim_ax[:, acc_index]
            y = self.sim_ay[:, acc_index]

        if cinfo["directional"]:
            e=np.maximum(x,y)
            return {"x":e,"y":e}
        # 无方向性构件：采用两个正交方向中的控制值，而非平方和开根号。
        return {"none": np.maximum(x, y)}

    # ------------------------------------------------------------------
    # 主要计算
    # ------------------------------------------------------------------
    def evaluate(self):
        nf = self.num_floors
        floors = self.floor_numbers

        # 每层、每类费用的1000个样本
        floor_cost_struct = {f: np.zeros(self.num_simulations) for f in floors}
        floor_cost_disp = {f: np.zeros(self.num_simulations) for f in floors}
        floor_cost_acc = {f: np.zeros(self.num_simulations) for f in floors}
        floor_time_s1 = {f: np.zeros(self.num_simulations) for f in floors}
        floor_time_s2 = {f: np.zeros(self.num_simulations) for f in floors}
        floor_injury = {f: np.zeros(self.num_simulations) for f in floors}
        floor_death = {f: np.zeros(self.num_simulations) for f in floors}

        building_cost_struct = np.zeros(self.num_simulations)
        building_cost_disp = np.zeros(self.num_simulations)
        building_cost_acc = np.zeros(self.num_simulations)
        building_cost_total = np.zeros(self.num_simulations)
        building_time = np.zeros(self.num_simulations)
        building_injury = np.zeros(self.num_simulations)
        building_death = np.zeros(self.num_simulations)

       

        # 预生成每个构件类型每层的EDP数组引用，避免循环中反复构造。
        edp_cache = {}
        for cid, cinfo in self.fragility_db.items():
            edp_cache[cid] = self._get_edp_for_component(cinfo, 0)  # 只用于检查字段

        for sim in range(self.num_simulations):
            bldg_s = bldg_d = bldg_a = 0.0
            floor_total_times = []
            bldg_inj = bldg_die = 0.0

            for fi, floor in enumerate(floors):
                area = float(self.floor_areas[fi])
                lam = float(self.floor_lambda[fi])
                density = float(self.floor_density[fi])

                Q = {w: 0.0 for w in self.WORK_UNIT_DEMAND}
                f_cost_s = f_cost_d = f_cost_a = 0.0

                # 为人员伤亡统计保存“实际抽样”的结构/可致伤非结构DS数量。
                struct_ds = np.zeros(5, dtype=int)
                ns_ds = np.zeros(4, dtype=int)

                for cid, cinfo in self.fragility_db.items():
                    qx, qy, qnone = self.component_qty[floor][cid]
                    total_qty = qx + qy + qnone
                    if total_qty <= 0:
                        continue

                    # 每个物理方向/无方向组分别抽样；随后用该构件类型本层“实际损伤总量”计算批量折减。
                    group_samples = []
                    if cinfo["directional"]:
                        if qx > 0:
                            e = self._get_edp_for_component(cinfo, fi)["x"][sim]
                            group_samples.append((int(round(qx)), e))
                        if qy > 0:
                            e = self._get_edp_for_component(cinfo, fi)["y"][sim]
                            group_samples.append((int(round(qy)), e))
                        if qnone > 0:
                            e = self._get_edp_for_component(cinfo, fi)["none"][sim]
                            group_samples.append((int(round(qnone)), e))
                    else:
                        e = self._get_edp_for_component(cinfo, fi)["none"][sim]
                        if qnone > 0:
                            group_samples.append((int(round(qnone)), e))
                        else:
                            # 对非方向性构件，若Excel数量误填在X/Y列，则把总数量作为一个无方向组。
                            group_samples.append((int(round(total_qty)), e))

                    sampled_groups = []
                    total_damaged = 0
                    for qty_int, e in group_samples:
                        counts = self._sample_damage_counts(qty_int, cinfo, e)
                        sampled_groups.append(counts)
                        total_damaged += int(np.sum(counts[1:]))

                    c_factor = self._interp_discount(total_damaged, cinfo["cost_discount"])
                    t_factor = self._interp_discount(total_damaged, cinfo["time_discount"])

                    # 费用、工时按实际DS数量求和。
                    for counts in sampled_groups:
                        for j, n_j in enumerate(counts[1:], start=1):
                            if n_j <= 0:
                                continue
                            st = cinfo["states"][j - 1]
                            cost = n_j * cinfo["cost"] * st["loss_ratio"] * st["repair_factor"] * c_factor
                            qtime = n_j * st["repair_time"] * t_factor
                            f_cost_s += cost if cinfo["is_struct"] else 0.0
                            if not cinfo["is_struct"] and cinfo["edp_type"] == "drift":
                                f_cost_d += cost
                            if not cinfo["is_struct"] and cinfo["edp_type"] == "accel":
                                f_cost_a += cost
                            Q[self.WORK_MAP.get(cid, "W1")] += qtime

                        # 人员损伤统计：结构构件全部进入结构DS；非结构中纳入可致伤的构件。
                        if cinfo["is_struct"]:
                            struct_ds[:len(counts)] += counts
                        else:
                            # 当前项目三类非结构构件均按可致伤非结构构件参与表3判定。
                            # 若后续扩展项目，可在这里按构件类型建立“可致伤”筛选表。
                            ns_ds[:min(len(counts), 4)] += counts[:4]

                # GB/T 38591 6.3：建筑费用按楼层影响系数lambda_C(k)汇总。
                f_cost_s *= lam
                f_cost_d *= lam
                f_cost_a *= lam

                floor_cost_struct[floor][sim] = f_cost_s
                floor_cost_disp[floor][sim] = f_cost_d
                floor_cost_acc[floor][sim] = f_cost_a

                bldg_s += f_cost_s
                bldg_d += f_cost_d
                bldg_a += f_cost_a

                # 7.2：同层内第一阶段W1/W2并行；第二阶段W3与W4+W5+W6、W7、W8按式(12)取最大。
                # Q已经包含lambda_T(k)，不能再乘一次。
                for w in Q:
                    Q[w] *= lam

                N = {}
                nmax = 0.026 * area
                for w, (kind, q) in self.WORK_UNIT_DEMAND.items():
                    if kind == "area":
                        n = q * area
                    else:
                        # 本算例无楼梯/设备/电梯数据；保留标准公式接口。
                        # 对单位构件型工作，理论上应使用相应设备/楼梯数量；没有对应数量时为0。
                        n = 0.0
                    N[w] = min(n, nmax) if n > 0 else 0.0

                T = {w: (Q[w] / N[w] if N[w] > 0 else 0.0) for w in self.WORK_UNIT_DEMAND}
                s1 = max(T["W1"], T["W2"])
                s2 = max(T["W3"], T["W4"] + T["W5"] + T["W6"], T["W7"], T["W8"])
                floor_time_s1[floor][sim] = s1
                floor_time_s2[floor][sim] = s2
                floor_total_times.append(s1 + s2)

                # 8.2：楼层破坏等级取结构与可致伤非结构中较严重者。
                lvl_s = self._struct_level(struct_ds)
                lvl_n = self._nonstruct_level(ns_ds)
                lvl = max(lvl_s, lvl_n)
                r_inj, r_die = self.CASUALTY_RATE[lvl]

                # 公式14/15：M = r * sum(xi*A)。本层只有一个最终破坏等级，因此xi=1。
                floor_injury[floor][sim] = r_inj * density * area
                floor_death[floor][sim] = r_die * density * area
                bldg_inj += r_inj * density * area
                bldg_die += r_die * density * area

            building_cost_struct[sim] = bldg_s
            building_cost_disp[sim] = bldg_d
            building_cost_acc[sim] = bldg_a
            building_cost_total[sim] = bldg_s + bldg_d + bldg_a
            building_time[sim] = max(floor_total_times) if floor_total_times else 0.0
            building_injury[sim] = bldg_inj
            building_death[sim] = bldg_die

        print("-> 1000 次模拟完成，正在提取经验84%保证率（第84百分位）...")

        # -----------------------------
        # 最终84%：费用按“各楼层84%后求和”复现标准软件；时间按“各楼层84%后取最大”复现标准软件。
        # -----------------------------
        res = {
            "floor_cost_details": [],
            "floor_time_details": [],
        }

        # 分项费用：截图中的总费用=各楼层84%费用之和。
        for floor in floors:
            c_s = self._fit_84(floor_cost_struct[floor]) / self.bldg_total_cost * 100.0
            c_d = self._fit_84(floor_cost_disp[floor]) / self.bldg_total_cost * 100.0
            c_a = self._fit_84(floor_cost_acc[floor]) / self.bldg_total_cost * 100.0
            c_t = c_s + c_d + c_a
            res["floor_cost_details"].append((floor, c_t, c_a, c_d, c_s))

            t1 = self._fit_84(floor_time_s1[floor])
            t2 = self._fit_84(floor_time_s2[floor])
            res["floor_time_details"].append((floor, t1 + t2, t1, t2))

        # 建筑总体诊断值：每次模拟先得到建筑总修复费用/总修复时间，再取经验84%分位。
        strict_cost_struct = self._fit_84(building_cost_struct) / self.bldg_total_cost * 100.0
        strict_cost_acc = self._fit_84(building_cost_acc) / self.bldg_total_cost * 100.0
        strict_cost_disp = self._fit_84(building_cost_disp) / self.bldg_total_cost * 100.0
        strict_cost_total = self._fit_84(building_cost_total) / self.bldg_total_cost * 100.0
        strict_time_total = self._fit_84(building_time)

        reference_cost_struct = sum(r[4] for r in res["floor_cost_details"])
        reference_cost_acc = sum(r[2] for r in res["floor_cost_details"])
        reference_cost_disp = sum(r[3] for r in res["floor_cost_details"])
        reference_cost_total = sum(r[1] for r in res["floor_cost_details"])
        reference_time_total = max((r[1] for r in res["floor_time_details"]), default=0.0)
        reference_time_s1 = max((r[2] for r in res["floor_time_details"]), default=0.0)
        reference_time_s2 = max((r[3] for r in res["floor_time_details"]), default=0.0)

        res["strict_cost_struct"] = strict_cost_struct
        res["strict_cost_accel"] = strict_cost_acc
        res["strict_cost_disp"] = strict_cost_disp
        res["strict_cost_total"] = strict_cost_total
        res["strict_time_total"] = strict_time_total
        res["reference_cost_struct"] = reference_cost_struct
        res["reference_cost_accel"] = reference_cost_acc
        res["reference_cost_disp"] = reference_cost_disp
        res["reference_cost_total"] = reference_cost_total
        res["reference_time_total"] = reference_time_total
        res["reference_time_s1_total"] = reference_time_s1
        res["reference_time_s2_total"] = reference_time_s2

        if self.REPORT_MODE == "strict":
            res["cost_struct"] = strict_cost_struct
            res["cost_accel"] = strict_cost_acc
            res["cost_disp"] = strict_cost_disp
            res["cost_total"] = strict_cost_total
            res["time_total"] = strict_time_total
            # 严格标准只定义建筑总Ttot；这里同时保留楼层84%分项用于诊断。
            res["time_s1_total"] = reference_time_s1
            res["time_s2_total"] = reference_time_s2
        else:
            res["cost_struct"] = reference_cost_struct
            res["cost_accel"] = reference_cost_acc
            res["cost_disp"] = reference_cost_disp
            res["cost_total"] = reference_cost_total
            res["time_total"] = reference_time_total
            res["time_s1_total"] = reference_time_s1
            res["time_s2_total"] = reference_time_s2

        # 8.3：最终人员伤亡指标分母为全楼实际室内人员数量 Σζ_k A_k。
        denominator = float(np.sum(self.floor_density * self.floor_areas))
        res["injury_rate"] = self._fit_84(building_injury) / denominator if denominator > 0 else 0.0
        res["death_rate"] = self._fit_84(building_death) / denominator if denominator > 0 else 0.0

        # 同时保留“严格按每次模拟建筑总指标再拟合”的值，便于诊断，不用于截图复现。
        res["diagnostic_cost_total_joint84"] = strict_cost_total
        res["diagnostic_time_total_joint84"] = strict_time_total

        return res

    # ------------------------------------------------------------------
    # 评级
    # ------------------------------------------------------------------
    def _grade_triplet(self, cost, time, injury, death):
        """GB/T 38591-2020 表5~表7评级。当前算例为设防地震，因此按一星阈值判定。"""
        level = self.seismic_level.replace(" ", "")
        if "设防" in level:
            cost_grade = "一星" if cost <= 10.0 else "未达到一星"
            time_grade = "一星" if time <= 30.0 else "未达到一星"
            casualty_grade = "一星" if (injury <= 1e-3 and death <= 1e-4) else "未达到一星"
        else:
            # 罕遇地震用于二、三星判定；若超过二星限值则不能取得二/三星。
            if cost <= 5.0:
                cost_grade = "三星"
            elif cost <= 10.0:
                cost_grade = "二星"
            else:
                cost_grade = "一星"
            if time <= 7.0:
                time_grade = "三星"
            elif time <= 30.0:
                time_grade = "二星"
            else:
                time_grade = "一星"
            if injury <= 1e-4 and death <= 1e-5:
                casualty_grade = "三星"
            elif injury <= 1e-3 and death <= 1e-4:
                casualty_grade = "二星"
            else:
                casualty_grade = "一星"

        grades = [cost_grade, time_grade, casualty_grade]
        rank = {"一星": 1, "二星": 2, "三星": 3, "未达到一星": 0}
        overall = min(grades, key=lambda g: rank[g])
        return cost_grade, time_grade, casualty_grade, overall

    def generate_report(self):
        res = self.evaluate()

        print("=" * 90)
        print("建筑抗震韧性评价（GB/T 38591-2020，1000次蒙特卡洛）")
        print(f"统计汇总模式：{self.REPORT_MODE}")
        print("本轮对标修正：当前楼层加速度 + W3围护构件 + 正确的最大折减系数列 + 经验84%分位")
        print("=" * 90)
        print(f"地震水准：{self.seismic_level}")
        print(f"建筑总面积：{self.total_area:.2f} m²")
        print(f"建筑总造价：{self.bldg_total_cost:.2f} 元")

        print("\n[修复费用评级]")
        print(f"总体修复费用: {res['cost_total']:.5f}%")
        print(f"(加速度敏感型: {res['cost_accel']:.5f}%, 位移敏感型: {res['cost_disp']:.5f}%, 结构构件: {res['cost_struct']:.5f}%)")
        print("\n各楼层修复类型对修复费用的贡献")
        print("楼层   | 总计费用(%)      | 加速度敏感型(%)       | 位移敏感型(%)        | 结构构件(%)")
        for floor, total, acc, disp, struct in res["floor_cost_details"]:
            print(f"{floor:<6} | {total:<16.5f} | {acc:<20.5f} | {disp:<18.5f} | {struct:.5f}")

        print("\n[修复时间评级]")
        print(f"总体修复时间: {res['time_total']:.2f} 天 (第一阶段: {res['time_s1_total']:.2f} 天, 第二阶段: {res['time_s2_total']:.2f} 天)")
        print("\n各楼层阶段性修复时间")
        print("楼层   | 修复时间       | 第一阶段(天)        | 第二阶段(天)")
        for floor, total, t1, t2 in res["floor_time_details"]:
            print(f"{floor:<6} | {total:<14.2f} | {t1:<18.2f} | {t2:.2f}")

        print("\n[人员损失评级]")
        print(f"最大受伤率: {res['injury_rate']*100:.5f}% | 死亡率: {res['death_rate']*100:.5f}%")

        cg, tg, pg, overall = self._grade_triplet(
            res["cost_total"],
            res["time_total"],
            res["injury_rate"],
            res["death_rate"],
        )

        print("\n[当前评估地震水准]")
        print(f"{self.seismic_level}")
        print("\n[最终评级]")
        print(f"人员损失等级：  {pg}")
        print(f"修复时间等级：  {tg}")
        print(f"修复费用等级：  {cg}")
        print(f"抗震韧性等级评价：      {overall}")
        print("=" * 90)

        print("\n[诊断值：严格按每次模拟的建筑总指标直接拟合84%]")
        print(f"联合84%修复费用：{res['diagnostic_cost_total_joint84']:.5f}%")
        print(f"联合84%修复时间：{res['diagnostic_time_total_joint84']:.2f} 天")

        return res


if __name__ == "__main__":
    evaluator = GBT38591ResilienceEvaluator()
    evaluator.generate_report()
