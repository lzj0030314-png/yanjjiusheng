import os
import re
import math
import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm


class GBT38591ResilienceEvaluator:

    DEFAULT_FILE = "中震三层数据导出版改系数.xls"
    N_SIM = 1000
    RANDOM_SEED = 42

    WORK_MAP = {
        "A.98.Z.Z.Z.813": "W1",  # 柱
        "A.98.Z.Z.Z.814": "W1",  # 梁
        "B.98.Z.Z.Z.161": "W3",  # 填充墙
        "B.01.A.A.A.001": "W3",  # 玻璃幕墙
        "B.01.B.B.A.001": "W5",  # 吊顶
    }

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

        self.floor_areas = self.floor_info["楼层面积（m^2）"].astype(float).to_numpy()
        self.floor_lambda = self.floor_info["楼层影响系数"].astype(float).to_numpy()
        self.floor_density = self.floor_info["楼层人口密度"].astype(float).to_numpy()

        print("\n================ Excel数据提取检查 ================")
        print("\n【建筑信息1】")
        print("建筑总面积 =", self.total_area)
        print("单位造价 =", self.unit_cost)
        print("建筑总造价 =", self.bldg_total_cost)
        print("使用功能 =", self.usage)
        print("\n【建筑信息2】")
        print("楼层编号 =", self.floor_numbers)
        print("楼层数量 =", self.num_floors)
        print("楼层面积 =", self.floor_areas)
        print("楼层影响系数 =", self.floor_lambda)
        print("楼层人口密度 =", self.floor_density)
        print("\n【地震信息1】")
        print("地震水准 =", self.seismic_level)
        print("地震波数 =", self.wave_count)
        print("\n【结构构件信息】")
        print("shape =", self.struct_info.shape)
        print(self.struct_info.to_string(index=False))
        print("\n【非结构构件信息】")
        print("shape =", self.nonstruct_info.shape)
        print(self.nonstruct_info.to_string(index=False))
        print("\n================ 易损性数据提取检查 ================")
        print("\n【结构构件易损性】")
        print(f"数据行数：{len(self.struct_fragility_raw)}")
        print(f"数据列数：{len(self.struct_fragility_raw.columns)}")
        struct_check_cols = [
            "易损性编号", "构件造价", "构件单位",
            "构件类型", "EDP类型", "是否具有方向性",
            "使用上一楼层的EDP"
        ]
        print("\n① 构件基本信息：")
        struct_check_df = self.struct_fragility_raw[struct_check_cols].copy()
        struct_check_df["易损性编号"] = struct_check_df["易损性编号"].ffill()
        print(struct_check_df.drop_duplicates("易损性编号").to_string(index=False))
        struct_state_cols = [
            "易损性编号", "损伤状态名称", "中位值",
            "对数标准差", "损失系数", "修复系数",
            "修复时间（人/天）"
        ]
        print("\n② 损伤状态及修复参数：")
        struct_state_df = self.struct_fragility_raw[struct_state_cols].copy()
        struct_state_df["易损性编号"] = struct_state_df["易损性编号"].ffill()
        print(struct_state_df.to_string(index=False))
        struct_discount_cols = [
            "易损性编号",
            "最小工程量折减系数（修复费用）",
            "最大工程量折减系数（修复费用）",
            "最小工程量折减系数（修复时间）",
            "最大工程量折减系数（修复时间）"
        ]
        print("\n③ 工程量折减参数：")
        struct_discount_df = self.struct_fragility_raw[struct_discount_cols].copy()
        struct_discount_df["易损性编号"] = struct_discount_df["易损性编号"].ffill()
        print(struct_discount_df.drop_duplicates("易损性编号").to_string(index=False))
        print("\n----------------------------------------------------")
        print("\n【非结构构件易损性】")
        print(f"数据行数：{len(self.nonstruct_fragility_raw)}")
        print(f"数据列数：{len(self.nonstruct_fragility_raw.columns)}")
        nonstruct_check_cols = [
            "易损性编号", "构件造价", "构件单位",
            "构件类型", "EDP类型", "是否具有方向性",
            "使用上一楼层的EDP"
        ]
        print("\n① 构件基本信息：")
        nonstruct_check_df = self.nonstruct_fragility_raw[nonstruct_check_cols].copy()
        nonstruct_check_df["易损性编号"] = nonstruct_check_df["易损性编号"].ffill()
        print(nonstruct_check_df.drop_duplicates("易损性编号").to_string(index=False))
        nonstruct_state_cols = [
            "易损性编号", "损伤状态名称", "中位值",
            "对数标准差", "损失系数", "修复系数",
            "修复时间（人/天）"
        ]
        print("\n② 损伤状态及修复参数：")
        nonstruct_state_df = self.nonstruct_fragility_raw[nonstruct_state_cols].copy()
        nonstruct_state_df["易损性编号"] = nonstruct_state_df["易损性编号"].ffill()
        print(nonstruct_state_df.to_string(index=False))
        nonstruct_discount_cols = [
            "易损性编号",
            "最小工程量折减系数（修复费用）",
            "最大工程量折减系数（修复费用）",
            "最小工程量折减系数（修复时间）",
            "最大工程量折减系数（修复时间）"
        ]
        print("\n③ 工程量折减参数：")
        nonstruct_discount_df = self.nonstruct_fragility_raw[nonstruct_discount_cols].copy()
        nonstruct_discount_df["易损性编号"] = nonstruct_discount_df["易损性编号"].ffill()
        print(nonstruct_discount_df.drop_duplicates("易损性编号").to_string(index=False))
        print("\n====================================================")
    # ------------------------------------------------------------------
    # 易损性数据库
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

                if pd.isna(median) or pd.isna(beta) or pd.isna(loss) or pd.isna(repair_factor) or pd.isna(repair_time):
                    continue

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

            cfmin = pd.to_numeric(first.get("最小工程量折减系数（修复费用）"), errors="coerce")
            cfmax = pd.to_numeric(first.get("最大工程量折减系数（修复费用）"), errors="coerce")
            tfmin = pd.to_numeric(first.get("最小工程量折减系数（修复时间）"), errors="coerce")
            tfmax = pd.to_numeric(first.get("最大工程量折减系数（修复时间）"), errors="coerce")

            if is_struct:
                default_min_qty = 10
                default_max_qty = 50
            else:
                default_min_qty = 1
                default_max_qty = 10

            db[cid] = {
                "id": cid,
                "name": str(first.get("构件名称", "")),
                "cost": float(cost),
                "edp_type": edp_type,
                "directional": directional,
                "previous_floor_edp": previous_floor,
                "is_struct": bool(is_struct),
                "component_type": "struct" if is_struct else "nonstruct",
                "states": states,

                "cost_discount": (default_min_qty, default_max_qty, float(cfmin), float(cfmax)),
                "time_discount": (default_min_qty, default_max_qty, float(tfmin), float(tfmax)),
            }
        return db
    # ------------------------------------------------------------------
    # 系数插值计算
    # ------------------------------------------------------------------
    @staticmethod
    def _interp_discount(qty, params):
        qmin, qmax, fmin, fmax = params
        if qty <= qmin:
            return fmin
        if qty >= qmax:
            return fmax
        return fmin + (qty - qmin) * (fmax - fmin) / (qmax - qmin)
    # ------------------------------------------------------------------
    # 建立易损库数据
    # ------------------------------------------------------------------
    def _build_fragility_database(self):
        sdb = self._build_one_fragility(self.struct_fragility_raw, True)
        ndb = self._build_one_fragility(self.nonstruct_fragility_raw, False)
        self.fragility_db = {**sdb, **ndb}

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
    # ------------------------------------------------------------------
    # 构件方向
    # ------------------------------------------------------------------
    def get_component_quantity(self, cid, floor):
        g = self._component_rows(floor, cid)
        if g.empty:
            return 0.0, 0.0, 0.0
        
        qx = pd.to_numeric(g["X方向易损性数据"], errors="coerce").fillna(0).sum()
        qy = pd.to_numeric(g["Y方向易损性数据"], errors="coerce").fillna(0).sum()
        qn = pd.to_numeric(g["无方向易损性数据"], errors="coerce").fillna(0).sum()
        return float(qx), float(qy), float(qn)
    # ------------------------------------------------------------------
    # 各类构件全部数量
    # ------------------------------------------------------------------
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

        def block(rows, start, end):
            x = df.iloc[rows, start:end].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            return x

        self.drift_floors = [1, 2, 3]
        self.accel_floors = [0, 1, 2, 3]

        drift_rows = [2, 3, 4] 
        accel_rows = [1, 2, 3, 4] 

        self.raw_dx = block(drift_rows, 1, 12)
        self.raw_ax = block(accel_rows, 12, 23)
        self.raw_dy = block(drift_rows, 23, 34)
        self.raw_ay = block(accel_rows, 34, 45)

        print("\n================ EDP楼层对应检查 ================")
        print("\nX方向层间位移角：")
        for i, floor in enumerate(self.drift_floors):
            print(f"楼层 {floor} → raw_dx[{i}]")
            print(self.raw_dx[i])
        print("\nY方向层间位移角：")
        for i, floor in enumerate(self.drift_floors):
            print(f"楼层 {floor} → raw_dy[{i}]")
            print(self.raw_dy[i])
        print("\nX方向加速度：")
        for i, floor in enumerate(self.accel_floors):
            print(f"楼层 {floor} → raw_ax[{i}]")
            print(self.raw_ax[i])
        print("\nY方向加速度：")
        for i, floor in enumerate(self.accel_floors):
            print(f"楼层 {floor} → raw_ay[{i}]")
            print(self.raw_ay[i])
        print("==================================================\n")



        if self.raw_dx.shape[1] != 11:
            raise ValueError("结构响应X方向层间位移角应包含11条地震波。")

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



        print("\n================ EDP联合矩阵检查 ================")
        print("edp_keys =", self.edp_keys)
        print("edp_observed.shape =", self.edp_observed.shape)
        print("每一行 = 1条地震波，每一列 = 1个EDP参数")
        print(self.edp_observed)

        if np.isnan(self.edp_observed).any():
            raise ValueError("结构响应中存在无法转换为数值的EDP，请检查Excel。")
        if np.any(self.edp_observed <= 0):
            raise ValueError("结构响应EDP存在小于等于0的值，不满足对数空间扩充要求。")

        self.sim_edp = self._expand_joint_edp(self.edp_observed)

        print("\n================ EDP扩充结果检查 ================")
        print("sim_edp.shape =", self.sim_edp.shape)
        print("前5次模拟结果：")
        print(self.sim_edp[:5])

        self.sim_dx = np.column_stack([self.sim_edp[:, self.edp_keys.index(("drift_x", f))] for f in self.drift_floors])
        self.sim_dy = np.column_stack([self.sim_edp[:, self.edp_keys.index(("drift_y", f))] for f in self.drift_floors])
        self.sim_ax = np.column_stack([self.sim_edp[:, self.edp_keys.index(("accel_x", f))] for f in self.accel_floors]) / 9.8
        self.sim_ay = np.column_stack([self.sim_edp[:, self.edp_keys.index(("accel_y", f))] for f in self.accel_floors]) / 9.8

        print("\n================ EDP拆分结果检查 ================")
        print("sim_dx.shape =", self.sim_dx.shape)
        print("sim_dy.shape =", self.sim_dy.shape)
        print("sim_ax.shape =", self.sim_ax.shape)
        print("sim_ay.shape =", self.sim_ay.shape)
        print("==================================================")

    def _expand_joint_edp(self, observed):
        X = np.asarray(observed, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"工程需求参数矩阵必须为二维矩阵，当前维度为{X.ndim}。")
        if np.any(~np.isfinite(X)) or np.any(X <= 0):
            raise ValueError("工程需求参数矩阵存在无效值；联合对数正态分布要求所有EDP均大于0。")

        n_wave, n_param = X.shape

        Y = np.log(X)
        M_Y = np.mean(Y, axis=0)
        Y_centered = Y - M_Y
        Sigma_YY = np.cov(Y, rowvar=False, ddof=1)
        Sigma_YY = np.atleast_2d((Sigma_YY + Sigma_YY.T) / 2.0)

        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_YY)
        tol = max(float(np.max(np.abs(eigenvalues))), 1.0) * 1e-10
        keep = eigenvalues > tol
        rank = int(np.sum(keep))
        if rank == 0:
            raise np.linalg.LinAlgError("ΣYY的秩为0，无法进行联合EDP扩充。")

        L = eigenvectors[:, keep] @ np.diag(np.sqrt(eigenvalues[keep]))
        m = rank

        N = self.num_simulations
        if N <= 1 or N < m + 1:
            raise ValueError(f"扩充次数N={N}不足以构造{m}个独立标准正态变量。")
        U_raw = self.rng.standard_normal((N, m))
        U_raw -= np.mean(U_raw, axis=0, keepdims=True)
        Q, _ = np.linalg.qr(U_raw, mode="reduced")
        U = np.sqrt(N - 1.0) * Q

        Z = L @ U.T + M_Y[:, None]

        Z_mean = np.mean(Z, axis=1)
        Z_cov = np.atleast_2d(np.cov(Z, rowvar=True, ddof=1))
        mean_error = np.max(np.abs(Z_mean - M_Y))
        cov_error = np.max(np.abs(Z_cov - Sigma_YY))
        if mean_error > 1e-10 or cov_error > 1e-10:
            raise RuntimeError(
                f"G.2.2联合统计量检查未通过："
                f"对数均值最大误差={mean_error:.3e}，"
                f"协方差最大误差={cov_error:.3e}。"
            )

        sim = np.exp(Z).T

        print("\n================ G.2.2 均值方差统计检查 ================")
        print(f"原始矩阵：{n_wave}×{n_param}")
        print(f"扩充矩阵：{N}×{n_param}")
        print(f"ΣYY秩：{rank}")
        print(f"独立标准正态变量数量m：{m}")
        print(f"mean(Z)最大绝对误差：{mean_error:.3e}")
        print(f"Cov(Z)最大绝对误差：{cov_error:.3e}")
        print("G.2.1～G.2.8联合对数正态扩充计算完成。")
        print("==================================================")

        return sim
    # ------------------------------------------------------------------
    # 计算构件损伤状态概率
    # ------------------------------------------------------------------
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
    
     # 损伤抽样
    def _sample_damage_counts(self, qty, cinfo, edp):
        """一次模拟中，对qty个同类构件按照G.3.2超越概率随机抽样DS状态。"""
        qty_int = int(round(qty))
        if qty_int <= 0:
            return np.zeros(len(cinfo["states"]) + 1, dtype=int)

        p = self._fragility_probabilities(cinfo, edp)

        p_exceed = np.cumsum(p[::-1])[::-1][1:]
        counts = np.zeros(len(p), dtype=int)
        for _ in range(qty_int):

            R = self.rng.random()
            ds = 0

            while ds < len(p_exceed) and R < p_exceed[ds]:
                ds += 1
            counts[ds] += 1
          
        return counts

    @staticmethod
    def _fit_84(arr):
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0
        if np.any(x < 0):
            raise ValueError("韧性指标出现负值，无法进行对数正态拟合。")
        x = x[x > 0]
        if x.size < 2:
            return 0.0
        shape, loc, scale = lognorm.fit(
            x,
            floc=0
        )
        value_84 = lognorm.ppf(
            0.84,
            shape,
            loc,
            scale
        )
        return float(value_84)
    # ------------------------------------------------------------------
    # 楼层破坏等级
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
        if ds2 > 0.50 or ds3 > 0.20 or ds4 > 0.10:
            return 5
        raise ValueError(f"结构构件破坏等级无法判定：DS2={ds2:.6f}, DS3={ds3:.6f}, DS4={ds4:.6f}")

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

        if ds2 > 0.50 or ds3 > 0.10:
            return 5

        return 3
    # ------------------------------------------------------------------
    # EDP选择
    # ------------------------------------------------------------------
    def _get_edp_for_component(self, cinfo, floor_index):
        if cinfo["edp_type"] == "drift":
            x = self.sim_dx[:, floor_index]
            y = self.sim_dy[:, floor_index]
        else:

            acc_index = self.accel_floors.index(floor_index + 1)
            x = self.sim_ax[:, acc_index]
            y = self.sim_ay[:, acc_index]

        if cinfo["directional"]:
            return {
                "x": x,
                "y": y
            }

        return {
            "none": np.maximum(x, y)
        }
    # ------------------------------------------------------------------
    # 主要计算
    # ------------------------------------------------------------------
    def evaluate(self):
        nf = self.num_floors
        floors = self.floor_numbers


        floor_cost_struct = {f: np.zeros(self.num_simulations) for f in floors}
        floor_cost_disp = {f: np.zeros(self.num_simulations) for f in floors}
        floor_cost_acc = {f: np.zeros(self.num_simulations) for f in floors}
        floor_time_s1 = {f: np.zeros(self.num_simulations) for f in floors}
        floor_time_s2 = {f: np.zeros(self.num_simulations) for f in floors}
        floor_time_total = {f: np.zeros(self.num_simulations) for f in floors}
        floor_injury = {f: np.zeros(self.num_simulations) for f in floors}
        floor_death = {f: np.zeros(self.num_simulations) for f in floors}

        building_cost_struct = np.zeros(self.num_simulations)
        building_cost_disp = np.zeros(self.num_simulations)
        building_cost_acc = np.zeros(self.num_simulations)
        building_cost_total = np.zeros(self.num_simulations)
        building_time = np.zeros(self.num_simulations)
        building_time_s1 = np.zeros(self.num_simulations)
        building_time_s2 = np.zeros(self.num_simulations)
        building_injury = np.zeros(self.num_simulations)
        building_death = np.zeros(self.num_simulations)


        edp_cache = {}
        for cid, cinfo in self.fragility_db.items():
            edp_cache[cid] = self._get_edp_for_component(cinfo, 0) 

        damage_level_statistics = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0
        }
        # 循环主程序
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

                struct_ds = np.zeros(5, dtype=int)
                ns_ds = np.zeros(4, dtype=int)

                for cid, cinfo in self.fragility_db.items():
                    qx, qy, qnone = self.component_qty[floor][cid]
                    total_qty = qx + qy + qnone
                    if total_qty <= 0:
                        continue

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

                            group_samples.append((int(round(total_qty)), e))

                    sampled_groups = []
                    total_damaged = 0
                    for qty_int, e in group_samples:
                        counts = self._sample_damage_counts(qty_int, cinfo, e)
                        sampled_groups.append(counts)
                        total_damaged += int(np.sum(counts[1:]))

                    c_factor = self._interp_discount(total_damaged, cinfo["cost_discount"])
                    t_factor = self._interp_discount(total_damaged, cinfo["time_discount"])


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


                        if cinfo["is_struct"]:
                            struct_ds[:len(counts)] += counts
                        else:

                            ns_ds[:min(len(counts), 4)] += counts[:4]

                f_cost_s *= lam
                f_cost_d *= lam
                f_cost_a *= lam

                floor_cost_struct[floor][sim] = f_cost_s
                floor_cost_disp[floor][sim] = f_cost_d
                floor_cost_acc[floor][sim] = f_cost_a

                bldg_s += f_cost_s
                bldg_d += f_cost_d
                bldg_a += f_cost_a


                for w in Q:
                    Q[w] *= lam

                N = {}
                nmax = 0.026 * area

                for w, (kind, q) in self.WORK_UNIT_DEMAND.items():
                    if kind == "area":
                        n = q * area
                    else:

                        n = 0.0

                    # 施工人数：
                    N[w] = min(n, nmax) if n > 0 else 0.0

                T = {w: (Q[w] / N[w] if N[w] > 0 else 0.0) for w in self.WORK_UNIT_DEMAND}
               
                s1 = max(T["W1"], T["W2"])
                s2 = max(T["W3"], T["W4"] + T["W5"] + T["W6"], T["W7"], T["W8"])
                floor_time_s1[floor][sim] = s1
                floor_time_s2[floor][sim] = s2
                floor_time_total[floor][sim] = s1 + s2
                floor_total_times.append(s1 + s2)


                # 楼层破坏等级取结构与可致伤非结构中较严重者。
                lvl_s = self._struct_level(struct_ds)
                lvl_n = self._nonstruct_level(ns_ds)
                lvl = max(lvl_s, lvl_n)

                damage_level_statistics[lvl] += 1

                r_inj, r_die = self.CASUALTY_RATE[lvl]

                floor_injury[floor][sim] = r_inj * density * area
                floor_death[floor][sim] = r_die * density * area
                bldg_inj += r_inj * density * area
                bldg_die += r_die * density * area

            building_cost_struct[sim] = bldg_s
            building_cost_disp[sim] = bldg_d
            building_cost_acc[sim] = bldg_a
            building_cost_total[sim] = bldg_s + bldg_d + bldg_a

            # 建筑整体修复时间
            building_time[sim] = max(floor_total_times) if floor_total_times else 0.0
            building_injury[sim] = bldg_inj
            building_death[sim] = bldg_die
        # --------------------------------------------------
        # 最终84%指标计算
        # --------------------------------------------------
        res = {}

        floor_cost_84 = [
            (
                self._fit_84(floor_cost_struct[f]),
                self._fit_84(floor_cost_disp[f]),
                self._fit_84(floor_cost_acc[f])
            )
            for f in floors
        ]

        cost_struct = sum(c[0] for c in floor_cost_84)
        cost_disp = sum(c[1] for c in floor_cost_84)
        cost_acc = sum(c[2] for c in floor_cost_84)

        res["cost_struct"] = cost_struct / self.bldg_total_cost * 100.0
        res["cost_accel"] = cost_acc / self.bldg_total_cost * 100.0
        res["cost_disp"] = cost_disp / self.bldg_total_cost * 100.0
        res["cost_total"] = (
            cost_struct + cost_disp + cost_acc
        ) / self.bldg_total_cost * 100.0

        # 修复时间指标
        res["time_total"] = self._fit_84(building_time)

        # 人员损失指标
        denominator = np.sum(self.floor_density * self.floor_areas)
        res["injury_rate"] = self._fit_84(building_injury) / denominator if denominator > 0 else 0.0
        res["death_rate"] = self._fit_84(building_death) / denominator if denominator > 0 else 0.0

        # 各楼层修复费用贡献
        floor_cost_details = []
        for floor in floors:
            struct = self._fit_84(floor_cost_struct[floor]) / self.bldg_total_cost * 100.0
            disp = self._fit_84(floor_cost_disp[floor]) / self.bldg_total_cost * 100.0
            acc = self._fit_84(floor_cost_acc[floor]) / self.bldg_total_cost * 100.0
            floor_cost_details.append((floor, struct + disp + acc, acc, disp, struct))

        res["floor_cost_details"] = floor_cost_details

        # 各楼层阶段修复时间
        floor_time_details = []
        for floor in floors:
            t1 = self._fit_84(floor_time_s1[floor])
            t2 = self._fit_84(floor_time_s2[floor])
            total = t1 + t2

            floor_time_details.append(
                (floor, total, t1, t2)
            )

        res["floor_time_details"] = floor_time_details

        # 建筑总体修复时间
        floor_time_84 = [
            (
                self._fit_84(floor_time_s1[f]),
                self._fit_84(floor_time_s2[f])
            )
            for f in floors
        ]

        res["time_s1_total"] = max(t[0] for t in floor_time_84)
        res["time_s2_total"] = max(t[1] for t in floor_time_84)
        res["time_total"] = max(t[0] + t[1] for t in floor_time_84)
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

        return res

if __name__ == "__main__":
    evaluator = GBT38591ResilienceEvaluator()
    evaluator.generate_report()
