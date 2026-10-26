"""M8 HemoSim 数字孪生生理引擎（教学级集总参数模型，纯 numpy）。

模型构成（教学级简化，非商用数字孪生）：
  - Frank-Starling 曲线：前负荷指数 P → 每搏量因子（表型决定曲线增益）
  - 血管张力：基础张力 + 去甲肾上腺素 PK/PD 耦合（中/高精度档启用 PK 平滑）
  - MAP = 基础灌注压 + 收缩力×每搏量因子×血管张力因子
  - 乳酸动力学：产生 vs 灌注清除（中/高精度档启用）
  - 尿量：MAP 阈值型 + 液体过负荷惩罚
  - 液体过负荷指数 FO：低收缩力/高累积液量时累积，压低 Starling 增益并推高 SOFA
  - 氢化可的松（T2-C）：延迟起效，降低升压药需求、提升乳酸清除

所有随机性来自显式传入的 numpy Generator，保证可复现。
"""
import json
import numpy as np
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

with open(CONFIG_DIR / "patient_templates.json", encoding="utf-8") as f:
    TEMPLATES = json.load(f)

PHENOTYPES = {p["id"]: p for p in TEMPLATES["phenotypes"]}
PRECISION = TEMPLATES["precision_levels"]
V = TEMPLATES["vitals"]

# 床旁容量反应性评估结果（PLR/PPV 的演示代理指标，由表型决定 + 测量噪声）
FLUID_RESPONSE_BASE = {
    "fluid_responder": 0.85, "pressor_responder": 0.28,
    "mixed": 0.55, "cardiac_suppressed": 0.20,
}


def _rr(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def new_patient(rng, case_id="SIM", precision="mid"):
    """生成一名随机虚拟脓毒性休克患者（隐藏表型）。"""
    pheno_id = str(rng.choice(list(PHENOTYPES.keys())))
    pheno = PHENOTYPES[pheno_id]
    comorbidity = int(rng.integers(0, 4))
    sofa = float(rng.integers(V["sofa0"][0], V["sofa0"][1] + 1)) + comorbidity * 0.3
    return {
        "case_id": case_id,
        "phenotype": pheno_id,
        "phenotype_label": pheno["label"],
        "age": int(rng.integers(35, 82)),
        "sex": str(rng.choice(["男", "女"])),
        "comorbidity": comorbidity,
        # ---- 生理状态（前负荷/基础张力按表型标定，保证各表型入院 MAP 均在休克区间）----
        "preload": _rr(rng, *pheno["preload_range"]),
        "svr_base": float(np.clip(pheno["svr_base"] + rng.normal(0, 0.03), 0.35, 0.85)),
        "contractility": pheno["contractility"],
        "starling_gain": pheno["starling_gain"],
        "svr_gain": pheno["svr_gain"],
        "ne_dose": _rr(rng, *V["ne0"]),        # μg/kg/min
        "ne_effect": _rr(rng, *V["ne0"]),      # PK 平滑后的效应
        "fluid_total": 0.0,                     # mL 累积晶体液
        "fluid_overload": 0.0,                  # 0-1 过负荷指数
        "steroid_on": False,
        "steroid_age": 0,
        # ---- 床旁指标 ----
        "map": _rr(rng, *V["map0"]),
        "hr": _rr(rng, *V["hr0"]),
        "lactate": _rr(rng, *V["lactate0"]),
        "urine": _rr(rng, *V["urine0"]),
        "sofa": round(sofa, 1),
        "spo2": _rr(rng, *V["spo2"]),
        "rr": _rr(rng, *V["rr"]),
        "cvp": _rr(rng, *V["cvp"]),
        "fluid_response": float(np.clip(FLUID_RESPONSE_BASE[pheno_id] + rng.normal(0, 0.08), 0.0, 1.0)),
        "step": 0,
        "precision": precision,
        "history": [],
    }


def starling_curve(preload, contractility):
    """Frank-Starling：每搏量因子（1.0 = 正常前负荷正常收缩力）。"""
    base = 1.0 - np.exp(-2.5 * 0.85)          # 正常前负荷参考点
    cur = 1.0 - np.exp(-2.5 * max(preload, 0.0))
    return float(min(2.2, (cur / base) * contractility))


def starling_curve_series(contractility):
    """返回 (preloads, sv_factors) 用于前端实时 Frank-Starling 图。"""
    preloads = np.linspace(0.1, 1.5, 29)
    factors = [starling_curve(p, contractility) for p in preloads]
    return preloads.tolist(), factors


def _perfusion(MAP):
    """灌注因子：MAP 65 附近敏感区。"""
    return float(1.0 / (1.0 + np.exp(-(MAP - 60.0) / 6.0)))


def apply_action(st, action_id, rng, fluid_ml=1000.0, ne_delta=0.12):
    """在决策点施加干预（A=分次晶体液共 1000mL；B=上调 NE；AB=联合减半）。"""
    if action_id in ("A", "AB"):
        frac = 1.0 if action_id == "A" else 0.45
        st["fluid_total"] += fluid_ml * frac
        st["preload"] = min(1.6, st["preload"] + 0.22 * frac)
    if action_id in ("B", "AB"):
        frac = 1.0 if action_id == "B" else 0.5
        st["ne_dose"] = min(1.2, st["ne_dose"] + ne_delta * frac)
        # 低前负荷时单纯升后负荷有害（心肌耗氧↑、灌注不改善）
        st["afterload_harm"] = bool(st["preload"] < 0.55)


def apply_t2(st, action_id):
    """T2 决策：C=加用氢化可的松，D=不加。"""
    if action_id == "C":
        st["steroid_on"] = True
        st["steroid_age"] = 0


def advance(st, rng, steps=1, override_precision=None):
    """推进生理模拟。返回轨迹记录列表。override_precision 可临时换挡演示。"""
    prec = PRECISION[override_precision or st["precision"]]
    records = []
    for _ in range(steps):
        for _sub in range(prec["substeps"]):
            noise = prec["noise"]
            # --- PK/PD：去甲肾上腺素效应 ---
            if prec["pk_coupling"]:
                st["ne_effect"] += (st["ne_dose"] - st["ne_effect"]) * 0.5
            else:
                st["ne_effect"] = st["ne_dose"]
            # --- 类固醇延迟起效 ---
            if st["steroid_on"]:
                st["steroid_age"] += 1
            steroid_fac = 1.0
            if st["steroid_on"] and st["steroid_age"] >= 2:
                steroid_fac = 0.85                        # 升压药需求下降
            # --- 血管张力 ---
            svr = st["svr_base"] + st["svr_gain"] * 1.8 * st["ne_effect"] * steroid_fac
            # --- 液体过负荷累积（低收缩力 OR 容量无反应者接受补液 → 静脉淤张/过负荷）---
            nonresponder_fluid = st.get("fluid_response", 0.5) < 0.35
            if st["fluid_total"] > 900 and (st["contractility"] < 0.8 or nonresponder_fluid):
                st["fluid_overload"] = min(1.0, st["fluid_overload"] + (0.04 if st["contractility"] < 0.8 else 0.025) * prec["substeps"] * noise)
                st["fluid_overload"] = min(1.0, st["fluid_overload"] + 0.04 * prec["substeps"] * noise)
            st["fluid_overload"] = max(0.0, st["fluid_overload"] - 0.01)  # 缓慢代谢
            # --- Starling（表型增益在此生效）+ 过负荷压制 + 升压药轻度正性肌力 ---
            sv_factor = starling_curve(st["preload"], st["contractility"]) * st["starling_gain"]
            sv_factor *= (1.0 - 0.35 * st["fluid_overload"]) * (1.0 + 0.12 * st["ne_effect"])
            sv_factor = min(sv_factor, 1.9)
            # --- MAP ---
            MAP = 36.0 + 30.0 * sv_factor * (0.55 + 0.45 * min(svr, 1.9))
            MAP += rng.normal(0, 1.6 * noise)
            st["map"] = float(np.clip(MAP, 35, 130))
            # --- 乳酸动力学（后负荷有害：产生增加）---
            harm = 1.18 if st.get("afterload_harm") else 1.0
            if prec["lactate_dynamics"]:
                clear = 0.42 * _perfusion(st["map"]) * (1.15 if (st["steroid_on"] and st["steroid_age"] >= 2) else 1.0)
                prod = (0.10 + 0.03 * st["sofa"] / 10.0) * harm
                st["lactate"] = float(max(0.8, st["lactate"] + (prod - clear) / prec["substeps"]))
            else:
                st["lactate"] = float(max(0.8, st["lactate"] - 0.06 * (_perfusion(st["map"]) - 0.45)))
            # --- 尿量 ---
            urine = 34.0 * _perfusion(st["map"]) * (1.0 - 0.5 * st["fluid_overload"])
            st["urine"] = float(max(0.0, urine + rng.normal(0, 2.0 * noise)))
            # --- HR / SpO2 / RR 伴随 ---
            st["hr"] = float(np.clip(st["hr"] + rng.normal(-0.8, 2.0 * noise) + (70 - st["map"]) * 0.05, 55, 160))
            st["spo2"] = float(np.clip(st["spo2"] + rng.normal(0.05, 0.4 * noise) + (_perfusion(st["map"]) - 0.5) * 0.6, 80, 100))
            st["rr"] = float(np.clip(st["rr"] + rng.normal(-0.1, 0.6 * noise), 10, 45))
            st["cvp"] = float(np.clip(4.0 + 9.0 * st["preload"] + rng.normal(0, 0.8 * noise), 0, 25))
            # --- SOFA 漂移 ---
            sofa_push = (st["lactate"] - 2.0) * 0.15 + st["fluid_overload"] * 0.8 + (0.5 if st["urine"] < 15 else 0)
            st["sofa"] = float(np.clip(st["sofa"] + rng.normal(0, 0.25 * noise) + np.sign(sofa_push) * 0.06 * min(abs(sofa_push), 2), 1, 20))
            st["preload"] = max(0.3, st["preload"] - 0.008)   # 液体再分布
        st["step"] += 1
        rec = snapshot(st)
        st["history"].append(rec)
        records.append(rec)
    return records


def snapshot(st):
    return {
        "step": st["step"], "map": round(st["map"], 1), "hr": round(st["hr"], 0),
        "lactate": round(st["lactate"], 2), "urine": round(st["urine"], 1),
        "ne_dose": round(st["ne_dose"], 3), "sofa": st["sofa"],
        "spo2": round(st["spo2"], 0), "rr": round(st["rr"], 0),
        "cvp": round(st["cvp"], 1), "fluid_total": round(st["fluid_total"], 0),
        "fluid_overload": round(st["fluid_overload"], 2),
        "steroid_on": st["steroid_on"],
    }


def features(st):
    """MAB 特征向量（与 pool_config.mab.feature_names 对齐，已归一化）。"""
    return np.array([
        (st["age"] - 55.0) / 25.0,
        st["sofa"] / 12.0,
        st["lactate"] / 6.0,
        st["map"] / 100.0,
        st["urine"] / 60.0,
        st["ne_dose"] / 1.0,
        st["comorbidity"] / 4.0,
        st["cvp"] / 15.0,
        st["fluid_response"],
    ])


def survival_prob(st, lactate0=None):
    """由最终状态估计 28 天生存概率（演示用 logistic 映射）。"""
    sev = (1.15 * (st["lactate"] - 1.5) + 0.28 * st["sofa"]
           + 1.8 * st["fluid_overload"] + 0.6 * (st["ne_dose"] > 0.55)
           + 0.35 * st["comorbidity"] - 0.4)
    p = 1.0 / (1.0 + np.exp(sev - 4.4))
    return float(np.clip(p, 0.02, 0.98))


def simulate_outcome(st0, action_id, rng, horizon=6, precision=None, t2_action=None):
    """从状态副本出发，执行动作并推进 horizon 步，返回结局向量。

    outcomes: survival, lactate_clearance, fluid_safety
    """
    import copy
    st = copy.deepcopy(st0)
    lactate0 = max(st["lactate"], 0.5)
    apply_action(st, action_id, rng)
    if t2_action:
        apply_t2(st, t2_action)
    advance(st, rng, steps=horizon, override_precision=precision)
    surv = survival_prob(st)
    lac_clear = float(np.clip((lactate0 - st["lactate"]) / lactate0, -0.5, 1.0))
    fluid_safety = float(1.0 - np.clip(st["fluid_overload"], 0.0, 1.0))
    return {
        "survival": round(surv, 3),
        "lactate_clearance": round(lac_clear, 3),
        "fluid_safety": round(fluid_safety, 3),
        "final_state": snapshot(st),
        "trajectory": st["history"][-horizon:],
    }


def composite_reward(outcomes, weights):
    """演示奖励：加权合成到 [0,1]。"""
    w = weights
    total = w.get("survival", 0.5) + w.get("lactate_clearance", 0.3) + w.get("fluid_safety", 0.2)
    r = (w.get("survival", 0.5) * outcomes["survival"]
         + w.get("lactate_clearance", 0.3) * max(outcomes["lactate_clearance"], 0.0)
         + w.get("fluid_safety", 0.2) * outcomes["fluid_safety"]) / total
    return float(np.clip(r, 0.0, 1.0))
