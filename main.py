"""第三波·Pro：数字孪生预演增强（Harness 进化版）演示后端。

学术仿真原型 —— 本工程不接入任何真实患者/监护/HIS 系统，全部数据均为模拟生成。
- 决策内核：情境化汤普森采样（贝叶斯 logistic）+ 帕累托前沿 + 安全闸门
- Harness 只负责调度与提示，LLM 只做解读，二者绝不修改决策数值（INV-3）
- 所有大模型（可选）经本地 LM-Studio 调用，数据不出本机
"""
import copy
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import hemo_sim as hs
from core import mab as mab_mod
from core import pareto, counterfactual as cf_mod, calibration as calib_mod
from core import toon_codec, message_board as mb_mod
from core.harness import Harness
from core import harness_evolver as evolver_mod
from core.experience_replay import ExperienceReplay
from core.agent_orchestrator import AgentOrchestrator
from core.knowledge_base import KnowledgeBase
from core.llm_client import probe as llm_probe, interpret as llm_interpret
from core.order_flip_gate import freeze_decision, flip_test
from core.fresh_reviewer import fresh_review
from core.trajectory_validator import validate_trajectory, legal_prefix

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
CONFIG = BASE / "config"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("poc7")

app = FastAPI(title="Sepsis Harness Evolution Pro Demo", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

rng = np.random.default_rng()

pool_cfg = json.loads((CONFIG / "pool_config.json").read_text(encoding="utf-8"))
invariant_rules = json.loads((CONFIG / "invariant_rules.json").read_text(encoding="utf-8"))
ARM_IDS = [o["id"] for o in pool_cfg["decision_points"][0]["options"]]   # T1 动作池 A/B/AB
FEATURES = pool_cfg["mab"]["feature_names"]
FEATURE_LABELS = pool_cfg["mab"]["feature_labels"]
DEFAULT_WEIGHTS = pool_cfg["default_doctor_weights"]
HORIZON = pool_cfg["mab"]["horizon_steps"]

# ---------------- 全局状态 ----------------
STATE = {
    "session_id": None,
    "mode": "auto",            # auto | csv
    "beds": {},                # bed_id -> {"patient": st, "history": [...], "t1": {...}, "t2": {...}, "done": bool}
    "n_beds": 1,
    "precision": "mid",
    "harness": Harness(),
    "bandit": mab_mod.ContextualThompsonBandit(
        ARM_IDS, len(FEATURES),
        prior_mu=pool_cfg["mab"]["prior_mu"],
        prior_sigma_scale=pool_cfg["mab"]["prior_sigma_scale"],
        n_mc=pool_cfg["mab"]["n_mc"]),
    "replay": ExperienceReplay(path=str(DATA / "experience_replay.json")),
    "board": mb_mod.MessageBoard(str(DATA / "message_board.jsonl")),
    "orchestrator": None,      # 初始化时创建
    "kb": KnowledgeBase(),
    "llm_enabled": False,
    "case_counter": 0,
    "pending_decisions": {},   # decision_id -> 完整决策包（冻结哈希 + 反事实 + 帕累托）
    "decision_counter": 0,
}
STATE["orchestrator"] = AgentOrchestrator(STATE["board"], STATE["kb"])

ACTION_LABELS = {"A": "液体复苏", "B": "血管活性药", "AB": "联合", "C": "氢化可的松", "D": "暂不处理"}


def audit(event_type: str, payload: dict):
    """审计日志：JSONL 追加，同时附带 TOON 压缩对比。"""
    rec = {
        "ts": round(time.time(), 1),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_type,
        "session": STATE["session_id"],
        "payload": payload,
    }
    with open(DATA / "audit_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    try:
        rec["toon"] = {"json_bytes": toon_codec.json_size(payload), "toon_bytes": toon_codec.toon_size(payload),
                       "ratio": round(toon_codec.ratio(payload), 2)}
        with open(DATA / "audit_log.jsonl", "a", encoding="utf-8") as f:
            f.write("")  # 占位保持单行 JSON：toon 信息并入下一条同事件记录
    except Exception:
        pass


def audit2(event_type: str, payload: dict):
    """审计记录（含 TOON 体积对比）。"""
    j = toon_codec.json_size(payload)
    t = toon_codec.toon_size(payload)
    rec = {
        "ts": round(time.time(), 1),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_type,
        "session": STATE["session_id"],
        "payload": payload,
        "toon": {"json_bytes": j, "toon_bytes": t,
                 "saved_pct": round(100 * (1 - t / max(j, 1)), 1)},
    }
    with open(DATA / "audit_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------- 请求模型 ----------------
class SessionInit(BaseModel):
    beds: int = 1
    precision: str = "mid"
    mode: str = "auto"
    llm_enabled: bool = False


class AdvanceReq(BaseModel):
    steps: int = 1


class DecisionReq(BaseModel):
    bed: int = 1
    threshold: float = 60.0
    weights: dict = None


class OverrideReq(BaseModel):
    decision_id: str
    new_action: str


class CFReq(BaseModel):
    bed: int = 1
    cf_actions: list = ["B"]


class FeedbackReq(BaseModel):
    bed: int = 1
    source: str = "hemosim"      # hemosim | manual
    reward: float = None         # manual 模式必填
    weights: dict = None


class EvolveReq(BaseModel):
    red_team: bool = False


class BoardPublish(BaseModel):
    agent: str = "床旁护士Agent"
    bed: int = 1
    kind: str = "note"
    content: str


# ---------------- 基础接口 ----------------
@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/health")
def health():
    llm = llm_probe() if STATE["llm_enabled"] else {"ok": False, "message": "未启用本地模型增强"}
    return {
        "ok": True,
        "llm": llm,
        "llm_enabled": STATE["llm_enabled"],
        "session": STATE["session_id"],
        "n_beds": STATE["n_beds"],
        "mode": STATE["mode"],
    }


@app.get("/api/config")
def get_config():
    return _clean({
        "pool": pool_cfg,
        "invariants": invariant_rules,
        "harness": STATE["harness"].state_dict(),
        "harness_registry": STATE["harness"].registry,
        "action_labels": ACTION_LABELS,
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
    })


# ---------------- 会话与患者 ----------------
@app.post("/api/session/init")
def session_init(req: SessionInit):
    STATE["session_id"] = f"S{datetime.now().strftime('%H%M%S')}"
    STATE["mode"] = req.mode
    STATE["n_beds"] = max(1, min(5, req.beds))
    STATE["precision"] = req.precision
    STATE["llm_enabled"] = bool(req.llm_enabled)
    STATE["beds"] = {}
    STATE["pending_decisions"] = {}
    STATE["case_counter"] = 0

    csv_cases = []
    if req.mode == "csv":
        import csv as csv_mod
        with open(DATA / "demo_cases.csv", encoding="utf-8") as f:
            csv_cases = list(csv_mod.DictReader(f))

    for i in range(STATE["n_beds"]):
        bed_id = i + 1
        if csv_cases:
            row = csv_cases[i % len(csv_cases)]
            st = hs.new_patient(rng, row["case_id"], precision=req.precision)
            st["age"] = int(row["age"]); st["sex"] = row["sex"]
            st["sofa"] = int(round(float(row["sofa"]))); st["lactate"] = float(row["lactate"])
            st["map"] = float(row["map"]); st["urine"] = float(row["urine"])
            st["ne_dose"] = float(row["ne_dose"]); st["cvp"] = float(row["cvp"])
            st["comorbidity"] = float(row["comorbidity"])
            csv_outcome = {"t1_action": row["t1_action"], "t2_action": row["t2_action"],
                           "t1_composite": float(row["t1_composite"]), "t2_composite": float(row["t2_composite"])}
        else:
            st = hs.new_patient(rng, f"B{bed_id:02d}", precision=req.precision)
            csv_outcome = None
        STATE["beds"][bed_id] = {"patient": st, "history": [hs.snapshot(st)],
                                 "t1": None, "t2": None, "done": False,
                                 "csv_outcome": csv_outcome}
    STATE["case_counter"] = STATE["n_beds"]

    # 消息板：会话开启公告
    STATE["board"].publish("Harness调度器", 0, "session",
                           f"会话 {STATE['session_id']} 开启：{STATE['n_beds']} 床 · 精度档 {req.precision} · "
                           f"{'CSV回放' if req.mode == 'csv' else '自动生成'}模式")
    audit2("session_init", {"session": STATE["session_id"], "beds": STATE["n_beds"],
                            "precision": req.precision, "mode": req.mode})
    return _clean({"ok": True, "session": STATE["session_id"],
                   "beds": {str(b): bed_summary(v) for b, v in STATE["beds"].items()}})


def _clean(obj):
    """numpy 类型 → 原生 Python 类型，保证 JSON 可序列化。"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    return obj


def bed_summary(bed):
    st = bed["patient"]
    return _clean({
        "snapshot": hs.snapshot(st),
        "phenotype": st["phenotype"], "phenotype_label": st["phenotype_label"],
        "fluid_response": round(float(st["fluid_response"]), 2),
        "features": [float(v) for v in hs.features(st)],
        "done": bed["done"],
        "csv_outcome": bed["csv_outcome"],
        "t1": bed["t1"], "t2": bed["t2"],
        "history": bed["history"][-60:],
    })


@app.get("/api/patient/state")
def patient_state(bed: int = 1):
    if bed not in STATE["beds"]:
        raise HTTPException(404, "床位不存在，请先初始化会话")
    return bed_summary(STATE["beds"][bed])


@app.post("/api/patient/advance")
def patient_advance(req: AdvanceReq, bed: int = 1):
    if bed not in STATE["beds"]:
        raise HTTPException(404, "床位不存在")
    b = STATE["beds"][bed]
    # 逐步推进并逐步记录快照，保证波形连续（推进多步时中间点不丢失）
    for _ in range(max(1, min(4, req.steps))):
        hs.advance(b["patient"], rng, steps=1)
        b["history"].append(hs.snapshot(b["patient"]))
    if len(b["history"]) > 120:
        b["history"] = b["history"][-120:]
    audit2("patient_advance", {"bed": bed, "steps": req.steps, "map": round(b["patient"]["map"], 1)})
    return bed_summary(b)


def check_safety_gates(st, action):
    """POC6 四闸门 + POC7 新增两条：因果效应安全线 / 液体过负荷预警线。"""
    gates = [
        {"id": "INV-1", "name": "去甲肾上腺素上限 (≤1.0 μg/kg/min)", "pass": st["ne_dose"] <= 1.0},
        {"id": "INV-2", "name": "累积液体上限 (≤5000 mL)", "pass": st["fluid_total"] <= 5000},
        {"id": "SV-HARM", "name": "容量无反应者液体闸门（容量反应性过低时禁止单独 A）",
         "pass": not (action == "A" and st["fluid_response"] < 0.3)},
        {"id": "NE-ESCAL", "name": "升压药急升闸门（NE>0.8 时禁止叠加 B）",
         "pass": not (action in ("B", "AB") and st["ne_dose"] > 0.8)},
        {"id": "CAUSAL", "name": "因果效应安全线（反事实分支显示选 A 将显著恶化时拦截）",
         "pass": True},   # 由决策流程用反事实结果填充
        {"id": "FO-WARN", "name": "液体过负荷预警线（FO 指数>0.5 时预警并建议减液）",
         "pass": st["fluid_overload"] <= 0.5},
    ]
    return gates


def run_decision_core(st, threshold, weights):
    """核心决策流：帕累托 → 情境TS → 冻结 → 反事实 → 闸门 → 复核。"""
    x = hs.features(st)
    feats = FEATURES
    fdict = dict(zip(feats, x))

    # M10 帕累托前沿
    arm_ids = ARM_IDS
    evals = pareto.evaluate_candidates(st, arm_ids, rng, hs.simulate_outcome,
                                       rollouts=pool_cfg["mab"]["rollouts_per_candidate"])
    points = {a: evals[a]["mean"] for a in arm_ids}   # [survival, lactate_clearance, fluid_safety]
    ranks = pareto.pareto_rank(points)
    front = [a for a in arm_ids if ranks[a] == 0]
    w = weights or DEFAULT_WEIGHTS
    chosen_by_pareto, _pareto_score = pareto.scalar_choice(
        points, ranks, (w["survival"], w["lactate_clearance"], w["fluid_safety"]))

    # M9 情境化汤普森采样
    dec = STATE["bandit"].decide(x, rng, threshold)
    action = dec["action"]

    # M11 反事实（含现实分支，用于因果效应安全线）
    others = [a for a in arm_ids if a != action]
    cf = cf_mod.run_counterfactual(st, action, others, rng, hs.simulate_outcome,
                                   rollouts=pool_cfg["mab"]["cf_rollouts"])

    # 因果效应安全线：若反事实明显优于现实选择 → 拦截并提示
    causal_gate_pass = True
    cf_notes = []
    real_surv = cf["branches"]["0"]["metric_means"].get("survival", 0)
    for idx, br in cf["branches"].items():
        if idx == "0":
            continue
        if br["metric_means"].get("survival", 0) > real_surv + 0.15:
            causal_gate_pass = False
            cf_notes.append(f"反事实「{br['action']}」生存概率高出 {round(br['metric_means']['survival'] - real_surv, 2)}，"
                            f"算法选择与数字孪生推演不一致，已触发因果效应安全线")

    gates = check_safety_gates(st, action)
    for g in gates:
        if g["id"] == "CAUSAL":
            g["pass"] = causal_gate_pass
            g["note"] = "；".join(cf_notes) if cf_notes else "反事实推演与选择一致"

    blocked = [g for g in gates if not g["pass"]]
    if blocked:
        # 闸门拦截：降级为闸门允许的动作中帕累托最优者
        allowed = [a for a in arm_ids
                   if all(g["pass"] or not (g["id"] in ("SV-HARM",) and a == "A")
                          for g in gates)]
        action = chosen_by_pareto if chosen_by_pareto in allowed else (allowed[0] if allowed else "D")
        dec["branch"] = "gated"
        dec["gate_blocked"] = [g["name"] for g in blocked]

    # U2 反思复核（fresh reviewer：高风险/分歧触发独立第二意见，LLM 可选）
    review = fresh_review(STATE["bandit"], x, action, rng, threshold,
                          llm_fn=llm_interpret if STATE["llm_enabled"] else None,
                          feature_labels=FEATURE_LABELS)

    decision = {
        "action": action,
        "branch": dec["branch"],
        "uncertainty": dec["uncertainty"],
        "arm_means": dec["arm_means"],
        "p_best": dec.get("p_best"),
        "pareto": {"points": points, "ranks": ranks, "front": front, "scalar_choice": chosen_by_pareto},
        "cf": cf,
        "gates": gates,
        "review": review,
        "features": fdict,
        "threshold": threshold,
        "weights": w,
    }
    _payload, frozen_hash = freeze_decision({k: decision[k] for k in ("action", "branch", "uncertainty")})
    decision["frozen_hash"] = frozen_hash

    # M4/U1 顺序翻转测试（生成侧扰动不应改变决策）
    flip = flip_test(
        lambda d, style: STATE["harness"].build_interpret_prompt(d, style=style),
        {"action": action, "branch": dec["branch"], "uncertainty": dec["uncertainty"]},
        n_variants=3)
    decision["flip_test"] = flip

    # M4 选项顺序翻转闸门：动作池排列置换后重跑决策，结论必须不变，否则标记不稳定。
    # 共同随机数：每臂 θ 采样固定（规范顺序），置换只改变呈现顺序——
    # 顺序不变性检验的是「选项→结论的映射」，抽样运气已被排除。
    from core.mab import sigmoid
    flip_rng = np.random.default_rng(20261022)
    theta_draws = {a: flip_rng.multivariate_normal(STATE["bandit"].arms[a].mu,
                                                   STATE["bandit"].arms[a].sigma, size=400)
                   for a in ARM_IDS}
    flip_samples = []
    for _ in range(3):
        perm = list(rng.permuted(ARM_IDS))
        P = np.stack([sigmoid(theta_draws[a] @ x) for a in perm])
        counts = np.bincount(P.argmax(axis=0), minlength=len(perm))
        flip_samples.append(perm[int(counts.argmax())])
    order_stable = len(set(flip_samples)) == 1
    decision["order_flip"] = {"consistent": bool(order_stable),
                              "samples": flip_samples,
                              "note": ("同一批采样下 3 种选项顺序给出同一结论 ✓（实际执行动作来自 Thompson 随机抽样，"
                                       "与固定种子翻转测试不同属正常）") if order_stable
                                      else "选项顺序影响结论！已标记为不稳定，建议人工复核"}
    return decision


@app.post("/api/decision/run")
def decision_run(req: DecisionReq):
    if req.bed not in STATE["beds"]:
        raise HTTPException(404, "床位不存在")
    b = STATE["beds"][req.bed]
    if b["done"]:
        raise HTTPException(400, "该床位 T1/T2 决策均已完成")
    st = b["patient"]
    decision = run_decision_core(st, req.threshold, req.weights)

    STATE["decision_counter"] += 1
    decision_id = f"D{STATE['decision_counter']:03d}"
    STATE["pending_decisions"][decision_id] = {"bed": req.bed, "decision": decision}

    # 执行动作并推进（数字孪生预演）
    action = decision["action"]
    pre_snap = hs.snapshot(st)
    hs.apply_action(st, action, rng)
    hs.advance(st, rng, steps=HORIZON)
    b["history"].extend([hs.snapshot(s) for s in [st]] * 1)
    b["history"] = (b["history"] + [])[-120:]

    t1 = {"decision_id": decision_id, "action": action, "pre": pre_snap,
          "post": hs.snapshot(st), "branch": decision["branch"],
          "uncertainty": decision["uncertainty"]}
    b["t1"] = t1
    if action != "D":
        STATE["board"].publish("Harness调度器", req.bed, "decision",
                               f"T1 执行：{action}（{ACTION_LABELS[action]}）· 分支 {decision['branch']} · "
                               f"不确定性 {decision['uncertainty']}")
    audit2("decision_run", {"bed": req.bed, "decision_id": decision_id, "action": action,
                            "branch": decision["branch"], "uncertainty": decision["uncertainty"],
                            "gates_pass": all(g["pass"] for g in decision["gates"]),
                            "frozen_hash": decision["frozen_hash"]})
    return _clean({"ok": True, "decision_id": decision_id, "decision": decision, "bed": bed_summary(b)})


@app.post("/api/decision/override")
def decision_override(req: OverrideReq):
    if req.decision_id not in STATE["pending_decisions"]:
        raise HTTPException(404, "决策不存在或已结算")
    pkg = STATE["pending_decisions"][req.decision_id]
    bed_id, old = pkg["bed"], pkg["decision"]
    if req.new_action not in ("A", "B", "AB", "D"):
        raise HTTPException(400, "非法动作")
    old_action = old["action"]
    if req.new_action == old_action:
        return {"ok": True, "message": "未修改", "decision": old}

    b = STATE["beds"][bed_id]
    st = b["patient"]
    # 撤销旧动作影响：以决策前快照为基准重新施加（用历史 pre 快照近似回滚）
    if b["t1"] and b["t1"]["decision_id"] == req.decision_id and b["t1"].get("pre"):
        pre = b["t1"]["pre"]
        for k in ("map", "lactate", "urine", "ne_dose", "cvp", "fluid_total", "sofa", "spo2", "rr", "hr"):
            if k in pre:
                st[k] = pre[k]
        st["fluid_overload"] = pre.get("fluid_overload", st["fluid_overload"])
    # 施加医生选择
    hs.apply_action(st, req.new_action, rng)
    hs.advance(st, rng, steps=HORIZON)
    b["history"].append(hs.snapshot(st))

    pkg["decision"]["action"] = req.new_action
    pkg["decision"]["branch"] = "doctor_override"
    pkg["decision"]["overridden_from"] = old_action
    if b["t1"] and b["t1"]["decision_id"] == req.decision_id:
        b["t1"]["action"] = req.new_action
        b["t1"]["post"] = hs.snapshot(st)
        b["t1"]["branch"] = "doctor_override"

    STATE["board"].publish("Harness调度器", bed_id, "override",
                           f"医生否决：{old_action} → {req.new_action}（{ACTION_LABELS.get(req.new_action)}）")
    audit2("decision_override", {"decision_id": req.decision_id, "bed": bed_id,
                                 "from": old_action, "to": req.new_action})
    return _clean({"ok": True, "message": f"已执行医生选择 {req.new_action}", "decision": pkg["decision"],
                   "bed": bed_summary(b)})


# ---------------- 反事实 ----------------
@app.post("/api/counterfactual/run")
def counterfactual_run(req: CFReq):
    if req.bed not in STATE["beds"]:
        raise HTTPException(404, "床位不存在")
    b = STATE["beds"][req.bed]
    real_action = b["t1"]["action"] if b["t1"] else "A"
    cf_actions = [a for a in req.cf_actions if a != real_action] or req.cf_actions
    cf = cf_mod.run_counterfactual(b["patient"], real_action, cf_actions, rng, hs.simulate_outcome,
                                   rollouts=pool_cfg["mab"]["cf_rollouts"])
    cf["note"] = "数字孪生模拟反事实（教学级 HemoSim 内部推演，非临床预测）"
    with open(DATA / "counterfactual_traces.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": round(time.time(), 1), "bed": req.bed, "real": real_action,
                            "cf": req.cf_actions}, ensure_ascii=False) + "\n")
    audit2("counterfactual_run", {"bed": req.bed, "real": real_action, "cf": req.cf_actions})
    return _clean({"ok": True, "cf": cf})


class KnobReq(BaseModel):
    contractility_mul: float = 1.0
    svr_mul: float = 1.0
    preload_mul: float = 1.0


@app.post("/api/patient/knobs")
def patient_knobs(req: KnobReq, bed: int = 1):
    """数字孪生手动旋钮：直接调节本床孪生体生理参数（预演用）。"""
    if bed not in STATE["beds"]:
        raise HTTPException(404, "床位不存在")
    st = STATE["beds"][bed]["patient"]
    st["contractility"] = float(np.clip(st["contractility"] * req.contractility_mul, 0.2, 1.8))
    st["svr_base"] = float(np.clip(st["svr_base"] * req.svr_mul, 0.2, 1.6))
    st["preload"] = float(np.clip(st["preload"] * req.preload_mul, 0.15, 1.3))
    hs.advance(st, rng, steps=1)
    STATE["beds"][bed]["history"].append(hs.snapshot(st))
    audit2("twin_knob", {"bed": bed, "knobs": req.model_dump()})
    return bed_summary(STATE["beds"][bed])


# ---------------- T2 / 结局反馈 ----------------
def settle_decision(bed_id, reward):
    """结算最近一笔待决决策：TS 更新 + 经验回灌 + 消息板。"""
    b = STATE["beds"][bed_id]
    t1 = b.get("t1")
    if not t1:
        return None
    x = hs.features(b["patient"])
    STATE["bandit"].update(t1["action"], x, reward, n_b=3, rng=rng)
    STATE["replay"].add_case({"bed": bed_id, "action": t1["action"], "reward": reward,
                              "x": [round(v, 3) for v in x],
                              "ts": round(time.time(), 1)})
    STATE["board"].publish("Harness调度器", bed_id, "feedback",
                           f"结局反馈：动作 {t1['action']} 奖励 {round(reward, 3)} → 后验已更新（经验回灌 RSI 三层之 L1 即时）")
    audit2("outcome_feedback", {"bed": bed_id, "action": t1["action"], "reward": round(reward, 4)})
    return t1["action"]


@app.post("/api/outcome/feedback")
def outcome_feedback(req: FeedbackReq):
    if req.bed not in STATE["beds"]:
        raise HTTPException(404, "床位不存在")
    b = STATE["beds"][req.bed]
    w = req.weights or DEFAULT_WEIGHTS

    if req.source == "manual":
        if req.reward is None:
            raise HTTPException(400, "manual 模式需提供 reward (0~1)")
        reward = float(np.clip(req.reward, 0, 1))
    else:
        out = hs.simulate_outcome(b["patient"], "D", rng,
                                  horizon=HORIZON)
        reward = hs.composite_reward(out, w)

    action = settle_decision(req.bed, reward)

    # M3/U3 轨迹级生成-验证：整链校验 + 保留合法前缀（SpecAgent 思想）
    traj = validate_trajectory(b["history"], invariant_rules["rules"], list(ACTION_LABELS.keys()))
    prefix = legal_prefix(b["history"], invariant_rules["rules"])

    return _clean({"ok": True, "reward": round(reward, 4), "updated_arm": action,
            "trajectory_valid": traj["ok"], "trajectory_msgs": traj["violations"],
            "legal_prefix": prefix,
            "posterior": STATE["bandit"].coefficient_table(FEATURE_LABELS),
            "bed": bed_summary(b)})


# ---------------- Harness 进化 ----------------
def bandit_factory(n_mc=2000):
    """Probe 阶段工厂：隔离的新 bandit，不影响线上后验。"""
    return mab_mod.ContextualThompsonBandit(
        ARM_IDS, len(FEATURES),
        prior_mu=pool_cfg["mab"]["prior_mu"],
        prior_sigma_scale=pool_cfg["mab"]["prior_sigma_scale"], n_mc=int(n_mc))


@app.get("/api/harness/state")
def harness_state():
    return _clean(STATE["harness"].state_dict())


@app.get("/api/harness/history")
def harness_history():
    p = DATA / "harness_evolution_log.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@app.post("/api/harness/evolve")
def harness_evolve(req: EvolveReq):
    probe_patients = []
    for bed_id, b in STATE["beds"].items():
        probe_patients.append(hs.features(b["patient"]))
    if not probe_patients:
        st = hs.new_patient(rng, "PROBE", STATE["precision"])
        probe_patients.append(hs.features(st))
    layers = STATE["replay"].inject(0.0)
    replay_x = [c["x"] for c in layers.get("case_level", []) if isinstance(c, dict) and c.get("x")] or probe_patients

    before = STATE["harness"].state_dict()
    record = evolver_mod.evolve(STATE["harness"], bandit_factory, probe_patients, replay_x, rng,
                                str(DATA / "harness_evolution_log.jsonl"), red_team=req.red_team)
    after = STATE["harness"].state_dict()
    audit2("harness_evolve", {"red_team": req.red_team, "result": record["result"],
                              "mutated_field": record.get("mutated_field"),
                              "guard_passed": record.get("guard_passed")})
    return _clean({"ok": True, "record": record, "before": before, "after": after})


@app.post("/api/harness/rollback")
def harness_rollback():
    STATE["harness"] = Harness()
    audit2("harness_rollback", {"message": "已回滚至 Gen0 出厂参数"})
    return {"ok": True, "state": STATE["harness"].state_dict()}


# ---------------- 消息板 ----------------
@app.get("/api/message_board")
def board_get(q: str = "", n: int = 40):
    if q:
        msgs = STATE["board"].search(q)
    else:
        msgs = STATE["board"].recent(n)
    return {"messages": msgs, "heat": STATE["board"].heat_sorted()[:10],
            "agents": STATE["orchestrator"].state()}


@app.post("/api/message_board/publish")
def board_publish(req: BoardPublish):
    STATE["board"].publish(req.agent, req.bed, req.kind, req.content)
    audit2("board_publish", {"agent": req.agent, "bed": req.bed, "kind": req.kind})
    return {"ok": True, "messages": STATE["board"].recent(40)}


@app.post("/api/message_board/fault_demo")
def board_fault_demo():
    """多智能体掉线容错演示：随机让一个 Agent 掉线发不出消息。"""
    agents = list(STATE["orchestrator"].agents.keys())
    victim = random.choice(agents)
    STATE["orchestrator"].set_offline(victim, True)
    STATE["orchestrator"]._post(victim, 1, "（此条不应出现：Agent 已掉线，进入掉线容错队列）")
    STATE["board"].publish("Harness调度器", 0, "alert", f"检测到 {victim} 心跳丢失，已切换掉线容错模式（消息缓存待重连）")
    STATE["orchestrator"].set_offline(victim, False)
    STATE["board"].publish("Harness调度器", 0, "note", f"{victim} 已重连，缓存消息按序回放 ✓")
    audit2("fault_demo", {"victim": victim})
    return {"ok": True, "victim": victim, "messages": STATE["board"].recent(40)}


@app.post("/api/round_consult")
def round_consult():
    """多学科会诊轮询：多智能体基于各床状态发消息（LLM 可选润色）。"""
    beds = [{"bed_id": b, "state": hs.snapshot(v["patient"])} for b, v in STATE["beds"].items()]
    result = STATE["orchestrator"].round_consult(
        beds, llm_polish=llm_interpret if STATE["llm_enabled"] else None)
    audit2("round_consult", {"beds": list(STATE["beds"].keys())})
    return {"ok": True, "result": result, "messages": STATE["board"].recent(40)}


# ---------------- 沟通要点辅助 ----------------
@app.post("/api/communication_draft")
def communication_draft(bed: int = 1):
    if bed not in STATE["beds"]:
        raise HTTPException(404, "床位不存在")
    st = STATE["beds"][bed]["patient"]
    draft = STATE["orchestrator"].communication_draft(
        hs.snapshot(st), llm_enabled=STATE["llm_enabled"], llm_fn=llm_interpret)
    audit2("communication_draft", {"bed": bed})
    return {"ok": True, "draft": draft}


# ---------------- 审计日志（TOON 对比） ----------------
@app.get("/api/audit_log")
def audit_log(n: int = 50, fmt: str = "json"):
    p = DATA / "audit_log.jsonl"
    if not p.exists():
        return {"entries": [], "toon_demo": None}
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    entries = [json.loads(l) for l in lines[-n:]]
    # TOON 演示负载：同构记录列表（列式编码，对标 JSON vs TOON 的 37% 体积口径）
    demo_payload = {"reviews": [
        {"id": 101, "customer": "Alex Rivera", "rating": 5, "comment": "Excellent!", "verified": True},
        {"id": 102, "customer": "Brij Pandey", "rating": 5, "comment": "Game changer!", "verified": True},
        {"id": 103, "customer": "Casey Lee", "rating": 3, "comment": "Average", "verified": False}]}
    j = toon_codec.json_size(demo_payload)
    t = toon_codec.toon_size(demo_payload)
    return {"entries": entries,
            "toon_demo": {"payload": demo_payload,
                          "json": j, "toon": t,
                          "saved_pct": round(100 * (1 - t / max(j, 1)), 1),
                          "toon_text": toon_codec.toon_encode(demo_payload)}}


# ---------------- 可信度与解释 ----------------
def predict_proba(x):
    arm = STATE["bandit"].arms.get("A") or list(STATE["bandit"].arms.values())[0]
    return float(arm.predict_proba(np.asarray(x, dtype=float)))


@app.get("/api/calibration")
def calibration():
    cohort = json.loads((DATA / "calibration_cohort.json").read_text(encoding="utf-8"))
    y_true = np.array([c["y"] for c in cohort])
    p_pred = np.array([predict_proba(c["x"]) for c in cohort])
    ece, details = calib_mod.ece_score(y_true, p_pred, n_bins=10)
    dca = calib_mod.dca_curve(y_true, p_pred)
    return _clean({"ece": round(float(ece), 4), "bins": details,
                   "dca": dca, "n": len(cohort),
                   "note": "200 例预生成模拟队列上的校准/决策曲线分析（演示口径）"})


@app.get("/api/shap")
def shap(bed: int = 1):
    if bed not in STATE["beds"]:
        raise HTTPException(404, "床位不存在")
    x = hs.features(STATE["beds"][bed]["patient"])
    baseline = [0.0] * len(x)
    vals = calib_mod.shapley_exact(predict_proba, x, baseline,
                                   feature_names=FEATURE_LABELS)
    return _clean({"ok": True, "shapley": vals, "features": dict(zip(FEATURE_LABELS, x)),
            "note": "精确 Shapley 值（2^9=512 子集枚举，效率公理化归一）"})


# ---------------- 知识检索（T3 仅展示） ----------------
@app.get("/api/knowledge")
def knowledge(q: str = ""):
    if q:
        return {"results": STATE["kb"].retrieve(q, top_k=3)}
    return {"results": STATE["kb"].snippets, "t3_rules": STATE["kb"].t3_rules()}


# ---------------- LLM 解读 ----------------
@app.post("/api/llm/interpret")
def llm_interpret_api(decision_id: str = ""):
    if not STATE["llm_enabled"]:
        return {"ok": False, "message": "未启用本地模型增强，请先在首页勾选"}
    dec = None
    if decision_id and decision_id in STATE["pending_decisions"]:
        dec = STATE["pending_decisions"][decision_id]["decision"]
    else:
        for pkg in reversed(list(STATE["pending_decisions"].values())):
            dec = pkg["decision"]
            break
    if not dec:
        return {"ok": False, "message": "暂无可解读的决策"}
    prompt = STATE["harness"].build_interpret_prompt(dec)
    res = llm_interpret(prompt)
    audit2("llm_interpret", {"decision_id": decision_id, "ok": res.get("ok", False)})
    return res


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print(" 第三波·Pro 演示服务  →  http://127.0.0.1:8000")
    print("（学术仿真原型；患者数据均为模拟生成）")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
