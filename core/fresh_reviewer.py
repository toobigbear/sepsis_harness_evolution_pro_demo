"""M2 Fresh Reviewer 独立复核。

对决策结论做一次"全新上下文"的独立采样复核（重跑 Thompson 采样 + 核对证据检索），
与首次结论不一致时给出告警。复核只读，不改写决策。
"""
import numpy as np


def fresh_review(bandit, x, first_action, rng, threshold, evidence_fn=None, llm_fn=None,
                 features=None, feature_labels=None):
    """Fresh 独立复核（M2，密歇根 Subagents 讲义 fresh 模式）。

    高风险（不确定性 ≥ 阈值）或结论分歧时触发独立第二意见：
    - 数值层：独立重采样 Thompson 决策；
    - 文本层（可选）：LLM 作为独立文本复核员，只读特征与结论、不回传主决策链。
    返回 {agree, second_action, note, triggered, llm_opinion}。
    """
    second = bandit.decide(x, rng, threshold=threshold)
    agree = second["action"] == first_action
    triggered = bool(second["uncertainty"] >= threshold or not agree)
    note = "独立复核与首次结论一致。" if agree else (
        f"独立复核建议 {second['action']}，与首次结论 {first_action} 不一致——"
        "提示该患者处于证据模糊区，建议医生重点复核床旁评估。")
    if triggered:
        note += "（已达 Fresh 复核触发条件：高风险/分歧）"
    evidence = evidence_fn() if evidence_fn else None
    llm_opinion = None
    if triggered and llm_fn and feature_labels:
        feats = "、".join(f"{n}={round(float(v),2)}" for n, v in zip(feature_labels, x))
        prompt = (f"你是独立复核医师（与决策系统无共享上下文）。已知算法首次结论 {first_action}，"
                  f"独立数值复核建议 {second['action']}。患者特征：{feats}。"
                  "请用三句话给出你的独立第二意见（只做解读，严禁修改或重新计算任何数值，"
                  "结尾注明【仿真演示，非临床建议】）。")
        r = llm_fn(prompt)
        if r.get("ok"):
            llm_opinion = r["text"]
    return {"agree": bool(agree), "second_action": second["action"],
            "second_uncertainty": second["uncertainty"], "note": note,
            "triggered": triggered, "evidence": evidence, "llm_opinion": llm_opinion}
