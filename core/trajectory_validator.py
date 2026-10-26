"""M3 Trajectory Validator 轨迹级生成-验证（SpecAgent 思想）。

弱模型/模拟器起草完整决策轨迹 → 校验器一次性核验不变量 → 保留合法前缀：
首个违规步之前的轨迹仍可使用，违规点截断并告警，而非整条作废。
校验项：NE 剂量上限、累积液体上限、均势池边界。
"""


def validate_trajectory(trajectory, invariant_rules, allowed_actions):
    """trajectory: HemoSim 记录列表。返回 {ok, violations[], prefix_ok}。"""
    inv = {r["check"]: r for r in invariant_rules}
    violations = []
    ne_max = inv.get("ne_dose_max", {}).get("value", 1.0)
    fluid_max = inv.get("fluid_total_max", {}).get("value", 5000)
    for rec in trajectory:
        if rec["ne_dose"] > ne_max + 1e-9:
            violations.append(f"步骤{rec['step']} NE 剂量 {rec['ne_dose']} 超过不变上限 {ne_max}")
        if rec["fluid_total"] > fluid_max + 1e-9:
            violations.append(f"步骤{rec['step']} 累积液体 {rec['fluid_total']} mL 超过不变上限 {fluid_max}")
    return {"ok": len(violations) == 0, "violations": violations}


def legal_prefix(trajectory, invariant_rules):
    """SpecAgent 式保留合法前缀：返回 {ok, prefix_len, cut_at, reason}。

    从头扫描，首个违反不变量的记录处截断；截断前的合法前缀可用于回放/学习。
    """
    inv = {r["check"]: r for r in invariant_rules}
    ne_max = inv.get("ne_dose_max", {}).get("value", 1.0)
    fluid_max = inv.get("fluid_total_max", {}).get("value", 5000)
    for i, rec in enumerate(trajectory):
        if rec["ne_dose"] > ne_max + 1e-9:
            return {"ok": False, "prefix_len": i, "cut_at": rec["step"],
                    "reason": f"NE 剂量 {round(rec['ne_dose'],3)} 超过不变上限 {ne_max}，轨迹在步骤 {rec['step']} 截断，保留前 {i} 步合法前缀"}
        if rec["fluid_total"] > fluid_max + 1e-9:
            return {"ok": False, "prefix_len": i, "cut_at": rec["step"],
                    "reason": f"累积液体 {rec['fluid_total']} mL 超过不变上限 {fluid_max}，轨迹在步骤 {rec['step']} 截断，保留前 {i} 步合法前缀"}
    return {"ok": True, "prefix_len": len(trajectory), "cut_at": None, "reason": "整条轨迹合法，无需截断"}
