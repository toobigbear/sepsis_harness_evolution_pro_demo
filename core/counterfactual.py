"""M11 因果反事实推演（数字孪生平行世界，纯 numpy）。

对同一患者状态分叉：现实分支（采用实际动作）与反事实分支（采用备选动作），
各运行 n 次随机 rollout 取均值曲线与分位数带，估计"平均处理效应"与个体化置信区间。

⚠️ 反事实来自模拟器内部——是"数字孪生内的因果推演"，不是真实世界因果证据。
"""
import numpy as np


def run_counterfactual(st0, real_action, cf_actions, rng, sim_fn, rollouts=20, metric_keys=("survival",)):
    """sim_fn(st0, action, rng) -> outcomes dict（含 trajectory）。

    返回 {branch_id: {label, curves, mean_metric, ci_lo, ci_hi}}，branch 0 = 现实。
    """
    branches = [(real_action, "现实（实际选择）")] + [(a, f"反事实：{a}") for a in cf_actions]
    result = {}
    for idx, (act, label) in enumerate(branches):
        metrics = {k: [] for k in metric_keys}
        map_curves, lac_curves = [], []
        for _ in range(rollouts):
            o = sim_fn(st0, act, rng)
            for k in metric_keys:
                metrics[k].append(o[k])
            map_curves.append([r["map"] for r in o["trajectory"]])
            lac_curves.append([r["lactate"] for r in o["trajectory"]])
        M = np.array([metrics[k] for k in metric_keys])
        curves_arr = lambda C: np.array(C)
        mc, lc = curves_arr(map_curves), curves_arr(lac_curves)
        result[str(idx)] = {
            "action": act, "label": label,
            "map_mean": mc.mean(axis=0).round(1).tolist(),
            "map_lo": np.percentile(mc, 10, axis=0).round(1).tolist(),
            "map_hi": np.percentile(mc, 90, axis=0).round(1).tolist(),
            "lac_mean": lc.mean(axis=0).round(2).tolist(),
            "lac_lo": np.percentile(lc, 10, axis=0).round(2).tolist(),
            "lac_hi": np.percentile(lc, 90, axis=0).round(2).tolist(),
            "metric_means": {k: round(float(M[i].mean()), 3) for i, k in enumerate(metric_keys)},
            "metric_ci": {k: [round(float(np.percentile(M[i], 10)), 3), round(float(np.percentile(M[i], 90)), 3)]
                          for i, k in enumerate(metric_keys)},
        }
    # 平均处理效应（以第一个反事实分支为对照）
    ate = {}
    for k in metric_keys:
        if len(metric_keys) and "1" in result:
            ate[k] = round(result["0"]["metric_means"][k] - result["1"]["metric_means"][k], 3)
    return {"branches": result, "ate_vs_first_cf": ate}
