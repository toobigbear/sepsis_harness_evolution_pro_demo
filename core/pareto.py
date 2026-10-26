"""M10 多目标 Pareto 决策（采样式多目标评估，纯 numpy）。

对每个候选动作：用 HemoSim 批量 rollout 估计三维目标
（28天生存概率↑、乳酸清除率↑、液体安全↑），取经验 Pareto 前沿；
医生权重滑块在前沿上做标量化选择。
"""
import numpy as np


def evaluate_candidates(st0, action_ids, rng, sim_fn, rollouts=8):
    """sim_fn(st0, action, rng) -> outcomes dict。返回 {action: {mean, std, objs}}"""
    out = {}
    for a in action_ids:
        objs = []
        for _ in range(rollouts):
            o = sim_fn(st0, a, rng)
            objs.append([o["survival"], o["lactate_clearance"], o["fluid_safety"]])
        arr = np.array(objs)
        out[a] = {"mean": arr.mean(axis=0).round(3).tolist(), "std": arr.std(axis=0).round(3).tolist()}
    return out


def pareto_rank(points):
    """非支配排序（最大化问题）。points: dict{action: [o1,o2,o3]} → {action: rank}，rank 0 = 前沿。"""
    ids = list(points.keys())
    P = np.array([points[i] for i in ids])
    n = len(ids)
    ranks = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(P[j] >= P[i]) and np.any(P[j] > P[i]):
                ranks[i] += 1          # 被 j 支配一次（演示级计秩：被支配次数）
    # 归一：被支配次数 0 → rank 1（第一前沿）
    return {ids[i]: int(ranks[i]) for i in range(n)}


def scalar_choice(points, ranks, weights, objective_dirs=None):
    """医生权重标量化：只在第一前沿（被支配次数=0）中选择加权得分最高者。"""
    front = [a for a, r in ranks.items() if r == 0] or list(points.keys())
    w = np.array(weights, dtype=float)
    best, best_score = None, -1e9
    for a in front:
        s = float(np.dot(w, np.array(points[a])))
        if s > best_score:
            best, best_score = a, s
    return best, round(best_score, 3)
