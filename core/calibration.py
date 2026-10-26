"""M12 可信度校准：ECE / 决策曲线 DCA / 精确 Shapley 归因（纯 numpy）。

- ECE：校准队列（预生成模拟患者）上，模型预测成功率 vs 模拟结局频率的分箱偏差。
- DCA：不同决策阈值下的净收益 = TP/n − FP/n × pt/(1−pt)。
- Shapley：特征仅 9 个，对预测函数暴力枚举 2^9 子集求精确 Shapley 值（不是近似）。
"""
import numpy as np
from itertools import combinations


def ece_score(y_true, p_pred, n_bins=10):
    """Expected Calibration Error。y_true∈{0,1}，p_pred∈[0,1]。"""
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, details = 0.0, []
    for i in range(n_bins):
        mask = (p_pred > bins[i]) & (p_pred <= bins[i + 1]) if i > 0 else (p_pred >= bins[i]) & (p_pred <= bins[i + 1])
        if mask.sum() == 0:
            details.append({"bin": i, "n": 0, "pred": None, "obs": None, "gap": None})
            continue
        conf = p_pred[mask].mean()
        acc = y_true[mask].mean()
        gap = abs(conf - acc)
        ece += (mask.sum() / len(p_pred)) * gap
        details.append({"bin": i, "n": int(mask.sum()), "pred": round(float(conf), 3),
                        "obs": round(float(acc), 3), "gap": round(float(gap), 3)})
    return round(float(ece), 4), details


def dca_curve(y_true, p_pred, thresholds=None):
    """决策曲线分析。返回 [(pt, net_benefit_model, net_benefit_all, net_benefit_none)]。"""
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    n = len(y_true)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.8, 16)
    out = []
    for pt in thresholds:
        treat = p_pred >= pt
        tp = (treat & (y_true == 1)).sum()
        fp = (treat & (y_true == 0)).sum()
        nb_model = tp / n - fp / n * (pt / (1 - pt))
        nb_all = y_true.mean() - (1 - y_true.mean()) * (pt / (1 - pt))
        out.append({"pt": round(float(pt), 2), "model": round(float(nb_model), 4),
                    "treat_all": round(float(nb_all), 4), "treat_none": 0.0})
    return out


def shapley_exact(predict_fn, x, baseline, feature_names=None):
    """对 predict_fn（输入特征向量→标量概率）在 x 处求精确 Shapley 值。

    暴力枚举全部 2^d 子集；d=9 → 512 次求值，毫秒级。
    """
    x = np.asarray(x, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    d = len(x)
    phi = np.zeros(d)
    fact = [1.0]
    for i in range(1, d + 1):
        fact.append(fact[-1] * i)
    for j in range(d):
        others = [k for k in range(d) if k != j]
        for size in range(d):
            for subset in combinations(others, size):
                S = list(subset)
                w = fact[size] * fact[d - size - 1] / fact[d]
                x_with = baseline.copy(); x_without = baseline.copy()
                x_with[S] = x[S]; x_without[S] = x[S]
                x_with[j] = x[j]
                phi[j] += w * (predict_fn(x_with) - predict_fn(x_without))
    total = phi.sum()
    pred_full = predict_fn(x)
    if abs(total) > 1e-9:
        phi = phi * (pred_full - predict_fn(baseline)) / total     # 效率性归一
    names = feature_names or [f"f{i}" for i in range(d)]
    return [{"feature": names[i], "phi": round(float(phi[i]), 4)} for i in range(d)]
