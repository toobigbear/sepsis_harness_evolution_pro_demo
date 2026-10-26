"""M9 上下文贝叶斯多臂老虎机（Contextual Thompson Sampling，纯 numpy）。

模型：每臂一个贝叶斯逻辑回归  P(success | x, arm) = sigmoid(β_arm · x)，
β_arm ~ N(μ_arm, Σ_arm)，无信息先验 μ=0, Σ=10I。
后验更新采用 Laplace 近似（IRLS 求 MAP，Hessian 求协方差）。
保留 POC6 的高置信推荐 / 低置信随机分支逻辑；LLM 不得接触本模块输出之前的任何环节。
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class BayesianLogisticArm:
    def __init__(self, n_features, prior_mu=0.0, prior_sigma_scale=10.0):
        self.mu = np.full(n_features, float(prior_mu))
        self.prior_mu = np.full(n_features, float(prior_mu))
        self.sigma = np.eye(n_features) * float(prior_sigma_scale)
        self.n_updates = 0

    def sample_theta(self, rng):
        return rng.multivariate_normal(self.mu, self.sigma)

    def predict_proba(self, x, rng=None, n=1):
        """返回 P(success|x) 采样；rng 为 None 时用后验均值点估计。"""
        if rng is None:
            return float(sigmoid(self.mu @ x))
        thetas = rng.multivariate_normal(self.mu, self.sigma, size=n)
        return sigmoid(thetas @ x)

    def update(self, x, y, n_iter=8):
        """单观测 (x, y∈{0,1}) 的 Laplace 近似后验更新。

        注意：先验锚点必须是上一后验均值 mu_prev（而非原始 prior_mu），
        否则随更新次数增加后验会被持续拖回 0（在线 Laplace 的经典陷阱）。
        """
        mu_prev = self.mu.copy()
        mu = mu_prev.copy()
        sigma_inv = np.linalg.inv(self.sigma)
        for _ in range(n_iter):
            p = sigmoid(mu @ x)
            w = max(p * (1 - p), 1e-4)
            H = sigma_inv + w * np.outer(x, x)
            g = (y - p) * x + sigma_inv @ (mu_prev - mu)
            step = np.linalg.solve(H, g)
            mu = mu + step
            if np.linalg.norm(step) < 1e-6:
                break
        p = sigmoid(mu @ x)
        w = max(p * (1 - p), 1e-4)
        H = sigma_inv + w * np.outer(x, x)
        self.sigma = np.linalg.inv(H)
        self.mu = mu
        self.n_updates += 1

    def state_dict(self):
        return {"mu": self.mu.tolist(), "sigma_diag": np.diag(self.sigma).tolist(),
                "n_updates": self.n_updates}


class ContextualThompsonBandit:
    def __init__(self, arm_ids, n_features, prior_mu=0.0, prior_sigma_scale=10.0, n_mc=2000):
        self.arm_ids = list(arm_ids)
        self.n_mc = int(n_mc)
        self.arms = {a: BayesianLogisticArm(n_features, prior_mu, prior_sigma_scale) for a in self.arm_ids}
        self.total_updates = 0

    def _mc_matrix(self, x, rng, n=None):
        """每臂 n 次采样 → (n_arms, n) 概率矩阵 + 每臂均值概率。"""
        n = n or self.n_mc
        P = np.stack([self.arms[a].predict_proba(x, rng, n=n) for a in self.arm_ids])
        return P

    def arm_means(self, x, rng=None, n=None):
        P = self._mc_matrix(x, rng, n)
        return {a: float(P[i].mean()) for i, a in enumerate(self.arm_ids)}

    def uncertainty(self, x, rng, n_mc=None):
        """决策不确定性 = 100 × (1 − |2·P(最优臂) − 1|)。100 = 完全无法区分。"""
        P = self._mc_matrix(x, rng, n_mc)
        best = P.argmax(axis=0)
        counts = np.bincount(best, minlength=len(self.arm_ids)) / P.shape[1]
        p_best = float(counts.max())
        return round(100.0 * (1.0 - abs(2.0 * p_best - 1.0)), 1)

    def decide(self, x, rng, threshold):
        """返回 dict：branch / action / uncertainty / arm_means / P_best。"""
        P = self._mc_matrix(x, rng)
        means = {a: float(P[i].mean()) for i, a in enumerate(self.arm_ids)}
        best_counts = np.bincount(P.argmax(axis=0), minlength=len(self.arm_ids)) / P.shape[1]
        p_best = float(best_counts.max())
        uncertainty = round(100.0 * (1.0 - abs(2.0 * p_best - 1.0)), 1)
        if uncertainty < threshold:
            branch = "recommend"
            action = max(means, key=means.get)
        else:
            branch = "randomize"
            # Thompson 采样：每臂采一个 θ，取 sigmoid 最大者
            action = self.arm_ids[int(np.argmax([sigmoid(self.arms[a].sample_theta(rng) @ x) for a in self.arm_ids]))]
        return {"branch": branch, "action": action, "uncertainty": uncertainty,
                "arm_means": means, "p_best": round(p_best, 3)}

    def update(self, arm, x, reward, n_b=3, rng=None):
        """reward∈[0,1] → 视为 n_b 次同上下文 Bernoulli(reward) 观测的充分统计量。

        结局本身由连续指标合成，直接量化为单次二值会损失大量信息；
        以多次伯努利观测回灌是标准的多项/二项近似（Demo 级）。
        """
        rng = rng or np.random.default_rng()
        x = np.asarray(x, dtype=float)
        for _ in range(int(n_b)):
            self.arms[arm].update(x, int(rng.random() < reward))
        self.total_updates += 1

    def posterior_stats(self, x, rng):
        """各臂后验期望成功率 + 95% 区间（MC）。"""
        out = {}
        for a in self.arm_ids:
            s = self.arms[a].predict_proba(x, rng, n=max(self.n_mc, 2000))
            out[a] = {"mean": float(s.mean()), "lo": float(np.percentile(s, 2.5)),
                      "hi": float(np.percentile(s, 97.5)), "n": self.arms[a].n_updates}
        return out

    def coefficient_table(self, feature_labels):
        rows = []
        for a in self.arm_ids:
            d = self.arms[a]
            for i, lab in enumerate(feature_labels):
                rows.append({"arm": a, "feature": lab, "mu": round(float(d.mu[i]), 3),
                             "sd": round(float(np.sqrt(d.sigma[i, i])), 3)})
        return rows

    def state_dict(self):
        return {"arms": {a: self.arms[a].state_dict() for a in self.arm_ids},
                "total_updates": self.total_updates}
