"""M5 Experience Replay 三层经验回灌（RSI 思想）。

- 病例级：历史病例特征+动作+结局摘要
- 轨迹级：关键生理轨迹片段
- 策略级：可复用的决策规则文字（由审计中高频模式归纳，演示级规则）

注入比例由 Harness 的 replay_ratio 控制（mutable 参数）。
"""
import json
from pathlib import Path


class ExperienceReplay:
    def __init__(self, path=None, capacity=200):
        self.cases, self.trajectories, self.policies = [], [], []
        self.capacity = capacity
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.cases = data.get("cases", [])
                self.trajectories = data.get("trajectories", [])
                self.policies = data.get("policies", [])
            except (json.JSONDecodeError, OSError):
                pass

    def add_case(self, case_record):
        self.cases.append(case_record)
        self.cases = self.cases[-self.capacity:]
        self._auto_policy(case_record)

    def add_trajectory(self, traj_record):
        self.trajectories.append(traj_record)
        self.trajectories = self.trajectories[-self.capacity:]

    def _auto_policy(self, rec):
        """演示级策略归纳：高奖励病例的动作沉淀为策略文字。"""
        if rec.get("reward", 0) >= 0.6:
            txt = f"病例 {rec.get('case_id','?')}：在 {rec.get('context_note','该情境')} 下动作 {rec.get('action')} 获得高奖励 {rec['reward']}。"
            if txt not in self.policies:
                self.policies.append(txt)
                self.policies = self.policies[-50:]

    def inject(self, ratio=0.2):
        """按 ratio 截取各层经验返回（供 Harness 提示组装）。"""
        n_c = max(1, int(len(self.cases) * ratio)) if self.cases else 0
        n_t = max(1, int(len(self.trajectories) * ratio)) if self.trajectories else 0
        n_p = max(1, int(len(self.policies) * ratio)) if self.policies else 0
        return {"case_level": self.cases[-n_c:], "trajectory_level": self.trajectories[-n_t:],
                "policy_level": self.policies[-n_p:]}

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(
                {"cases": self.cases, "trajectories": self.trajectories, "policies": self.policies},
                ensure_ascii=False, indent=1), encoding="utf-8")
