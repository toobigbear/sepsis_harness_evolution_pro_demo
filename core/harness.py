"""Harness 调度内核（递归优化的外层框架，冻结模型权重）。

职责：组装决策解释提示、调度 M2-M7 组件、维护可变异参数。
Harness 进化的只是本文件的参数（见 config/harness_registry.json 的 mutable 字段），
模型权重（MAB/模拟器）永不改变。
"""
import json
import time
import copy
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

DEFAULT_PARAMS = {
    "prompt_interpret": {
        "template_text": "你是 ICU 重症临床助理。请用{style}解释以下算法决策："
                         "推荐动作 {action}（{branch}，不确定性 {uncertainty}）。"
                         "要求：1) 严禁修改、质疑或重新计算任何数值；2) 引用检索到的指南证据；"
                         "3) 强调医生拥有最终决策权；4) 结尾必须提示【仿真演示，非临床建议】。",
        "style": "简洁临床口吻",
    },
    "review_threshold": {"threshold": 60},
    "retrieval_topk": {"top_k": 2},
    "ts_sample_count": {"n_mc": 2000},
    "replay_ratio": {"ratio": 0.2},
}


class Harness:
    def __init__(self):
        self.registry = json.loads((CONFIG_DIR / "harness_registry.json").read_text(encoding="utf-8"))
        self.params = copy.deepcopy(DEFAULT_PARAMS)
        self.generation = 0
        self.created_at = round(time.time(), 1)
        self.metrics = {"decision_agreement_rate": None, "cases_evaluated": 0}

    def mutable_fields(self):
        return {c["id"]: c["mutable"] for c in self.registry["components"] if c["mutable"]}

    def build_interpret_prompt(self, decision, style=None):
        t = self.params["prompt_interpret"]
        return t["template_text"].format(style=style or t["style"], **{k: decision.get(k, "") for k in
                ("action", "branch", "uncertainty")})

    def state_dict(self):
        return {
            "generation": self.generation,
            "params": self.params,
            "mutable_fields": self.mutable_fields(),
            "metrics": self.metrics,
            "created_at": self.created_at,
        }

    def apply_candidate(self, candidate_params):
        """应用进化候选（已被 evolver 做不变量守卫检查过）。"""
        self.params = copy.deepcopy(candidate_params)
        self.generation += 1
