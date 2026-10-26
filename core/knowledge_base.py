"""指南知识库 + numpy 向量检索（nomic-embed 可用时启用，否则关键词降级）。"""
import json
import numpy as np
from pathlib import Path
from . import llm_client

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class KnowledgeBase:
    def __init__(self):
        self.snippets = json.loads((CONFIG_DIR / "sepsis_knowledge.json").read_text(encoding="utf-8"))["snippets"]
        self.drug_kg = json.loads((CONFIG_DIR / "drug_kg.json").read_text(encoding="utf-8"))
        self._vectors = None

    def _ensure_vectors(self):
        if self._vectors is None:
            vecs = llm_client.embed_texts([s["text"] for s in self.snippets])
            self._vectors = np.array(vecs) if vecs else None
        return self._vectors

    def retrieve(self, query, top_k=2):
        vecs = self._ensure_vectors()
        if vecs is None:
            # 关键词降级
            kw = [w for w in query if w.strip()]
            scored = sorted(self.snippets, key=lambda s: sum(1 for w in str(query).split() if w in s["text"]), reverse=True)
            return {"mode": "keyword", "items": scored[:top_k]}
        q = llm_client.embed_texts([query])
        if not q:
            return {"mode": "keyword", "items": self.snippets[:top_k]}
        qv = np.array(q[0])
        sims = vecs @ qv / (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(qv) + 1e-9))
        idx = np.argsort(-sims)[:top_k]
        return {"mode": "embedding", "items": [self.snippets[i] for i in idx],
                "sims": [round(float(sims[i]), 3) for i in idx]}

    def t3_rules(self):
        """T3 知识图谱规则（仅展示，不进决策算法）。"""
        return {"triples_sample": self.drug_kg["triples"][:8], "rules": self.drug_kg["rules_t3"]}
