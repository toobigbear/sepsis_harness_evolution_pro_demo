"""M6 共享消息板（SCALEMAS 思想）：多床/多 Agent 发布、搜索、引用热度排序。

持久化到 data/message_board.jsonl；支持 Agent 掉线容错（offline 标记 + 消息暂存提示）。
"""
import json
import time
from pathlib import Path


class MessageBoard:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.msgs = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        self.msgs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    def publish(self, agent, bed, kind, content, refs=None):
        m = {"ts": round(time.time(), 1), "agent": agent, "bed": bed, "kind": kind,
             "content": content, "refs": refs or []}
        self.msgs.append(m)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
        return m

    def search(self, keyword):
        kw = keyword.lower()
        return [m for m in self.msgs if kw in m["content"].lower() or kw in m["agent"].lower()]

    def heat_sorted(self):
        """引用热度：被 refs 提及次数降序。"""
        cite = {}
        for m in self.msgs:
            for r in m.get("refs", []):
                cite[r] = cite.get(r, 0) + 1
        for m in self.msgs:
            m["heat"] = cite.get(m["agent"], 0) + len(m.get("refs", []))
        return sorted(self.msgs, key=lambda m: (-m["heat"], -m["ts"]))

    def recent(self, n=30):
        return self.msgs[-n:]
