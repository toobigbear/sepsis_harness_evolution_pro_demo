"""M13 扩展多专科 Agent 编排（SCALEMAS 消息板协调 + 掉线容错）。

Agent 均为脚本化规则引擎（数据来自模拟器与知识库），LLM 仅可选润色文本。
包含：微生物、检验、影像、沟通要点辅助（医生审阅后使用）、床旁护士。
"""
from .message_board import MessageBoard


class AgentOrchestrator:
    def __init__(self, board: MessageBoard, kb):
        self.board = board
        self.kb = kb
        self.agents = {
            "微生物Agent": {"role": "培养与药敏解读（模拟）", "offline": False},
            "检验Agent": {"role": "乳酸/肾功/血气趋势解读（模拟）", "offline": False},
            "影像Agent": {"role": "感染灶影像线索（模拟）", "offline": False},
            "沟通要点辅助": {"role": "病情沟通要点草稿（须医生审阅）", "offline": False},
            "床旁护士Agent": {"role": "护理要点与观察提醒（模拟）", "offline": False},
        }

    def _post(self, name, bed, content, refs=None):
        if self.agents[name]["offline"]:
            return {"queued": True, "note": f"{name} 掉线，消息已暂存（容错演示）"}
        self.board.publish(name, bed, "observation", content, refs=refs)
        return {"queued": False}

    def round_consult(self, beds, llm_polish=None):
        """对若干床做一次联合会诊发帖。"""
        out = []
        for b in beds:
            st = b["state"]
            bid = b["bed_id"]
            if st["lactate"] >= 3.0:
                out.append(self._post("检验Agent", bid,
                    f"乳酸 {st['lactate']} mmol/L 偏高，建议 2-4h 复查动态趋势；结合 ScvO2 评估灌注。",
                    refs=["检验Agent"]))
            if st["urine"] < 20:
                out.append(self._post("检验Agent", bid,
                    f"尿量 {st['urine']} mL/h 偏少，警惕 AKI：监测肌酐、必要时床旁超声评估容量。",
                    refs=["检验Agent"]))
            if st["map"] < 60:
                out.append(self._post("床旁护士Agent", bid,
                    f"MAP {st['map']} mmHg 低于目标 65：核对血管活性药通路/浓度，避免间断中断。",
                    refs=["床旁护士Agent"]))
        # 微生物与影像每次发帖（模拟数据源）
        out.append(self._post("微生物Agent", beds[0]["bed_id"] if beds else "-",
                              "血培养（模拟）：革兰阴性杆菌阳性，待药敏结果；经验覆盖建议参照本院耐药谱。", refs=["微生物Agent"]))
        out.append(self._post("影像Agent", beds[0]["bed_id"] if beds else "-",
                              "床旁超声（模拟）：下腔静脉变异度小、心室充盈可，提示容量反应性有限。", refs=["影像Agent"]))
        return out

    def communication_draft(self, st, llm_enabled=False, llm_fn=None):
        """沟通要点辅助：生成病情沟通要点草稿（LLM 可选润色；须医生审阅）。"""
        draft = (f"【沟通要点草稿｜须医生审阅后使用】\n"
                 f"1. 目前诊断：脓毒性休克（模拟），SOFA {st['sofa']} 分，乳酸 {st['lactate']} mmol/L。\n"
                 f"2. 当前治疗：去甲肾上腺素 {st['ne_dose']} μg/kg/min 维持灌注，液体策略已个体化。\n"
                 f"3. 病情危重程度与主要风险已向家属说明（器官支持需求、病情变化可能）。\n"
                 f"4. 下一步：动态评估灌注指标，必要时调整器官支持策略。")
        if llm_enabled and llm_fn:
            r = llm_fn(f"请将以下病情沟通要点润色为更通俗、更有同理心的版本（不改医学事实）：\n{draft}")
            if r.get("ok"):
                draft = r["text"]
        self.board.publish("沟通要点辅助", "-", "communication", draft, refs=["沟通要点辅助"])
        return draft

    def set_offline(self, name, offline=True):
        if name in self.agents:
            self.agents[name]["offline"] = offline

    def state(self):
        return {k: v for k, v in self.agents.items()}
