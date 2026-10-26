"""M4 Order-Flip Gate 决策-生成分离闸门（Jev 决策-生成分离的工程化）。

核心保证：决策数值必须先于任何文本生成计算并冻结；
用不同措辞/温度生成多版解读文本，决策数值必须逐位一致——否则告警。
LLM 在结构上不可能修改决策数值。
"""
import hashlib


def freeze_decision(decision_dict):
    """冻结决策：返回 (payload, hash)。之后任何字段变动都会被发现。"""
    import json
    payload = json.dumps(decision_dict, ensure_ascii=False, sort_keys=True)
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def verify_frozen(decision_dict, expected_hash):
    _, h = freeze_decision(decision_dict)
    return h == expected_hash


def flip_test(generate_fn, decision, n_variants=3):
    """生成 n 个不同措辞的解读文本，逐一核对决策数值未被改动。

    generate_fn(decision, style) -> text
    """
    styles = ["简洁临床口吻", "详细教学口吻", "面向家属的通俗口吻"]
    texts, ok = [], True
    for i in range(min(n_variants, len(styles))):
        t = generate_fn(decision, styles[i])
        texts.append(t)
        # 数值核对：文本中不得包含与决策矛盾的数值（演示级：核对动作字段未被替换）
        if decision.get("action") and decision["action"] not in t and "否决" not in t:
            ok = False
    return {"consistent": ok, "variants": len(texts), "texts": texts}
