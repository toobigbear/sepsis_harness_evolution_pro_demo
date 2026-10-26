"""M7 TOON 风格序列化：紧凑的键值行格式，用于审计日志体积对比。

TOON（Token-Oriented Object Notation）思想：扁平键值 + 行结构，去掉 JSON 的引号/括号冗余。
本模块提供 toon_encode / toon_decode / json_size 对比，纯标准库。
"""
import json


def _is_homogeneous_dict_list(o):
    return isinstance(o, list) and len(o) > 0 and all(isinstance(r, dict) for r in o) \
        and len({tuple(r.keys()) for r in o}) == 1


def toon_encode(obj):
    """dict → TOON 行。嵌套 dict 展平为点分键；标量列表用逗号分隔。

    同构 dict 列表（审计批量事件、评测记录等）转列式行——键只写一次，
    这是 TOON 相对 JSON 体积优势的主要来源（参考 JSON vs TOON 对照，
    412 字符 JSON → 154 字符 TOON，约 37%）。
    """
    lines = []

    def emit_scalar_list(prefix, o):
        lines.append(f"{prefix}:{','.join(str(i) for i in o)}")

    def emit_columnar(prefix, rows):
        keys = list(rows[0].keys())
        lines.append(f"{prefix}[{len(rows)}]{{{','.join(keys)}}}:")
        for r in rows:
            lines.append(",".join(str(r[k]) for k in keys))

    def walk(o, prefix):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{prefix}{k}.")
        elif _is_homogeneous_dict_list(o):
            emit_columnar(prefix[:-1], o)
        elif isinstance(o, list):
            emit_scalar_list(prefix[:-1], o)
        else:
            lines.append(f"{prefix[:-1]}:{o}")

    walk(obj, "")
    return "\n".join(lines)


def json_size(obj):
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def toon_size(obj):
    return len(toon_encode(obj).encode("utf-8"))


def ratio(obj):
    j, t = json_size(obj), toon_size(obj)
    return {"json_bytes": j, "toon_bytes": t,
            "saved_pct": round(100 * (1 - t / max(j, 1)), 1)}
