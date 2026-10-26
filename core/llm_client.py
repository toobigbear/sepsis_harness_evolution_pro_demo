"""本地 LLM 客户端（LM-Studio，OpenAI 兼容）。只做解读/取证/文本润色，永不参与决策计算。

硬性隔离：本模块没有任何修改数值的接口；决策在调用本模块之前已完成并冻结（见 order_flip_gate）。
"""
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
LMSTUDIO_BASE = "http://127.0.0.1:1234/v1"
LLM_MODELS = ["qwen3-8b", "qwen/qwen3.5-9b"]
EMBED_MODEL_CANDIDATES = ["nomic-ai/text-embedding-nomic-embed-text-v1.5",
                          "yorkieoh10/text-embedding-nomic-embed-text-v1.5"]

_client = None


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=LMSTUDIO_BASE, api_key="lm-studio", timeout=60)
    return _client


def probe():
    """探测 LM-Studio 可用性。返回 {connected, models[], message}。"""
    try:
        models = [m.id for m in get_client().models.list()]
        return {"connected": True, "models": models, "message": f"已连接（{len(models)} 个模型）"}
    except Exception as e:
        return {"connected": False, "models": [],
                "message": f"未连接：LM-Studio 未启动或未开启 API 服务（{type(e).__name__}）"}


def interpret(prompt, model=None):
    """决策解读 / 文本生成。失败返回 {ok: False, message}，绝不抛异常。"""
    try:
        m = model or LLM_MODELS[0]
        r = get_client().chat.completions.create(
            model=m,
            messages=[{"role": "system", "content": "你是 ICU 重症临床助理，只做解释与沟通辅助，严禁给出诊断结论、治疗医嘱或修改任何算法数值。"},
                      {"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=600)
        return {"ok": True, "text": r.choices[0].message.content.strip(), "model": m}
    except Exception as e:
        return {"ok": False, "message": f"本地大模型调用失败：{type(e).__name__}。请确认 LM-Studio 已启动且模型已加载。", "model": model}


def embed_texts(texts, model=None):
    """批量 embedding。失败返回 None。"""
    for cand in ([model] if model else []) + EMBED_MODEL_CANDIDATES:
        try:
            r = get_client().embeddings.create(model=cand, input=texts)
            return [[float(v) for v in d.embedding] for d in r.data]
        except Exception:
            continue
    return None
