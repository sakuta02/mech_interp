"""Сборка промптов опыта: сообщение пользователя с пятью значениями, схема инструмента в чат-шаблоне,
вызов с подставленными значениями и позиции токенов перед значениями аргументов."""
from data import N_ARGS, TOOLS, TEMPLATES
# формат вызова инструмента, которым отвечает модель; при другой модели поправить по выводу probe_format
CALL_START = '<function name="{name}"><param name="{key}">'
PARAM_SEP = '</param><param name="{key}">'


def render_user(t, tpl, vals):
    """Сообщение пользователя: просьба и пять значений с подписями в порядке аргументов (fwd) или обратном (rev)."""
    items = [(a["label"], vals[k]) for k, a in enumerate(t["args"], 1)]
    if tpl["rev"]:
        items = items[::-1]
    if tpl["style"] == "kv":
        return t["request"] + "\n" + "\n".join(f"{l}: {v}" for l, v in items)
    return t["request"] + " " + ", ".join(f"{l}: {v}" for l, v in items) + "."


def prompt_ids(lm, t, tpl, vals):
    """Токены промпта: чат-шаблон со схемой инструмента и сообщением пользователя."""
    props = {a["key"]: {"type": "string", "description": a["desc"]} for a in t["args"]}
    schema = [{"type": "function", "function": {"name": t["name"], "description": t["desc"],
               "parameters": {"type": "object", "properties": props, "required": list(props)}}}]
    return lm.enc(lm.tok.apply_chat_template([{"role": "user", "content": render_user(t, tpl, vals)}], tools=schema,
                                             add_generation_prompt=True, tokenize=False, enable_thinking=False))


def call_ids(lm, t, tpl, vals, upto=None):
    """Вызов с верными значениями (teacher forcing) и pos[k] — позиция токена перед значением аргумента k (1..5).
    upto=k обрезает последовательность на pos[k] включительно: значения аргументов k и далее не подаются."""
    ids = prompt_ids(lm, t, tpl, vals) + lm.enc(CALL_START.format(name=t["name"], key=t["args"][0]["key"]))
    pos = {}
    for k in range(1, N_ARGS + 1):
        pos[k] = len(ids) - 1
        if upto == k:
            return ids, pos
        ids += lm.enc(vals[k])
        if k < N_ARGS:
            ids += lm.enc(PARAM_SEP.format(key=t["args"][k]["key"]))
    return ids, pos


def probe_format(lm, log):
    """Печатает, что модель выдаёт без вмешательства, чтобы сверить формат вызова."""
    vals = dict(zip(range(1, N_ARGS + 1), ["Alice", "red", "Paris", "cat", "May"]))
    x = lm.tensor(prompt_ids(lm, TOOLS[0], TEMPLATES[0], vals))
    text = lm.tok.decode(lm.generate(x, 60)[0][x.shape[1]:])
    log("формат вызова:", repr(text))
    return text
