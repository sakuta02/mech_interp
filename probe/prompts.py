"""Сборка промптов: сообщение пользователя с маршрутом, схема инструмента в чат-шаблоне, вызов с подставленными значениями
(teacher forcing) и позиции токенов: слоты (токен перед значением поля) и последние токены городов в сообщении."""
from data import N_SLOTS, REQ, ROLE_WORD

TOOL_NAME, TOOL_DESC = "book_trip", "Book a trip."
# формат вызова инструмента, которым отвечает модель; при другой модели поправить по выводу probe_format
CALL_START = '<function name="{name}"><param name="{key}">'
PARAM_SEP = '</param><param name="{key}">'


def render_user(sp):
    """Сообщение пользователя и символьные интервалы городов {роль: (начало, конец)}."""
    S, D, t = (sp["src_city"], "src"), (sp["dst_city"], "dst"), sp["tpl"]
    A, B = (S, D) if sp["first"] == "src" else (D, S)
    parts = {"from_to": [(f"{REQ} from ", None), S, (" to ", None), D, (".", None)],
             "to_from": [(f"{REQ} to ", None), D, (" from ", None), S, (".", None)],
             "depart_arrive": [(f"{REQ}, departing ", None), S, (" and arriving at ", None), D, (".", None)],
             "start_end": [(f"{REQ}, starting in ", None), S, (" and ending in ", None), D, (".", None)],
             "paren": [(f"{REQ}: ", None), A, (f" ({ROLE_WORD[A[1]]}), ", None), B, (f" ({ROLE_WORD[B[1]]}).", None)],
             "is_the": [(f"{REQ}: ", None), A, (f" is the {ROLE_WORD[A[1]]} and ", None), B, (f" is the {ROLE_WORD[B[1]]}.", None)]}[t]
    parts.append((" " + " ".join(f"{f['label']}: {f['value']}." for f in sp["fields"] if "label" in f), None))
    text, spans = "", {}
    for s, role in parts:
        if role:
            spans[role] = (len(text), len(text) + len(s))
        text += s
    return text, spans


def prompt_ids(lm, sp):
    """Токены промпта (поля схемы идут в порядке слотов) и cpos: cpos[роль] — последний токен города в сообщении,
    cpos[роль + "_first"] — первый."""
    text, spans = render_user(sp)
    props = {f["key"]: {"type": "string", "description": f["desc"]} for f in sp["fields"]}
    schema = [{"type": "function", "function": {"name": TOOL_NAME, "description": TOOL_DESC,
               "parameters": {"type": "object", "properties": props, "required": list(props)}}}]
    s = lm.tok.apply_chat_template([{"role": "user", "content": text}], tools=schema, add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)
    enc = lm.tok(s, add_special_tokens=False, return_offsets_mapping=True)
    off = s.rindex(text)
    cpos = {}
    for r, (a, b) in spans.items():
        toks = [i for i, (x, y) in enumerate(enc["offset_mapping"]) if x < off + b and y > off + a]
        cpos[r], cpos[r + "_first"] = max(toks), min(toks)
    return enc["input_ids"], cpos


def call_ids(lm, sp, upto=None):
    """Промпт и вызов с записанными значениями; pos[k] — токен перед значением поля k (1..4), cpos — токены городов.
    upto=k обрезает последовательность на pos[k] включительно: значение поля k и дальше не подаются."""
    ids, cpos = prompt_ids(lm, sp)
    f = sp["fields"]
    ids = ids + lm.enc(CALL_START.format(name=TOOL_NAME, key=f[0]["key"]))
    pos = {}
    for k in range(1, N_SLOTS + 1):
        pos[k] = len(ids) - 1
        if upto == k:
            return ids, pos, cpos
        ids += lm.enc(f[k - 1]["written"])
        if k < N_SLOTS:
            ids += lm.enc(PARAM_SEP.format(key=f[k]["key"]))
    return ids, pos, cpos


def probe_format(lm, sp, log):
    """Печатает, что модель выдаёт без вмешательства, чтобы сверить формат вызова."""
    ids, _ = prompt_ids(lm, sp)
    x = lm.tensor(ids)
    text = lm.tok.decode(lm.generate(x, 60)[0][x.shape[1]:])
    log("формат вызова:", repr(text))
    return text
