"""Сборка промптов: схема инструмента в чат-шаблоне, сообщение пользователя, вызов с подставленными значениями
и позиции токенов перед значениями слотов.
"""
from data import DOM, TOOLS, TEMPLATES

# формат вызова инструмента, которым отвечает модель; при другой модели поправить по выводу probe_format
CALL_START = '<function name="{name}"><param name="{key}">'
PARAM_SEP = '</param><param name="{key}">'
DATE_VAL = "2026-03-14"


def arg_seq(t):
    """Аргументы вызова по порядку: (роль, ключ); роль o, d или x (дата)."""
    (r1, k1), (r2, k2) = [("o", t["ko"]), ("d", t["kd"])][::1 if t["order"] == "od" else -1]
    return {"none": [(r1, k1), (r2, k2)], "date_mid": [(r1, k1), ("x", "date"), (r2, k2)]}.get(
        t["layout"], [("x", "date"), (r1, k1), (r2, k2)])


def prompt_ids(lm, t, tpl, O, D, X):
    """Токены промпта: чат-шаблон со схемой инструмента и сообщением пользователя."""
    dm = DOM[t["domain"]]
    props = {k: {"type": "string", "description": "Date of the trip" if r == "x" else dm["vdesc"]}
             for r, k in arg_seq(t)}
    schema = [{"type": "function", "function": {"name": t["name"], "description": t["desc"],
               "parameters": {"type": "object", "properties": props, "required": list(props)}}}]
    content = tpl["text"].format(O=O, D=D) + dm["extra"] + " " + dm["distract"].format(X=X)
    return lm.enc(lm.tok.apply_chat_template([{"role": "user", "content": content}], tools=schema,
                                             add_generation_prompt=True, tokenize=False, enable_thinking=False))


def tf_ids(lm, t, tpl, O, D, X):
    """Полный вызов с верными значениями (teacher forcing) и позиции токенов перед значениями слотов."""
    val, seq = {"o": O, "d": D, "x": DATE_VAL}, arg_seq(t)
    ids = prompt_ids(lm, t, tpl, O, D, X) + lm.enc(CALL_START.format(name=t["name"], key=seq[0][1]))
    pos = {}
    for i, (r, _) in enumerate(seq):
        if r != "x":
            pos[r] = len(ids) - 1
        ids += lm.enc(val[r])
        if i + 1 < len(seq):
            ids += lm.enc(PARAM_SEP.format(key=seq[i + 1][1]))
    return lm.tensor(ids), pos


def prefix(lm, t, tpl, O, D, X, slot):
    """Контекст до значения выбранного слота; первое значение в вызове дописывает сама модель (greedy)."""
    seq = arg_seq(t)
    cities = [i for i, (r, _) in enumerate(seq) if r != "x"]
    target = cities[0] if slot == "first" else cities[1]
    ids = prompt_ids(lm, t, tpl, O, D, X) + lm.enc(CALL_START.format(name=t["name"], key=seq[0][1]))
    tok1 = None
    for i in range(target):
        if seq[i][0] == "x":
            ids += lm.enc(DATE_VAL)
        else:
            tok1 = lm.fwd_last(lm.tensor(ids))[0].argmax().item()
            ids.append(tok1)
        ids += lm.enc(PARAM_SEP.format(key=seq[i + 1][1]))
    return ids, seq[target][0], tok1


def probe_format(lm, log):
    """Печатает, что модель выдаёт без вмешательства, чтобы сверить формат вызова."""
    x = lm.tensor(prompt_ids(lm, TOOLS[0], TEMPLATES[0], "Paris", "Tokyo", "Rome"))
    text = lm.tok.decode(lm.generate(x, 40)[0][x.shape[1]:])
    log("формат вызова:", repr(text))
    return text
