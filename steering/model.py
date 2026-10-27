"""Загрузка модели и работа с активациями через TransformerLens: чтение resid_pre хуками и правка их при вмешательстве."""
import torch
from transformer_lens.model_bridge import TransformerBridge

RESID = "blocks.{}.hook_resid_pre".format          # вход слоя l (в HF это hidden_states[l])


def resolve_device(name):
    """Заменяет auto на cuda, если она доступна, иначе на cpu."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


class LM:
    """Модель, токенайзер и устройство. Модель грузится без enable_compatibility_mode(): он переписывает веса, а нужны исходные активации."""
    def __init__(self, cfg, log):
        """Загружает модель через TransformerBridge."""
        self.device = resolve_device(cfg.DEVICE)
        log(f"загрузка модели {cfg.MODEL} на {self.device} (при первом запуске скачивание весов может занять несколько минут)")
        self.model = TransformerBridge.boot_transformers(cfg.MODEL, device=self.device, dtype=torch.bfloat16,
                                                         trust_remote_code=True)
        self.tok, self.n_layers = self.model.tokenizer, self.model.cfg.n_layers
        log("модель загружена, слоёв:", self.n_layers)

    def enc(self, s):
        """Токены строки без специальных токенов."""
        return self.tok.encode(s, add_special_tokens=False)

    def tensor(self, ids):
        """Список токенов в тензор [1, T] на устройстве модели."""
        return torch.tensor([ids], device=self.device)

    def fwd_last(self, x, hooks=()):
        """Логиты последней позиции [B, V]; hooks = [(имя хука, функция)] действуют только на этот прогон."""
        return self.model.run_with_hooks(x, fwd_hooks=list(hooks), logits_to_keep=1, use_cache=False)[:, -1].float()

    def run_states(self, ids, pos, logits=True):
        """Читающим хуком берёт resid_pre всех слоёв в позициях pos; возвращает (логиты или None, {роль: [L, d]})."""
        idx = list(pos.values())
        store = torch.zeros(self.n_layers, len(idx), self.model.cfg.d_model, device=self.device)

        def read_resid(act, hook):
            """Пишет resid_pre слоя в заранее выделенный тензор."""
            store[hook.layer()] = act[0, idx].float()

        out = self.model.run_with_hooks(ids, return_type="logits" if logits else None,
                                        fwd_hooks=[(lambda name: name.endswith("hook_resid_pre"), read_resid)])
        return (out[0] if logits else None), {r: store[:, i] for i, r in enumerate(pos)}

    def generate(self, x, n):
        """Жадная генерация n токенов."""
        return self.model.generate(x, max_new_tokens=n, do_sample=False, verbose=False)


def steer(adds=None, repl=None):
    """Хук на resid_pre: прибавляет adds[b] к последнему токену строки b или подменяет его на repl."""
    def hook(act, hook):
        """Правит активацию последней позиции."""
        if adds is not None:
            act[:, -1] += adds.to(act.dtype)
        if repl is not None:
            act[:, -1] = repl.to(act.dtype)
        return act
    return hook
