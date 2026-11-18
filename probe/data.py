"""Данные опыта: инструмент с четырьмя полями (два городских с ролями src и dst, два посторонних), названия и описания полей,
шаблоны сообщения и пулы однотокенных значений. Всё, что делится на train и test, задаёт отложенные наборы для проб."""
import copy, random

CITY_CANDIDATES = [
    "Paris", "London", "Berlin", "Madrid", "Rome", "Tokyo", "Moscow", "Vienna", "Prague", "Oslo",
    "Lima", "Cairo", "Dubai", "Delhi", "Seoul", "Sydney", "Boston", "Dallas", "Denver", "Miami",
    "Atlanta", "Chicago", "Houston", "Phoenix", "Seattle", "Toronto", "Vancouver", "Montreal",
    "Mumbai", "Beijing", "Shanghai", "Bangkok", "Jakarta", "Manila", "Lagos", "Nairobi", "Accra",
    "Dublin", "Lisbon", "Athens", "Warsaw", "Budapest", "Helsinki", "Stockholm", "Copenhagen",
    "Brussels", "Zurich", "Geneva", "Milan", "Munich", "Hamburg", "Istanbul", "Tehran", "Baghdad",
    "Karachi", "Dhaka", "Hanoi", "Taipei", "Osaka", "Kyoto", "Auckland", "Perth", "Santiago",
    "Bogota", "Caracas", "Quito", "Havana", "Lyon", "Nice", "Porto",
    "Kyiv", "Minsk", "Riga", "Vilnius", "Tallinn", "Sofia", "Belgrade", "Zagreb", "Bucharest",
    "Reykjavik", "Edinburgh", "Glasgow", "Cardiff", "Belfast", "Leeds", "Bristol", "Seville",
    "Valencia", "Bilbao", "Naples", "Turin", "Venice", "Florence", "Cologne", "Dresden",
    "Frankfurt", "Stuttgart", "Antwerp", "Amsterdam", "Bern", "Basel", "Krakow", "Gdansk",
    "Kazan", "Sochi", "Lviv", "Odesa", "Tbilisi", "Baku", "Yerevan", "Almaty", "Tashkent",
    "Kabul", "Kathmandu", "Colombo", "Doha", "Muscat", "Riyadh", "Amman", "Beirut", "Tunis",
    "Algiers", "Rabat", "Dakar", "Lusaka", "Harare", "Maputo", "Luanda", "Kampala", "Kigali",
]
NAMES_POOL = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack", "Kate",
    "Liam", "Mary", "Nick", "Olivia", "Peter", "Quinn", "Rachel", "Steve", "Tom", "Uma", "Victor",
    "Wendy", "Xavier", "Yuri", "Zoe", "Anna", "Boris", "Clara", "Dmitry", "Elena", "Felix", "George",
    "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Mike", "Nina", "Oscar", "Paula", "Robert", "Sara",
]
PERIODS = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "Spring", "Summer", "Autumn", "Winter",
]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown", "gray", "grey",
              "silver", "gold", "violet", "cyan", "beige", "maroon", "navy", "teal", "olive", "crimson", "scarlet",
              "indigo", "lime", "magenta", "azure", "ivory", "tan", "amber", "coral", "khaki", "lavender", "turquoise",
              "bronze", "ruby", "mint", "peach", "plum"]
ANIMALS = ["cat", "dog", "horse", "tiger", "lion", "wolf", "bear", "fox", "deer", "rabbit", "eagle", "shark", "whale",
               "zebra", "monkey", "camel", "goat", "sheep", "mouse", "snake", "frog", "duck", "swan", "crow", "owl", "bee",
               "ant", "crab", "seal", "hawk", "cow", "pig", "hen", "rat", "bat", "bull", "ox", "lamb", "lynx", "otter",
               "panda", "koala", "moose", "bison", "gorilla", "parrot", "falcon", "dolphin", "turtle", "squirrel",
               "hedgehog", "penguin", "elephant", "giraffe", "leopard", "cheetah", "rhino", "hippo", "donkey", "mule", "pony"]

N_SLOTS = 4
ROLES = ("src", "dst")
OTHER = {"src": "dst", "dst": "src"}
ROLE_WORD = {"src": "origin", "dst": "destination"}
REQ = "Book the trip"

# посторонние поля: ключ в вызове, подпись в сообщении, описание в схеме, тип значения
FILLERS = [
    dict(key="passenger", label="Passenger", desc="Name of the passenger", ty="name"),
    dict(key="month", label="Month", desc="Month or weekday of the trip", ty="period"),
    dict(key="pet", label="Pet", desc="Animal travelling with the passenger", ty="animal"),
    dict(key="bag_color", label="Bag color", desc="Color of the suitcase", ty="color"),
]

# названия городских полей по роли и split; при ручной правке держите в каждом split хотя бы по одной паре src/dst одной длины в токенах
NAMES = {
    "src": {"train": ["origin", "from_city", "start_city", "origin_city", "leave_from", "home_city", "starting_point",
                      "source_city", "startpoint", "departure", "pickup_city", "departure_city"],
            "test": ["start_location", "from_location", "start_point", "pickup", "depart_from", "from_place"]},
    "dst": {"train": ["destination", "arrival", "to_city", "end_city", "arrival_city", "go_to", "goal_city", "final_stop",
                      "target_city", "destination_city", "dropoff", "dropoff_city"],
            "test": ["end_location", "to_location", "end_point", "endpoint", "arrive_at", "to_place"]},
}
# описания городских полей в схеме
DESCS = {
    "src": {"train": ["City the trip starts from", "City where the journey begins", "Place the traveler departs from"],
            "test": ["Starting point of the trip", "Where the traveler sets off from"]},
    "dst": {"train": ["City the trip goes to", "City where the journey ends", "Place the traveler is heading to"],
            "test": ["Final stop of the trip", "Where the traveler ends up"]},
}
# шаблоны сообщения: prefix — маркер роли стоит перед городом, postfix — после (на токене города роль принципиально неизвестна);
# first — какой город упомянут первым (None: выбирается случайно)
TEMPLATES = [
    dict(id="from_to", pos="prefix", split="train", first="src"),
    dict(id="to_from", pos="prefix", split="train", first="dst"),
    dict(id="paren", pos="postfix", split="train", first=None),
    dict(id="depart_arrive", pos="prefix", split="test", first="src"),
    dict(id="start_end", pos="prefix", split="test", first="src"),
    dict(id="is_the", pos="postfix", split="test", first=None),
]
# из какого split (train или test) берётся каждый признак примера в каждой группе;
# postfix — только postfix-шаблоны (для контроля на утечку), conflict — описание поля взято у другой роли, swap — см. make_spec
OOD_GROUPS = ["ood_city", "ood_name", "ood_desc", "ood_tpl", "ood_all"]
_TR = dict(city="train", name="train", desc="train", tpl="train")
GROUP_SPLITS = {"iid": _TR, "swap": _TR, "postfix": _TR, "conflict": _TR, "ood_all": {k: "test" for k in _TR},
                **{f"ood_{k}": {**_TR, k: "test"} for k in _TR}}
FIT_GROUPS = ("iid", "postfix")     # группы, которые делятся на fit / val / test
CF_KINDS = ("marker", "city", "schema")
VAL_TYPES = {"city": CITY_CANDIDATES, "name": NAMES_POOL, "period": PERIODS, "color": COLORS, "animal": ANIMALS}


def build_pools(lm, log):
    """Пулы однотокенных значений по типам (токены не повторяются между типами) и длины названий полей в токенах."""
    vals, seen = {}, set()
    for ty, cand in VAL_TYPES.items():
        p = []
        for c in cand:
            ids = lm.enc(c)
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0]); p.append(c)
        random.shuffle(p)
        h = int(len(p) * 0.6)
        assert h >= 8 and len(p) - h >= 4, f"{ty}: слишком мало однотокенных значений ({len(p)})"
        vals[ty] = {"train": p[:h], "test": p[h:]}
        log(ty, ":", len(p), "однотокенных, train/test", h, "/", len(p) - h)
    keys = {k for r in NAMES.values() for s in r.values() for k in s}
    return dict(vals=vals, tlen={k: len(lm.enc(k)) for k in keys})


def make_spec(rng, pools, group, tpl_ids=None):
    """Один пример: порядок полей, названия и описания, города, посторонние значения, шаблон.
    swap: первое поле городское и заполняется городом другой роли; conflict: описание городского поля взято у другой роли;
    postfix: только postfix-шаблоны; tpl_ids ограничивает выбор шаблона."""
    sp = GROUP_SPLITS[group]
    roles = ["src", "dst", "f1", "f2"]
    rng.shuffle(roles)
    if group == "swap" and roles[0] not in ROLES:
        j = next(i for i, r in enumerate(roles) if r in ROLES)
        roles[0], roles[j] = roles[j], roles[0]
    fills = rng.sample(FILLERS, 2)
    tl = pools["tlen"]
    dst_keys = NAMES["dst"][sp["name"]]
    src_key = rng.choice([k for k in NAMES["src"][sp["name"]] if any(tl[d] == tl[k] for d in dst_keys)])
    dst_key = rng.choice([k for k in dst_keys if tl[k] == tl[src_key]])
    city = dict(zip(ROLES, rng.sample(pools["vals"]["city"][sp["city"]], 2)))
    tpls = [t for t in TEMPLATES if t["split"] == sp["tpl"] and (group != "postfix" or t["pos"] == "postfix")
            and (tpl_ids is None or t["id"] in tpl_ids)]
    tpl = rng.choice(tpls)
    desc_role = {r: OTHER[r] if group == "conflict" else r for r in ROLES}
    desc = {r: rng.choice(DESCS[desc_role[r]][sp["desc"]]) for r in ROLES}
    fl = dict(zip(["f1", "f2"], fills))
    fields = []
    for r in roles:
        if r in ROLES:
            fields.append(dict(role=r, key=src_key if r == "src" else dst_key, desc=desc[r], desc_role=desc_role[r], value=city[r]))
        else:
            f = fl[r]
            v = rng.choice(pools["vals"][f["ty"]]["train"] + pools["vals"][f["ty"]]["test"])
            fields.append(dict(role=r, key=f["key"], desc=f["desc"], value=v, label=f["label"]))
    for f in fields:
        f["written"] = f["value"]
    if group == "swap":
        fields[0]["written"] = city[OTHER[fields[0]["role"]]]
    return dict(group=group, tpl=tpl["id"], tpl_pos=tpl["pos"], first=tpl["first"] or rng.choice(ROLES), fields=fields,
                src_city=city["src"], dst_city=city["dst"], name_len=tl[src_key], **{f"{k}_split": v for k, v in sp.items()})


def counterfactual(sp, kind):
    """Пара к примеру той же длины в токенах. marker: в сообщении from и to меняются местами, города стоят на месте
    (только для from_to и to_from); city: города меняются местами, маркеры на месте; schema: у двух городских полей меняются
    местами названия и описания, сообщение то же. Во всех трёх в каждый городской слот пишется другой город, чем в исходном."""
    cf = copy.deepcopy(sp)
    if kind in ("marker", "city"):
        cf["src_city"], cf["dst_city"] = sp["dst_city"], sp["src_city"]
        if kind == "marker":
            cf["tpl"] = {"from_to": "to_from", "to_from": "from_to"}[sp["tpl"]]
            cf["first"] = OTHER[sp["first"]]
    else:
        a, b = [f for f in cf["fields"] if f["role"] in ROLES]
        a["key"], b["key"] = b["key"], a["key"]
        a["desc"], b["desc"] = b["desc"], a["desc"]
        for f in (a, b):
            f["role"] = OTHER[f["role"]]
            f["desc_role"] = f["role"]
    for f in cf["fields"]:
        if f["role"] in ROLES:
            f["value"] = f["written"] = cf[f"{f['role']}_city"]
    return cf


def make_dataset(cfg, pools, rng):
    """Список примеров по группам; в группах FIT_GROUPS каждому примеру добавлено поле fit_split (fit, val, test)."""
    specs = []
    sizes = [("iid", cfg.N_IID)] + [(g, cfg.N_OOD) for g in OOD_GROUPS] + \
            [("swap", cfg.N_SWAP), ("postfix", cfg.N_POSTFIX), ("conflict", cfg.N_CONFLICT)]
    for g, n in sizes:
        specs += [make_spec(rng, pools, g) for _ in range(n)]
    for g in FIT_GROUPS:
        part = [s for s in specs if s["group"] == g]
        rng.shuffle(part)
        a, b = int(len(part) * cfg.FIT_FRAC[0]), int(len(part) * (cfg.FIT_FRAC[0] + cfg.FIT_FRAC[1]))
        for i, s in enumerate(part):
            s["fit_split"] = "fit" if i < a else "val" if i < b else "test"
    for s in specs:
        s.setdefault("fit_split", "-")
    return specs
