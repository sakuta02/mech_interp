"""Данные опыта: пять инструментов по пять аргументов разной семантики, шаблоны сообщений пользователя
и пулы однотокенных значений по семантическим типам."""
import random

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
NAMES = [
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

N_ARGS = 5
TRANSITIONS = [(1, 3), (3, 5), (5, 3), (3, 1)]                    # (o, d): стиринг двигает значение аргумента o к значению аргумента d
SLOTS = sorted({k for tr in TRANSITIONS for k in tr})             # аргументы, чьи состояния нужны для векторов: 1, 3, 5

# семантический тип -> кандидаты; в пул попадают только однотокенные, токены не повторяются между типами
# (при совпадении значение остаётся у типа, который стоит в словаре раньше)
TYPE_CANDIDATES = {
    "city": CITY_CANDIDATES,
    "name": NAMES,
    "period": PERIODS,
    "color": ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown", "gray", "grey",
              "silver", "gold", "violet", "cyan", "beige", "maroon", "navy", "teal", "olive", "crimson", "scarlet",
              "indigo", "lime", "magenta", "azure", "ivory", "tan", "amber", "coral", "khaki", "lavender", "turquoise",
              "bronze", "ruby", "mint", "peach", "plum"],
    "animal": ["cat", "dog", "horse", "tiger", "lion", "wolf", "bear", "fox", "deer", "rabbit", "eagle", "shark", "whale",
               "zebra", "monkey", "camel", "goat", "sheep", "mouse", "snake", "frog", "duck", "swan", "crow", "owl", "bee",
               "ant", "crab", "seal", "hawk", "cow", "pig", "hen", "rat", "bat", "bull", "ox", "lamb", "lynx", "otter",
               "panda", "koala", "moose", "bison", "gorilla", "parrot", "falcon", "dolphin", "turtle", "squirrel",
               "hedgehog", "penguin", "elephant", "giraffe", "leopard", "cheetah", "rhino", "hippo", "donkey", "mule", "pony"],
    "food": ["apple", "banana", "grape", "lemon", "mango", "pear", "cherry", "melon", "olive", "bread", "rice", "cheese",
             "pizza", "pasta", "soup", "salad", "cake", "honey", "butter", "sugar", "salt", "pepper", "onion", "carrot",
             "potato", "tomato", "garlic", "bean", "corn", "nut", "egg", "fish", "beef", "pork", "ham", "cookie", "candy",
             "cream", "pie", "berry", "coconut", "papaya", "apricot", "kiwi", "fig", "pumpkin", "burger", "sandwich", "noodle"],
    "country": ["France", "Spain", "Italy", "Japan", "China", "Brazil", "Canada", "Egypt", "India", "Mexico", "Norway", "Sweden",
                "Poland", "Turkey", "Greece", "Cuba", "Peru", "Chile", "Kenya", "Ghana", "Iran", "Iraq", "Israel", "Ireland",
                "Portugal", "Austria", "Belgium", "Denmark", "Finland", "Germany", "Russia", "Ukraine", "Serbia", "Croatia",
                "Hungary", "Romania", "Bulgaria", "Argentina", "Colombia", "Vietnam", "Thailand", "Indonesia", "Pakistan",
                "Nigeria", "Morocco", "Algeria", "Libya", "Syria", "Jordan", "Nepal", "Cambodia", "Australia", "Iceland",
                "Estonia", "Latvia", "Lithuania", "Belarus", "Georgia", "Armenia", "Kazakhstan"],
    "language": ["English", "French", "German", "Spanish", "Russian", "Chinese", "Arabic", "Hindi", "Japanese", "Korean",
                 "Italian", "Portuguese", "Dutch", "Greek", "Turkish", "Polish", "Swedish", "Latin", "Hebrew", "Persian",
                 "Thai", "Czech", "Hungarian", "Finnish", "Danish", "Norwegian", "Ukrainian", "Vietnamese", "Swahili",
                 "Bengali", "Urdu", "Tamil", "Welsh", "Irish", "Serbian", "Romanian", "Bulgarian", "Catalan"],
    "body": ["hand", "foot", "head", "arm", "leg", "eye", "ear", "nose", "mouth", "tooth", "finger", "knee", "neck", "back",
             "chest", "heart", "brain", "lung", "skin", "bone", "hair", "lip", "tongue", "thumb", "shoulder", "elbow",
             "ankle", "wrist", "hip", "belly", "palm", "toe", "jaw", "cheek", "chin", "brow", "spine", "rib"],
    "building": ["house", "school", "church", "bank", "hotel", "hospital", "museum", "library", "shop", "store", "market",
                 "cinema", "theater", "station", "airport", "factory", "castle", "palace", "tower", "bridge", "temple",
                 "mosque", "prison", "farm", "barn", "hut", "cabin", "villa", "mall", "gym", "office", "garage", "stadium",
                 "pharmacy", "bakery", "cafe", "inn", "pub", "lab", "studio", "tent", "dome"],
    "tool": ["hammer", "saw", "drill", "knife", "axe", "wrench", "shovel", "rake", "ladder", "brush", "scissors", "needle",
             "pliers", "chisel", "spoon", "fork", "plate", "cup", "bowl", "pot", "pan", "bottle", "key", "lock", "rope",
             "chain", "nail", "screw", "bolt", "hook", "pin", "pen", "ruler", "broom", "bucket", "tape", "glue", "file"],
    "shape": ["circle", "square", "triangle", "cube", "sphere", "cone", "oval", "star", "heart", "line", "arc", "ring", "cross",
              "diamond", "spiral", "wave", "dot", "curve", "arrow", "cylinder", "pyramid", "hexagon", "crescent", "loop"],
    "subject": ["math", "physics", "biology", "chemistry", "history", "art", "music", "geography", "literature", "law",
                "medicine", "economics", "philosophy", "algebra", "geometry", "grammar", "science", "logic", "ethics",
                "poetry", "drama", "dance", "sociology", "astronomy", "geology", "calculus", "statistics", "theology"],
    "clothing": ["shirt", "dress", "coat", "hat", "shoe", "boot", "scarf", "glove", "sock", "jacket", "skirt", "belt", "tie",
                 "suit", "jeans", "cap", "vest", "robe", "sweater", "shorts", "pants", "sandal", "apron", "mask", "veil",
                 "cloak", "hood", "uniform", "mitten", "bag", "purse", "wallet", "badge", "hoodie", "blouse", "gown", "tux"],
    "material": ["wood", "steel", "glass", "stone", "iron", "paper", "silk", "wool", "copper", "brass", "plastic", "leather",
                 "cotton", "rubber", "clay", "marble", "concrete", "granite", "cement", "ceramic", "nylon", "linen", "velvet",
                 "fabric", "tin", "zinc", "lead", "nickel", "titanium", "aluminum", "carbon", "bamboo", "cardboard", "wax",
                 "sand", "brick", "foam", "chrome", "tile", "straw", "coal", "salt", "opal", "diamond"],
    "furniture": ["chair", "table", "desk", "bed", "sofa", "shelf", "lamp", "door", "window", "couch", "stool", "bench",
                  "closet", "cabinet", "mirror", "carpet", "rug", "curtain", "pillow", "blanket", "fridge", "oven", "stove",
                  "sink", "bath", "clock", "vase", "basket", "box", "crib", "wardrobe"],
    "plant": ["oak", "pine", "birch", "maple", "cedar", "palm", "willow", "elm", "ash", "fir", "bush", "grass", "weed", "vine",
              "tree", "leaf", "seed", "root", "bark", "stem", "moss", "fern", "ivy", "cactus", "rose", "lily", "tulip", "daisy",
              "orchid", "iris", "lotus", "poppy", "lilac", "peony", "clover", "thistle", "reed", "sage", "mint", "herb"],
    "vehicle": ["car", "bus", "train", "bike", "taxi", "boat", "ship", "plane", "truck", "tram", "metro", "van", "jeep", "yacht",
                "ferry", "rocket", "scooter", "tractor", "subway", "canoe", "kayak", "balloon", "wagon", "cart", "sled",
                "helicopter", "bicycle", "motorcycle", "limousine", "minibus", "cab", "raft"],
}


def _arg(key, label, desc, ty):
    return dict(key=key, label=label, desc=desc, type=ty)


# инструмент: имя, описание, просьба пользователя и пять аргументов (ключ в вызове, подпись в сообщении, описание в схеме, тип значения);
# типы аргументов одного инструмента различны, а на позициях 1, 3, 5 разных инструментов стоят 15 разных типов
TOOLS = [
    dict(id="book_trip", name="book_trip", desc="Book a trip.", request="Book the trip.", args=[
        _arg("passenger", "Passenger", "Name of the passenger", "name"),
        _arg("bag_color", "Bag color", "Color of the suitcase", "color"),
        _arg("destination", "Destination", "City the trip goes to", "city"),
        _arg("pet", "Pet", "Animal travelling with the passenger", "animal"),
        _arg("month", "Month", "Month or weekday of the trip", "period")]),
    dict(id="submit_form", name="submit_form", desc="Submit a registration form.", request="Submit the form.", args=[
        _arg("injury", "Injury", "Body part that was injured", "body"),
        _arg("garden_plant", "Garden plant", "Plant growing in the garden", "plant"),
        _arg("lunch", "Lunch", "Food eaten for lunch", "food"),
        _arg("transport", "Transport", "Vehicle used to get here", "vehicle"),
        _arg("language", "Language", "Language spoken at home", "language")]),
    dict(id="register_item", name="register_item", desc="Register an item in the catalog.", request="Register the item.", args=[
        _arg("venue", "Venue", "Building where the item is stored", "building"),
        _arg("furniture", "Furniture", "Piece of furniture holding the item", "furniture"),
        _arg("tool", "Tool", "Tool used to make the item", "tool"),
        _arg("material", "Material", "Material the item is made of", "material"),
        _arg("origin_country", "Origin country", "Country the item comes from", "country")]),
    dict(id="log_record", name="log_record", desc="Add a record to the log.", request="Add the record.", args=[
        _arg("outfit", "Outfit", "Piece of clothing worn", "clothing"),
        _arg("ride", "Ride", "Vehicle used for the trip", "vehicle"),
        _arg("topic", "Topic", "School subject discussed", "subject"),
        _arg("badge_color", "Badge color", "Color of the badge", "color"),
        _arg("symbol", "Symbol", "Shape drawn on the badge", "shape")]),
    dict(id="create_entry", name="create_entry", desc="Create a new entry.", request="Create the entry.", args=[
        _arg("paint", "Paint", "Color of the paint", "color"),
        _arg("seat", "Seat", "Piece of furniture used for sitting or working", "furniture"),
        _arg("mascot", "Mascot", "Animal drawn in the entry", "animal"),
        _arg("tree", "Tree", "Plant mentioned in the entry", "plant"),
        _arg("medium", "Medium", "Material the entry is written on", "material")]),
]

# шаблоны сообщения пользователя: стиль (kv — по строкам, inline — в одну строку) и порядок перечисления значений
TEMPLATES = [
    dict(id="kv_fwd",     style="kv",     rev=False, split="calib"),
    dict(id="kv_rev",     style="kv",     rev=True,  split="calib"),
    dict(id="inline_fwd", style="inline", rev=False, split="test"),
    dict(id="inline_rev", style="inline", rev=True,  split="test"),
]


def build_pools(lm, log):
    """Пулы однотокенных значений по типам; токены не повторяются между типами, calib и test не пересекаются."""
    pool, seen = {}, set()
    for ty, cand in TYPE_CANDIDATES.items():
        p = []
        for c in cand:
            ids = lm.enc(c)
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0]); p.append(c)
        random.shuffle(p)
        h = len(p) // 2
        assert h >= 4, f"{ty}: слишком мало однотокенных значений ({len(p)}), добавьте кандидатов выше"
        pool[ty] = {"calib": p[:h], "test": p[h:2 * h]}
        log(ty, ":", len(p), "однотокенных, калибровка/тест по", h)
    return pool


def sample_vals(t, pool, split, rng=random):
    """Значения пяти аргументов инструмента: {номер аргумента 1..5: значение из пула его типа}."""
    return {k: rng.choice(pool[a["type"]][split]) for k, a in enumerate(t["args"], 1)}


assert all(len(t["args"]) == N_ARGS and len({a["type"] for a in t["args"]}) == N_ARGS and
           len({a["key"] for a in t["args"]}) == N_ARGS for t in TOOLS), "в каждом инструменте 5 аргументов разных типов"
assert all(a["type"] in TYPE_CANDIDATES for t in TOOLS for a in t["args"])
