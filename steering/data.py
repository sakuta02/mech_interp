"""Данные опыта: списки городов, имён и периодов, инструменты с ключами аргументов, шаблоны фраз пользователя
и построение пулов однотокенных значений.
"""
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

DOM = {
    "flight":  dict(vdesc="City name", extra=" The trip is on 2026-03-14.",
                    distract="{X} is also on my list, but it is not part of this trip."),
    "message": dict(vdesc="Person name", extra="",
                    distract="{X} is also in my contacts, but is not part of this message."),
    "period":  dict(vdesc="Month or weekday", extra="",
                    distract="{X} is also free, but it is not part of this plan."),
}

# пара ключей -> инструменты в двух порядках (od: первая роль первым, do: вторая роль первым)
FLIGHT_PAIRS = [
    ("book_flight",    "Book a flight ticket.",                "origin",     "destination", "calib"),
    ("reserve_flight", "Reserve a flight between two cities.", "departure",  "arrival",     "calib"),
    ("buy_ticket",     "Purchase an air ticket.",              "source",     "target",      "calib"),
    ("order_ticket",   "Order a plane ticket for a trip.",     "start_city", "end_city",    "calib"),
    ("search_flights", "Search and book available flights.",   "from",       "to",          "test"),
    ("plan_trip",      "Plan a trip and book the flight.",     "leave_from", "arrive_in",   "test"),
]
MSG_PAIRS = [
    ("send_message", "Send a message to a person.",     "sender",      "recipient", "calib"),
    ("write_letter", "Write a letter to a person.",     "from_person", "to_person", "calib"),
]
PER_PAIRS = [
    ("plan_vacation", "Plan a vacation for a period.",     "start",    "end",       "calib"),
    ("book_stay",     "Book a hotel stay for a period.",   "check_in", "check_out", "calib"),
]

LAYOUTS = ["none", "date_mid", "date_first"]      # варианты вызова: где в нём стоит дата


def mk_tools(domain, pairs, layouts):
    """Инструменты для пар ключей в обоих порядках и заданных вариантах вызова."""
    return [dict(id=f"{n}|{o}|{lay}", name=n, desc=ds, ko=ko, kd=kd, order=o, layout=lay, split=sp, domain=domain)
            for (n, ds, ko, kd, sp) in pairs for o in ("od", "do") for lay in layouts]


TOOLS = (mk_tools("flight", FLIGHT_PAIRS, LAYOUTS)
         + mk_tools("message", MSG_PAIRS, ["none"]) + mk_tools("period", PER_PAIRS, ["none"]))

_T = [
    ("flight", "prep_fwd",  "prep", "calib", "Book a flight from {O} to {D}."),
    ("flight", "prep_rev",  "prep", "calib", "I need a flight to {D} from {O}, please."),
    ("flight", "kv_fwd",    "kv",   "calib", "Departure: {O}\nArrival: {D}\nBook the flight."),
    ("flight", "kv_rev",    "kv",   "calib", "Arrival: {D}\nDeparture: {O}\nBook the flight."),
    ("flight", "kv2_fwd",   "kv",   "calib", "Origin: {O}, Destination: {D}. Book a flight."),
    ("flight", "kv2_rev",   "kv",   "calib", "Destination: {D}, Origin: {O}. Book a flight."),
    ("flight", "kv3_fwd",   "kv",   "test",  "Source: {O} | Target: {D} | Book a flight."),
    ("flight", "kv3_rev",   "kv",   "test",  "Target: {D} | Source: {O} | Book a flight."),
    ("flight", "nar_fwd",   "narr", "test",  "Book a flight from Misha to Masha. Misha lives in {O}. Masha lives in {D}."),
    ("flight", "nar_rev",   "narr", "test",  "Book a flight from Misha to Masha. Masha lives in {D}. Misha lives in {O}."),
    ("flight", "nar2_fwd",  "narr", "test",  "Misha lives in {O}. Masha lives in {D}. Book a flight for Misha to visit Masha."),
    ("flight", "nar2_rev",  "narr", "test",  "Masha lives in {D}. Misha lives in {O}. Book a flight for Misha to visit Masha."),
    ("message", "m_prep_fwd", "prep", "calib", "Send a message from {O} to {D}."),
    ("message", "m_kv_fwd",   "kv",   "calib", "Sender: {O}\nRecipient: {D}\nSend the message."),
    ("message", "m_prep_rev", "prep", "test",  "I need a message to {D} from {O}, please."),
    ("message", "m_kv_rev",   "kv",   "test",  "Recipient: {D}\nSender: {O}\nSend the message."),
    ("period", "p_prep_fwd",  "prep", "calib", "Plan a vacation from {O} to {D}."),
    ("period", "p_kv_fwd",    "kv",   "calib", "Start: {O}\nEnd: {D}\nPlan the vacation."),
    ("period", "p_prep_rev",  "prep", "test",  "I need a vacation that ends in {D} and starts in {O}, please."),
    ("period", "p_kv_rev",    "kv",   "test",  "End: {D}\nStart: {O}\nPlan the vacation."),
]
TEMPLATES = [dict(domain=dm, id=i, type=ty, split=sp, text=tx, o_first=tx.index("{O}") < tx.index("{D}"))
             for dm, i, ty, sp, tx in _T]


def build_pools(lm, cfg, log):
    """Пулы однотокенных значений для калибровки и теста по типам задач; наборы для вектора и для теста не пересекаются."""
    pool = {}
    for dom, cand in [("flight", CITY_CANDIDATES), ("message", NAMES), ("period", PERIODS)]:
        p = [c for c in cand if len(lm.enc(c)) == 1]
        random.shuffle(p)
        if dom == "flight":
            p = p[:cfg.N_CITIES]
        h = len(p) // 2
        assert h >= 4, f"{dom}: слишком мало однотокенных значений ({len(p)}), добавьте кандидатов выше"
        pool[dom] = {"calib": p[:h], "test": p[h:2 * h]}
        log(dom, ":", len(p), "однотокенных, калибровка/тест по", h)
    return pool
