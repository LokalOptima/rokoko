#!/usr/bin/env python3
"""Generate augmentation training data for underrepresented normalization classes.

Pipeline for each example:
  1. Programmatically generate raw_text (and optionally a spoken_form for
     patterns the normalizer does NOT handle)
  2. For normalizer-handled patterns: run normalize_cli on raw_text to get
     the canonical normalized form, then phonemize that with Misaki
  3. For model-learned patterns: phonemize spoken_form with Misaki directly
  4. Output: raw_text \t phonemes  (same format as g2p_train_v3.tsv)

IMPORTANT: Category A generators (dates, money, time, phone, numbers, etc.)
return (raw_text, None). The spoken form comes from the normalize_cli binary —
NEVER reimplement normalization logic in Python.

Category B generators (abbreviations, roman numerals, units, etc.) return
(raw_text, spoken_form) because the normalizer passes these through unchanged
and the model must learn them from augmentation.

Usage:
    python training/g2p/augment_data.py --output data/g2p_augment.tsv
    python training/g2p/augment_data.py --classes time money --output data/g2p_augment_time_money.tsv
    python training/g2p/augment_data.py --stats  # show planned counts per class
"""

import argparse
import calendar
import os
import random
import subprocess
import sys
import time

import num2words as _n2w

# ── Normalizer (calls the C++ binary — single source of truth) ──────────────

from normalizer import Normalizer


# ── Helpers (only for Category B generators that need spoken forms) ─────────

def year_to_words(y: int) -> str:
    """Year to spoken form via num2words. Only used by Category B generators."""
    s = _n2w.num2words(y, to="year")
    return s.replace("-", " ").replace(",", "").strip()


def n2w(n, **kwargs) -> str:
    """Number to words via num2words. Only used by Category B generators."""
    s = _n2w.num2words(n, **kwargs)
    s = s.replace("-", " ").replace(",", "")
    return " ".join(s.split())


def ordinal(n: int) -> str:
    return _n2w.num2words(n, to="ordinal").replace("-", " ")


def ordinal_suffix(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def address_number(n: int) -> str:
    if n < 100:
        return n2w(n)
    if n < 10000:
        hi, lo = divmod(n, 100)
        if lo == 0:
            return n2w(hi) + " hundred"
        return n2w(hi) + " " + n2w(lo)
    return n2w(n)


# ── Misaki G2P wrapper ─────────────────────────────────────────────────────

class MisakiG2P:
    def __init__(self):
        from misaki.en import G2P
        self.g2p = G2P()
        self.cache = {}

    def phonemize(self, text: str) -> str:
        if text in self.cache:
            return self.cache[text]
        try:
            ph, _ = self.g2p(text)
            self.cache[text] = ph or ""
            return ph or ""
        except Exception:
            self.cache[text] = ""
            return ""


# ── Generators ──────────────────────────────────────────────────────────────

def gen_cardinal(rng: random.Random) -> list[tuple[str, str | None]]:
    """Category A: Generate raw text with cardinal numbers. Normalizer handles expansion."""
    pairs = []

    cardinal_templates = [
        "The stadium holds {} spectators.",
        "She counted {} items on the shelf.",
        "The population reached {} last year.",
        "They produced {} units in a single day.",
        "There are {} students enrolled.",
        "He walked {} steps today.",
        "The company has {} employees.",
        "The book has {} pages.",
        "We need {} more volunteers.",
        "The warehouse stores {} boxes.",
        "About {} people attended the event.",
        "The city has {} residents.",
        "Only {} tickets remain.",
        "The project costs {} dollars.",
        "She has {} followers on social media.",
        "The report mentions {} cases.",
        "There were {} participants in the study.",
        "He scored {} points in the game.",
        "The factory produces {} items per day.",
        "Over {} applications were received.",
        "Approximately {} visitors came last month.",
        "The survey included {} respondents.",
        "They collected {} signatures.",
        "The database contains {} records.",
        "A total of {} votes were cast.",
    ]

    cardinal_nums = (
        list(range(0, 100)) +
        list(range(100, 1000, 10)) +
        list(range(1000, 10000, 50)) +
        list(range(10000, 100000, 500)) +
        list(range(100000, 1000001, 10000)) +
        [2000000, 5000000, 10000000, 50000000, 100000000]
    )

    for n in cardinal_nums:
        n_tmpls = 5 if n < 100 else (4 if n < 1000 else 3)
        for tmpl in rng.sample(cardinal_templates, min(n_tmpls, len(cardinal_templates))):
            pairs.append((tmpl.format(n), None))

    # Comma-separated numbers
    for n in rng.sample(range(1000, 10000000), 500):
        written = f"{n:,}"
        tmpl = rng.choice(cardinal_templates)
        pairs.append((tmpl.format(written), None))

    return pairs


def gen_ordinal(rng: random.Random) -> list[tuple[str, str | None]]:
    """Category A: Generate raw text with ordinals. Normalizer handles expansion."""
    pairs = []
    templates = [
        "the {}{} floor",
        "her {}{} birthday",
        "the {}{} anniversary",
        "finished {}{} in the race",
        "the {}{} president",
        "their {}{} attempt",
        "the {}{} episode",
        "his {}{} year of service",
        "the {}{} century",
        "on the {}{} day",
        "ranked {}{} overall",
        "the {}{} edition",
        "the {}{} largest city",
        "in {}{} place",
    ]
    for n in list(range(1, 200)) + list(range(200, 1100, 50)):
        suf = ordinal_suffix(n)
        n_tmpls = 4 if n <= 50 else 3
        for tmpl in rng.sample(templates, min(n_tmpls, len(templates))):
            raw = tmpl.format(n, suf)
            pairs.append((raw, None))
    return pairs


def gen_date(rng: random.Random) -> list[tuple[str, str | None]]:
    """Category A: Generate raw text with dates. Normalizer handles expansion.

    Formats generated: D/M/YYYY, YYYY-MM-DD, YYYY/M/D, "Month D, YYYY", "Mon D, YYYY"
    NOTE: No dot-separated dates (normalizer doesn't handle them — ambiguous with decimals).
    NOTE: No M/D/Y (ambiguous, non-standard). Slash dates use D/M/Y.
    """
    pairs = []

    sentence_templates = [
        "The deadline is {}.",
        "She was born on {}.",
        "The event takes place on {}.",
        "We moved here on {}.",
        "The conference starts {}.",
        "The report was filed on {}.",
        "He retired on {}.",
        "The meeting is scheduled for {}.",
        "Applications close on {}.",
        "The wedding is set for {}.",
        "The contract expires on {}.",
        "She arrived on {}.",
    ]

    for _ in range(1500):
        y = rng.randint(1000, 2099)
        m = rng.randint(1, 12)
        d = rng.randint(1, 28)
        month = calendar.month_name[m]
        month_abbr = calendar.month_abbr[m]

        # Textual: "March 9, 2026"
        tmpl = rng.choice(sentence_templates)
        pairs.append((tmpl.format(f"{month} {d}, {y}"), None))

        # Textual abbreviated: "Mar 9, 2026"
        tmpl = rng.choice(sentence_templates)
        pairs.append((tmpl.format(f"{month_abbr} {d}, {y}"), None))

        # D/M/YYYY: "9/3/2026"
        tmpl = rng.choice(sentence_templates)
        pairs.append((tmpl.format(f"{d}/{m}/{y}"), None))

        # ISO: "2026-03-09"
        tmpl = rng.choice(sentence_templates)
        pairs.append((tmpl.format(f"{y}-{m:02d}-{d:02d}"), None))

        # YYYY/M/D: "2026/3/9"
        tmpl = rng.choice(sentence_templates)
        pairs.append((tmpl.format(f"{y}/{m}/{d}"), None))

    # Decades — normalizer doesn't expand these (adjacent alpha skip),
    # so the model must learn. Category B.
    for decade in range(1920, 2030, 10):
        d_str = f"the {decade}s"
        d_spoken = f"the {n2w(decade)}s"
        pairs.append((f"It happened in {d_str}.", f"It happened in {d_spoken}."))
        pairs.append((f"Music from {d_str} was iconic.", f"Music from {d_spoken} was iconic."))

    return pairs


def gen_time(rng: random.Random) -> list[tuple[str, str | None]]:
    """Category A: Generate raw text with time patterns. Normalizer handles expansion.

    Only generates patterns the normalizer actually handles:
    - 12-hour with AM/PM (all variants)
    - 24-hour for hours 0 and 13-23 (normalizer skips ambiguous 1-12 without AM/PM)
    """
    pairs = []

    sentence_templates = [
        "The meeting is at {}.",
        "She arrived at {}.",
        "The flight departs at {}.",
        "Class starts at {}.",
        "The show begins at {}.",
        "He wakes up at {} every day.",
        "Dinner is served at {}.",
        "The store closes at {}.",
        "The alarm went off at {}.",
        "We left at {} sharp.",
        "The bus arrives at {}.",
        "Check-in is at {}.",
        "The ceremony begins at {}.",
        "Sunset is at {} today.",
    ]

    # 12-hour times with AM/PM — normalizer handles these
    for h in range(1, 13):
        for minute in range(0, 60):
            for period in ["AM", "PM"]:
                raw_time = f"{h}:{minute:02d} {period}"
                n_tmpls = 3 if minute % 15 == 0 else 2
                for tmpl in rng.sample(sentence_templates, n_tmpls):
                    pairs.append((tmpl.format(raw_time), None))

            # Lowercase am/pm and a.m./p.m. variants at common times
            if minute in [0, 15, 30, 45]:
                for period in ["AM", "PM"]:
                    for raw_period in [period.lower(), f"{period[0].lower()}.{period[1].lower()}."]:
                        raw_time = f"{h}:{minute:02d} {raw_period}"
                        tmpl = rng.choice(sentence_templates)
                        pairs.append((tmpl.format(raw_time), None))

    # 24-hour times — only hour 0 and 13-23 (normalizer skips 1-12 without AM/PM)
    for h in [0] + list(range(13, 24)):
        for minute in range(0, 60, 5):
            raw_time = f"{h}:{minute:02d}"
            tmpl = rng.choice(sentence_templates)
            pairs.append((tmpl.format(raw_time), None))

    return pairs


def gen_money(rng: random.Random) -> list[tuple[str, str | None]]:
    """Category A: Generate raw text with currency amounts. Normalizer handles expansion.

    Normalizer handles: $, €, £, ¥ with preceding symbol.
    Other currencies (CHF, ₹, named) are Category B — model must learn.
    """
    pairs = []

    sentence_templates = [
        "The price is {}.",
        "She paid {}.",
        "That costs {}.",
        "He earned {} per hour.",
        "The budget was {}.",
        "They donated {}.",
        "The ticket costs {}.",
        "Rent is {} per month.",
        "The fine was {}.",
        "She saved {}.",
        "The repair cost {}.",
        "He tipped {}.",
        "The total came to {}.",
        "Insurance costs {} annually.",
    ]

    # ── Category A: $, €, £, ¥ (normalizer handles these) ──────────────
    dollar_values = (
        list(range(0, 100)) +
        list(range(100, 1000, 10)) +
        list(range(1000, 10000, 100)) +
        list(range(10000, 100001, 5000)) +
        [250000, 500000, 1000000, 5000000, 10000000]
    )
    cent_values = [0, 1, 5, 10, 15, 20, 25, 33, 49, 50, 75, 95, 99]

    for dollars in dollar_values:
        for cents in cent_values:
            if dollars == 0 and cents == 0:
                continue
            raw = f"${dollars}.{cents:02d}" if cents else f"${dollars}"
            tmpl = rng.choice(sentence_templates)
            pairs.append((tmpl.format(raw), None))

    # Comma-formatted dollars
    for _ in range(500):
        dollars = rng.randint(1000, 9999999)
        cents = rng.randint(0, 99)
        raw = f"${dollars:,}.{cents:02d}" if cents else f"${dollars:,}"
        tmpl = rng.choice(sentence_templates)
        pairs.append((tmpl.format(raw), None))

    # Pounds
    for amount in list(range(1, 50)) + list(range(50, 1001, 50)) + [2000, 5000, 10000]:
        for pence in [0, 25, 50, 75, 99]:
            raw = f"£{amount}.{pence:02d}" if pence else f"£{amount}"
            tmpl = rng.choice(sentence_templates)
            pairs.append((tmpl.format(raw), None))

    # Euros
    euro_values = (
        list(range(0, 100)) +
        list(range(100, 1000, 10)) +
        list(range(1000, 10000, 100)) +
        list(range(10000, 100001, 5000)) +
        [250000, 500000, 1000000]
    )
    for amount in euro_values:
        for ct in [0, 1, 5, 10, 25, 50, 75, 99]:
            if amount == 0 and ct == 0:
                continue
            raw = f"€{amount}.{ct:02d}" if ct else f"€{amount}"
            tmpl = rng.choice(sentence_templates)
            pairs.append((tmpl.format(raw), None))

    # Comma-formatted euros
    for _ in range(500):
        amount = rng.randint(1000, 9999999)
        ct = rng.randint(0, 99)
        raw = f"€{amount:,}.{ct:02d}" if ct else f"€{amount:,}"
        tmpl = rng.choice(sentence_templates)
        pairs.append((tmpl.format(raw), None))

    # Yen
    yen_amounts = (
        list(range(1, 20)) + list(range(100, 1001, 100)) +
        [1500, 2000, 3000, 5000, 10000, 50000, 100000, 1000000]
    )
    for amount in yen_amounts:
        raw = f"¥{amount:,}" if amount >= 1000 else f"¥{amount}"
        tmpl = rng.choice(sentence_templates)
        pairs.append((tmpl.format(raw), None))

    # ── Category B: currencies normalizer doesn't handle ────────────────

    # Swiss francs (CHF prefix, no symbol normalizer knows)
    chf_amounts = list(range(1, 50)) + list(range(50, 1001, 50)) + [2000, 5000, 10000]
    for amount in chf_amounts:
        for ct in [0, 50]:
            if ct == 0:
                spoken = f"{n2w(amount)} Swiss franc{'s' if amount != 1 else ''}"
                raw = f"CHF {amount}"
            else:
                spoken = (f"{n2w(amount)} Swiss franc{'s' if amount != 1 else ''}"
                          f" and {n2w(ct)} centimes")
                raw = f"CHF {amount}.{ct:02d}"
            tmpl = rng.choice(sentence_templates)
            pairs.append((tmpl.format(raw), tmpl.format(spoken)))

    # Indian rupees
    rupee_amounts = (
        list(range(1, 20)) + list(range(100, 1001, 100)) +
        [1500, 2000, 5000, 10000, 50000, 100000]
    )
    for amount in rupee_amounts:
        spoken = f"{n2w(amount)} rupee{'s' if amount != 1 else ''}"
        raw = f"₹{amount:,}" if amount >= 1000 else f"₹{amount}"
        tmpl = rng.choice(sentence_templates)
        pairs.append((tmpl.format(raw), tmpl.format(spoken)))

    # Named currencies (no symbol)
    NAMED_CURRENCIES = [
        ("yen", "yen"), ("yuan", "yuan"), ("won", "won"),
        ("peso", "pesos"), ("ruble", "rubles"), ("krona", "kronor"),
        ("real", "reais"), ("franc", "francs"), ("rupee", "rupees"),
        ("dinar", "dinars"),
    ]
    for amount in [1, 2, 5, 10, 20, 50, 100, 500, 1000, 5000, 10000, 100000, 1000000]:
        for singular, plural in NAMED_CURRENCIES:
            name = singular if amount == 1 else plural
            spoken = f"{n2w(amount)} {name}"
            raw = f"{amount:,} {name}" if amount >= 1000 else f"{amount} {name}"
            tmpl = rng.choice(sentence_templates)
            pairs.append((tmpl.format(raw), tmpl.format(spoken)))

    return pairs


def gen_measure(rng: random.Random) -> list[tuple[str, str]]:
    pairs = []

    UNITS = {
        "ft": ("foot", "feet"), "in": ("inch", "inches"),
        "mi": ("mile", "miles"), "mph": ("miles per hour", "miles per hour"),
        "km/h": ("kilometers per hour", "kilometers per hour"),
        "kg": ("kilogram", "kilograms"), "lb": ("pound", "pounds"),
        "lbs": ("pound", "pounds"),
        "oz": ("ounce", "ounces"),
        "cm": ("centimeter", "centimeters"), "mm": ("millimeter", "millimeters"),
        "m": ("meter", "meters"), "km": ("kilometer", "kilometers"),
        "L": ("liter", "liters"), "ml": ("milliliter", "milliliters"),
        "gal": ("gallon", "gallons"),
        "GB": ("gigabyte", "gigabytes"), "MB": ("megabyte", "megabytes"),
        "TB": ("terabyte", "terabytes"), "KB": ("kilobyte", "kilobytes"),
        "GHz": ("gigahertz", "gigahertz"), "MHz": ("megahertz", "megahertz"),
        "Hz": ("hertz", "hertz"),
        "kWh": ("kilowatt hour", "kilowatt hours"),
        "W": ("watt", "watts"), "kW": ("kilowatt", "kilowatts"),
        "V": ("volt", "volts"), "mA": ("milliamp", "milliamps"),
        "dB": ("decibel", "decibels"),
        "cal": ("calorie", "calories"), "kcal": ("kilocalorie", "kilocalories"),
        "mg": ("milligram", "milligrams"), "g": ("gram", "grams"),
        "psi": ("P S I", "P S I"),
        "rpm": ("R P M", "R P M"),
    }

    templates = [
        "The distance is {}.",
        "She ran {}.",
        "It weighs {}.",
        "The file is {}.",
        "He drove {}.",
        "The tank holds {}.",
        "The room is {} wide.",
        "The package weighs {}.",
        "The speed limit is {}.",
        "The height is {}.",
        "She measured {}.",
        "The tower stands {} tall.",
        "The dose is {}.",
        "The capacity is {}.",
    ]

    ATTACHED = {"mph", "km/h", "ft", "in", "lbs", "rpm", "psi"}

    values = (
        list(range(1, 30)) +
        list(range(30, 100, 5)) +
        list(range(100, 1001, 25)) +
        list(range(1000, 10001, 500)) +
        [20000, 50000, 100000]
    )

    for val in values:
        val_word = n2w(val)
        for abbr, (singular, plural) in UNITS.items():
            unit_word = singular if val == 1 else plural
            if abbr in ATTACHED:
                raw_val = f"{val}{abbr}"
            else:
                raw_val = f"{val} {abbr}"
            tmpl = rng.choice(templates)
            raw = tmpl.format(raw_val)
            spoken = tmpl.format(f"{val_word} {unit_word}")
            pairs.append((raw, spoken))

    # Decimal measurements: "3.5 kg"
    for _ in range(500):
        whole = rng.randint(0, 100)
        frac = rng.randint(1, 9)
        val = f"{whole}.{frac}"
        val_spoken = f"{n2w(whole)} point {n2w(frac)}"
        abbr, (singular, plural) = rng.choice(list(UNITS.items()))
        unit_word = plural  # decimal values are always plural
        if abbr in ATTACHED:
            raw_val = f"{val}{abbr}"
        else:
            raw_val = f"{val} {abbr}"
        tmpl = rng.choice(templates)
        raw = tmpl.format(raw_val)
        spoken = tmpl.format(f"{val_spoken} {unit_word}")
        pairs.append((raw, spoken))

    # Temperature (extended range)
    for temp in list(range(-40, 120, 2)):
        for unit, name in [("C", "Celsius"), ("F", "Fahrenheit")]:
            temp_word = n2w(abs(temp))
            if temp < 0:
                temp_word = "minus " + temp_word
            raw = f"The temperature is {temp}°{unit}."
            spoken = f"The temperature is {temp_word} degrees {name}."
            pairs.append((raw, spoken))

    # Percentage — Category A (normalizer handles %)
    for pct in list(range(0, 101)) + [150, 200, 300, 500]:
        tmpl = rng.choice(["About {}% of users.", "The rate is {}%.", "A {}% increase.", "Only {}% remain."])
        raw = tmpl.format(pct)
        pairs.append((raw, None))

    return pairs


def gen_telephone(rng: random.Random) -> list[tuple[str, str | None]]:
    """Category A: Generate raw text with phone numbers. Normalizer handles expansion."""
    pairs = []

    templates = [
        "Call {}.",
        "Her number is {}.",
        "Reach us at {}.",
        "Contact {} for details.",
        "The office phone is {}.",
        "Dial {} now.",
        "The hotline is {}.",
        "For reservations call {}.",
        "Text {} for updates.",
        "The fax number is {}.",
    ]

    for _ in range(800):
        area = f"{rng.randint(200, 999)}"
        prefix = f"{rng.randint(200, 999)}"
        line = f"{rng.randint(0, 9999):04d}"

        for fmt in [f"({area}) {prefix}-{line}",
                     f"{area}-{prefix}-{line}",
                     f"{area}.{prefix}.{line}"]:
            tmpl = rng.choice(templates)
            pairs.append((tmpl.format(fmt), None))

    # 1-800 numbers
    for _ in range(200):
        prefix = f"{rng.randint(200, 999)}"
        line = f"{rng.randint(0, 9999):04d}"
        for toll_free in ["800", "888", "877", "866", "855"]:
            tmpl = rng.choice(templates)
            pairs.append((tmpl.format(f"1-{toll_free}-{prefix}-{line}"), None))

    return pairs


def gen_address(rng: random.Random) -> list[tuple[str, str]]:
    pairs = []

    STREET_TYPES = {
        "St.": "Street", "Dr.": "Drive", "Ave.": "Avenue",
        "Blvd.": "Boulevard", "Rd.": "Road", "Ln.": "Lane", "Ct.": "Court",
    }
    STREET_NAMES = ["Main", "Oak", "Park", "Elm", "Pine", "Maple", "Cedar",
                    "Walnut", "Cherry", "Spring", "Lake", "Hill", "River",
                    "Forest", "Valley", "Mountain", "Harbor", "Bay"]

    templates = [
        "She lives at {} {}.",
        "The office is at {} {}.",
        "Turn left on {} {}.",
        "Deliver it to {} {}.",
        "The store is at {} {}.",
    ]

    for _ in range(1000):
        num = rng.choice(list(range(1, 300)) + list(range(300, 15000, 50)))
        name = rng.choice(STREET_NAMES)
        abbr, full = rng.choice(list(STREET_TYPES.items()))
        num_spoken = address_number(num)
        tmpl = rng.choice(templates)
        raw = tmpl.format(num, f"{name} {abbr}")
        spoken = tmpl.format(num_spoken, f"{name} {full}")
        pairs.append((raw, spoken))

    return pairs


def gen_fraction(rng: random.Random) -> list[tuple[str, str | None]]:
    """Category A: Generate raw text with fractions. Normalizer handles expansion."""
    pairs = []

    templates = [
        "Add {} cup of flour.",
        "About {} of them agreed.",
        "She ate {} of the pizza.",
        "The tank is {} full.",
        "Approximately {} voted yes.",
        "Nearly {} of the work is done.",
        "Only {} of the budget remains.",
        "Mix with {} teaspoon of salt.",
        "He finished {} of the course.",
        "The recipe calls for {} cup of sugar.",
    ]

    for num in range(1, 15):
        for den in range(2, 25):
            if num >= den:
                continue
            n_tmpls = 3 if den <= 10 else 2
            for tmpl in rng.sample(templates, min(n_tmpls, len(templates))):
                pairs.append((tmpl.format(f"{num}/{den}"), None))

    return pairs


def gen_roman(rng: random.Random) -> list[tuple[str, str]]:
    pairs = []

    ROMAN_MAP = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]

    def to_roman(n: int) -> str:
        result = []
        for value, numeral in ROMAN_MAP:
            while n >= value:
                result.append(numeral)
                n -= value
        return "".join(result)

    # People (ordinal)
    names = ["Henry", "Louis", "Elizabeth", "Charles", "Pope Benedict",
             "King George", "Queen Victoria", "Edward", "Frederick",
             "Napoleon", "James", "Richard", "Philip"]
    for name in names:
        for n in range(1, 21):
            roman = to_roman(n)
            spoken_n = "the " + ordinal(n)
            pairs.append((f"{name} {roman}", f"{name} {spoken_n}"))

    # Sections/chapters (cardinal)
    for label in ["Chapter", "Part", "Volume", "Act", "Section", "Book", "Level"]:
        for n in range(1, 30):
            roman = to_roman(n)
            pairs.append((f"{label} {roman}", f"{label} {n2w(n)}"))

    return pairs


def gen_range(rng: random.Random) -> list[tuple[str, str]]:
    pairs = []

    templates = [
        "Expect {} {}.",
        "Between {} {}.",
        "It takes {} {}.",
        "The price is {} {}.",
        "There are {} {}.",
        "She needs {} {}.",
        "Roughly {} {} were affected.",
        "The range is {} {}.",
    ]

    units = ["days", "weeks", "months", "years", "hours", "minutes",
             "people", "pages", "items", "units", "miles", "degrees",
             "students", "employees"]

    for _ in range(600):
        lo = rng.randint(1, 500)
        hi = lo + rng.randint(1, 500)
        unit = rng.choice(units)
        tmpl = rng.choice(templates)
        raw = tmpl.format(f"{lo}-{hi}", unit)
        spoken = tmpl.format(f"{n2w(lo)} to {n2w(hi)}", unit)
        pairs.append((raw, spoken))

    # Year ranges
    for _ in range(200):
        y1 = rng.randint(1800, 2020)
        y2 = y1 + rng.randint(1, 50)
        raw = f"During {y1}-{y2}."
        spoken = f"During {year_to_words(y1)} to {year_to_words(y2)}."
        pairs.append((raw, spoken))

    return pairs


def gen_score(rng: random.Random) -> list[tuple[str, str]]:
    pairs = []

    templates = [
        "They won {}-{}.",
        "The final score was {}-{}.",
        "Leading {}-{} at halftime.",
        "The match ended {}-{}.",
        "Down {}-{} in the fourth quarter.",
        "The score is {}-{}.",
        "They trailed {}-{} in the second half.",
        "A {}-{} victory.",
    ]

    for _ in range(500):
        a = rng.randint(0, 60)
        b = rng.randint(0, 60)
        a_word = n2w(a) if a > 0 else "nothing"
        b_word = n2w(b) if b > 0 else "nothing"
        tmpl = rng.choice(templates)
        raw = tmpl.format(a, b)
        spoken = tmpl.format(a_word, b_word).replace("-", " to ")
        pairs.append((raw, spoken))

    return pairs


def gen_abbreviation(rng: random.Random) -> list[tuple[str, str]]:
    pairs = []

    TITLES = {
        "Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Misses", "Ms.": "Miz",
        "Prof.": "Professor", "Rev.": "Reverend",
        "Gov.": "Governor", "Sgt.": "Sergeant", "Lt.": "Lieutenant",
        "Col.": "Colonel", "Gen.": "General",
    }
    SUFFIXES = {"Jr.": "Junior", "Sr.": "Senior"}

    names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Davis",
             "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas",
             "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia",
             "Martinez", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee"]

    title_templates = [
        "{} {} arrived early.",
        "We met {} {} at the conference.",
        "{} {} gave a speech.",
        "The letter was from {} {}.",
        "{} {} is a respected leader.",
    ]

    for abbr, full in TITLES.items():
        for name in names:
            tmpl = rng.choice(title_templates)
            pairs.append((tmpl.format(abbr, name), tmpl.format(full, name)))

    # Jr./Sr.
    for abbr, full in SUFFIXES.items():
        for name in names[:10]:
            first = rng.choice(["Robert", "James", "John", "William", "Michael"])
            raw = f"{first} {name} {abbr} attended."
            spoken = f"{first} {name} {full} attended."
            pairs.append((raw, spoken))

    # Context-dependent Dr./St.
    for name in ["Oak", "Pine", "Elm", "Maple", "Cedar", "Park", "Lake"]:
        pairs.append((f"Turn on {name} Dr.", f"Turn on {name} Drive."))
        pairs.append((f"Walk down {name} St.", f"Walk down {name} Street."))

    # Street/place abbreviations
    PLACE_ABBRS = {
        "Ave.": "Avenue", "Blvd.": "Boulevard", "Ln.": "Lane",
        "Ct.": "Court", "Pl.": "Place", "Rd.": "Road",
        "Pkwy.": "Parkway", "Hwy.": "Highway",
    }
    street_names = ["Oak", "Pine", "Elm", "Maple", "Cedar", "Park", "Lake",
                    "Main", "High", "River", "Sunset", "Broadway"]
    place_templates = [
        "Turn on {} {}",
        "She lives on {} {}",
        "The office is at {} {}",
        "Drive down {} {}",
    ]
    for abbr, full in PLACE_ABBRS.items():
        for street in rng.sample(street_names, min(5, len(street_names))):
            tmpl = rng.choice(place_templates)
            pairs.append((tmpl.format(street, abbr), tmpl.format(street, full)))

    # Organization/corporate abbreviations
    ORG_ABBRS = {
        "Corp.": "Corporation", "Inc.": "Incorporated", "Ltd.": "Limited",
        "Co.": "Company", "Assn.": "Association", "Bros.": "Brothers",
        "Dept.": "Department", "Div.": "Division", "Univ.": "University",
        "Intl.": "International", "Natl.": "National",
    }
    org_names = ["Smith", "Johnson", "Western", "Eastern", "Pacific", "Atlantic",
                 "Central", "National", "American", "Global", "United", "Royal"]
    org_templates = [
        "{} {} announced earnings.",
        "She joined {} {} last year.",
        "The {} {} headquarters is downtown.",
        "{} {} filed the report.",
    ]
    for abbr, full in ORG_ABBRS.items():
        for org in rng.sample(org_names, min(4, len(org_names))):
            tmpl = rng.choice(org_templates)
            pairs.append((tmpl.format(org, abbr), tmpl.format(org, full)))

    # Fixed abbreviations
    FIXED = {
        "vs.": "versus", "etc.": "et cetera", "e.g.": "for example",
        "i.e.": "that is", "approx.": "approximately", "dept.": "department",
        "govt.": "government", "fig.": "figure", "vol.": "volume",
        "no.": "number", "est.": "established", "min.": "minimum",
        "max.": "maximum", "avg.": "average", "temp.": "temperature",
    }
    fixed_templates = [
        "She chose option A {} option B.",
        "Bring tools, supplies, {}",
        "Some items ({} pens and paper) were missing.",
        "The report stated, {}",
        "See {} for more information.",
    ]
    for abbr, full in FIXED.items():
        for _ in range(5):
            tmpl = rng.choice(fixed_templates)
            pairs.append((tmpl.format(abbr), tmpl.format(full)))

    return pairs


def gen_bare(rng: random.Random) -> list[tuple[str, str | None]]:
    """Generate bare patterns (no sentence wrapper) for ALL normalize.h patterns.

    Category A (normalizer handles): spoken=None → normalize_cli produces spoken form
    Category B (model learns): spoken form provided directly
    """
    pairs = []

    # ── Category A: bare dates (textual) ───────────────────────────────
    for _ in range(2000):
        y = rng.randint(1900, 2099)
        m = rng.randint(1, 12)
        d = rng.randint(1, 28)
        month = calendar.month_name[m]
        month_abbr = calendar.month_abbr[m]
        pairs.append((f"{month} {d}, {y}", None))
        pairs.append((f"{month_abbr} {d}, {y}", None))

    # ── Category A: bare dates (numeric) ───────────────────────────────
    for _ in range(2000):
        y = rng.randint(1900, 2099)
        m = rng.randint(1, 12)
        d = rng.randint(1, 28)
        pairs.append((f"{d}/{m}/{y}", None))            # D/M/Y
        pairs.append((f"{d}.{m:02d}.{y}", None))         # D.M.Y
        pairs.append((f"{y}-{m:02d}-{d:02d}", None))     # ISO
        pairs.append((f"{y}/{m}/{d}", None))              # YYYY/M/D

    # ── Category A: bare money ─────────────────────────────────────────
    currencies = ["$", "€", "£"]
    for _ in range(2000):
        cur = rng.choice(currencies)
        dollars = rng.choice(list(range(1, 100)) + list(range(100, 10001, 50)))
        cents = rng.choice([0, 0, 0, 25, 50, 75, 99, rng.randint(1, 99)])
        raw = f"{cur}{dollars}.{cents:02d}" if cents else f"{cur}{dollars}"
        pairs.append((raw, None))

    # ── Category A: bare times (colon) ─────────────────────────────────
    for h in range(1, 13):
        for minute in range(0, 60, 5):
            for period in ["AM", "PM", "am", "pm", "a.m.", "p.m."]:
                pairs.append((f"{h}:{minute:02d} {period}", None))

    # ── Category A: bare times (dot) ───────────────────────────────────
    for h in range(1, 13):
        for minute in range(0, 60, 5):
            for period in ["am", "pm", "a.m.", "p.m."]:
                pairs.append((f"{h}.{minute:02d} {period}", None))

    # ── Category A: bare fractions ─────────────────────────────────────
    for num in range(1, 20):
        for den in range(2, 20):
            if num < den:
                pairs.append((f"{num}/{den}", None))

    # ── Category A: bare ordinals ──────────────────────────────────────
    for n in list(range(1, 100)) + [100, 200, 500, 1000]:
        pairs.append((f"{n}{ordinal_suffix(n)}", None))

    # ── Category A: bare percentages ───────────────────────────────────
    for _ in range(500):
        n = rng.choice(list(range(1, 100)) + [0, 100, 150, 200])
        pairs.append((f"{n}%", None))
    for _ in range(500):
        n = rng.randint(0, 99)
        d = rng.randint(1, 9)
        pairs.append((f"{n}.{d}%", None))

    # ── Category A: bare number+unit ───────────────────────────────────
    units = ["kg", "lb", "lbs", "oz", "km", "ft", "mi", "mm", "cm",
             "ml", "MHz", "GHz", "kHz", "KB", "MB", "GB", "TB",
             "mph", "kph", "rpm", "ms", "ns", "Hz"]
    for _ in range(3000):
        unit = rng.choice(units)
        if rng.random() < 0.5:
            n = rng.randint(1, 9999)
            raw = f"{n}{unit}"
        else:
            n = rng.randint(0, 999)
            d = rng.randint(1, 9)
            raw = f"{n}.{d}{unit}"
        pairs.append((raw, None))
        # Also with space
        if rng.random() < 0.3:
            pairs.append((raw.replace(unit, f" {unit}"), None))

    # ── Category A: bare decimals ──────────────────────────────────────
    for _ in range(1000):
        whole = rng.randint(0, 999)
        frac = rng.randint(1, 99)
        pairs.append((f"{whole}.{frac}", None))

    # ── Category A: bare cardinals ─────────────────────────────────────
    for n in list(range(0, 100)) + list(range(100, 1000, 10)) + \
             list(range(1000, 10001, 100)):
        pairs.append((str(n), None))
    for _ in range(1000):
        pairs.append((str(rng.randint(1, 999999)), None))

    # ── Category A: bare telephone ─────────────────────────────────────
    for _ in range(500):
        area = f"{rng.randint(200, 999)}"
        prefix = f"{rng.randint(200, 999)}"
        line = f"{rng.randint(0, 9999):04d}"
        pairs.append((f"({area}) {prefix}-{line}", None))
        pairs.append((f"{area}-{prefix}-{line}", None))

    # ── Category A: bare symbols ───────────────────────────────────────
    for _ in range(200):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        pairs.append((f"{a} & {b}", None))
        pairs.append((f"{a} + {b}", None))

    # ── Category A: bare dotted initialisms ────────────────────────────
    initialisms = ["U.S.A.", "U.S.", "F.B.I.", "C.I.A.", "U.K.", "E.U.",
                   "N.A.S.A.", "U.N.", "D.C.", "N.Y.", "L.A.", "A.M.",
                   "P.M.", "B.C.", "A.D.", "Ph.D.", "M.D.", "D.J.",
                   "V.I.P.", "A.S.A.P.", "R.S.V.P.", "P.S.", "I.Q.",
                   "D.N.A.", "R.N.A.", "G.P.S.", "A.I.", "I.T."]
    for init in initialisms:
        pairs.append((init, None))

    # ── Category B: bare roman numerals ────────────────────────────────
    ROMAN_MAP = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    def to_roman(n):
        result = []
        for value, numeral in ROMAN_MAP:
            while n >= value:
                result.append(numeral)
                n -= value
        return "".join(result)

    names = ["Henry", "Louis", "Elizabeth", "Charles", "Pope Benedict",
             "King George", "Edward", "Frederick", "James", "Richard"]
    for name in names:
        for n in range(1, 25):
            roman = to_roman(n)
            if roman:
                pairs.append((f"{name} {roman}", f"{name} the {ordinal(n)}"))

    for label in ["Chapter", "Part", "Volume", "Act", "Section"]:
        for n in range(1, 30):
            roman = to_roman(n)
            if roman:
                pairs.append((f"{label} {roman}", f"{label} {n2w(n)}"))

    return pairs


def gen_decimal(rng: random.Random) -> list[tuple[str, str | None]]:
    """Category A: Generate raw text with decimal numbers. Normalizer handles expansion."""
    pairs = []

    templates = [
        "The value is {}.",
        "It measured {}.",
        "The result was {}.",
        "She recorded {}.",
        "The reading shows {}.",
        "The average is {}.",
        "It dropped to {}.",
        "The ratio is {}.",
        "The coefficient is {}.",
        "The index reached {}.",
        "The concentration is {}.",
        "It rose to {}.",
        "The estimate is {}.",
        "The output was {}.",
    ]

    whole_parts = list(range(0, 21)) + [25, 30, 42, 50, 75, 99, 100, 123,
                                         200, 314, 500, 750, 999]

    for whole in whole_parts:
        for frac_digit in range(1, 10):
            tmpl = rng.choice(templates)
            pairs.append((tmpl.format(f"{whole}.{frac_digit}"), None))

    for whole in whole_parts:
        for frac in rng.sample(range(1, 100), min(10, 99)):
            tmpl = rng.choice(templates)
            pairs.append((tmpl.format(f"{whole}.{frac:02d}"), None))

    for whole in whole_parts[:15]:
        for frac in rng.sample(range(1, 1000), min(5, 999)):
            tmpl = rng.choice(templates)
            pairs.append((tmpl.format(f"{whole}.{frac:03d}"), None))

    for _ in range(100):
        whole = rng.randint(0, 999)
        n_frac_digits = rng.choice([1, 2, 3])
        if n_frac_digits == 1:
            frac_s = str(rng.randint(1, 9))
        elif n_frac_digits == 2:
            frac_s = f"{rng.randint(1, 99):02d}"
        else:
            frac_s = f"{rng.randint(1, 999):03d}"
        tmpl = rng.choice(templates)
        pairs.append((tmpl.format(f"{whole}.{frac_s}"), None))

    return pairs


def gen_connector(rng: random.Random) -> list[tuple[str, str]]:
    """Generate (raw, spoken) pairs for connector patterns: w/, w/o, 24/7, NxN."""
    pairs = []

    # ── w/ (with) ─────────────────────────────────────────────────────────
    with_templates = [
        "She works w/ the team.",
        "Come w/ me.",
        "He arrived w/ a plan.",
        "Mix the flour w/ water.",
        "She left w/ her bag.",
        "Travel w/ caution.",
        "He spoke w/ confidence.",
        "A room w/ a view.",
        "Coffee w/ cream.",
        "She danced w/ joy.",
        "The car comes w/ heated seats.",
        "A man w/ a mission.",
        "She agreed w/ the proposal.",
        "He cooked w/ fresh ingredients.",
        "We drove w/ the windows down.",
        "A house w/ a garden.",
        "She returned w/ good news.",
        "He plays w/ skill.",
        "Deal w/ it.",
        "A burger w/ fries.",
    ]
    with_spoken = [t.replace("w/", "with") for t in with_templates]
    for raw, spoken in zip(with_templates, with_spoken):
        pairs.append((raw, spoken))

    # ── w/o (without) ─────────────────────────────────────────────────────
    without_templates = [
        "He left w/o warning.",
        "She managed w/o help.",
        "The project continued w/o funding.",
        "He ran w/o shoes.",
        "She passed w/o studying.",
        "They survived w/o water.",
        "The car runs w/o gas.",
        "She won w/o breaking a sweat.",
        "He spoke w/o hesitation.",
        "The job was done w/o errors.",
        "She traveled w/o luggage.",
        "He finished w/o complaining.",
        "The team succeeded w/o their captain.",
        "She left w/o saying goodbye.",
        "He worked w/o a break.",
        "Coffee w/o sugar.",
        "The door opened w/o a key.",
        "She smiled w/o effort.",
        "He answered w/o thinking.",
        "They moved w/o permission.",
    ]
    without_spoken = [t.replace("w/o", "without") for t in without_templates]
    for raw, spoken in zip(without_templates, without_spoken):
        pairs.append((raw, spoken))

    # ── 24/7 and N/N (ratio-style) ────────────────────────────────────────
    ratio_templates = [
        "The store is open {}.",
        "Support is available {}.",
        "They work {}.",
        "The service runs {}.",
        "Monitoring continues {}.",
        "Security patrols {}.",
    ]
    # 24/7
    for tmpl in ratio_templates:
        pairs.append((tmpl.format("24/7"), tmpl.format("twenty four seven")))

    split_templates = [
        "It was a {} split.",
        "The odds are {}.",
        "They agreed on a {} deal.",
        "A {} chance of success.",
        "The vote was {}.",
        "They divided it {}.",
    ]
    # 50/50
    for tmpl in split_templates:
        pairs.append((tmpl.format("50/50"), tmpl.format("fifty fifty")))
    # Other common splits
    split_values = [(60, 40), (70, 30), (80, 20), (90, 10), (75, 25)]
    for a, b in split_values:
        tmpl = rng.choice(split_templates)
        pairs.append((tmpl.format(f"{a}/{b}"), tmpl.format(f"{n2w(a)} {n2w(b)}")))

    # ── NxN (N by N) ─────────────────────────────────────────────────────
    dim_templates = [
        "The room is {}.",
        "Buy a {} rug.",
        "A {} photo.",
        "The grid is {}.",
        "She needs a {} board.",
        "The display is {}.",
        "The pool measures {}.",
        "It fits a {} space.",
        "The screen is {}.",
        "A {} matrix.",
        "The tile is {}.",
        "The plot is {}.",
        "The window is {}.",
        "Cut a {} piece.",
    ]
    dim_values = (
        [(2, 2), (2, 4), (3, 3), (3, 5), (4, 4), (4, 6), (4, 8),
         (5, 5), (5, 7), (6, 6), (6, 8), (6, 9), (8, 8), (8, 10),
         (8, 11), (9, 12), (10, 10), (10, 12), (10, 13), (10, 15),
         (11, 14), (11, 17), (12, 12), (12, 16), (16, 16), (16, 20),
         (20, 20), (20, 24), (20, 30), (24, 36)] +
        [(rng.randint(1, 100), rng.randint(1, 100)) for _ in range(50)]
    )
    for a, b in dim_values:
        tmpl = rng.choice(dim_templates)
        raw = tmpl.format(f"{a}x{b}")
        spoken = tmpl.format(f"{n2w(a)} by {n2w(b)}")
        pairs.append((raw, spoken))

    return pairs


def gen_symbol(rng: random.Random) -> list[tuple[str, str]]:
    """Generate (raw, spoken) pairs for symbol patterns."""
    pairs = []

    # ── Percent: 15% → fifteen percent ──────────────────────────────────
    pct_templates = [
        "Sales grew {}%.",
        "The rate is {}%.",
        "About {}% of users agree.",
        "She scored {}% on the test.",
        "Only {}% remain.",
        "The tax rate is {}%.",
        "Unemployment fell to {}%.",
        "He completed {}% of the task.",
        "The yield was {}%.",
        "A {}% increase was reported.",
    ]
    for val in list(range(1, 100)) + [0, 100, 150, 200]:
        tmpl = rng.choice(pct_templates)
        pairs.append((tmpl.format(val), tmpl.format(f"{n2w(val)} percent")))

    # ── Ampersand: Smith & Jones → Smith and Jones ──────────────────────
    amp_templates = [
        "{} & {} LLC.",
        "{} & {} announced a merger.",
        "The firm {} & {} is hiring.",
        "She works at {} & {}.",
        "{} & {} published the report.",
    ]
    name_pairs_list = [
        ("Smith", "Jones"), ("Johnson", "Williams"), ("Brown", "Davis"),
        ("Miller", "Wilson"), ("Moore", "Taylor"), ("Anderson", "Thomas"),
        ("Jackson", "White"), ("Harris", "Martin"), ("Garcia", "Martinez"),
        ("Clark", "Lewis"), ("Hall", "Allen"), ("Young", "King"),
        ("Wright", "Scott"), ("Green", "Adams"), ("Baker", "Nelson"),
        ("Hill", "Campbell"), ("Mitchell", "Roberts"), ("Carter", "Phillips"),
        ("Evans", "Turner"), ("Collins", "Stewart"),
    ]
    for a, b in name_pairs_list:
        tmpl = rng.choice(amp_templates)
        pairs.append((tmpl.format(a, b), tmpl.format(a, b).replace(" & ", " and ")))

    # ── At sign: info@example.com → info at example dot com ─────────────
    at_templates = [
        "Email us at {}.",
        "Contact {}.",
        "Send a message to {}.",
        "Her address is {}.",
        "Write to {}.",
    ]
    usernames = ["info", "support", "admin", "sales", "contact", "hello",
                 "john", "jane", "david", "sarah", "team", "help"]
    domains = [("example.com", "example dot com"), ("gmail.com", "gmail dot com"),
               ("yahoo.com", "yahoo dot com"), ("outlook.com", "outlook dot com"),
               ("company.org", "company dot org"), ("school.edu", "school dot edu"),
               ("work.net", "work dot net"), ("mail.co", "mail dot co")]
    for user in usernames:
        domain, spoken_domain = rng.choice(domains)
        tmpl = rng.choice(at_templates)
        pairs.append((tmpl.format(f"{user}@{domain}"),
                       tmpl.format(f"{user} at {spoken_domain}")))

    # ── Hash: #1 → number one, #hashtag → hashtag ──────────────────────
    hash_templates = [
        "Issue #{} is critical.",
        "She is ranked #{}.",
        "The #{} item on the list.",
        "He finished #{}.",
        "Track #{} is my favorite.",
        "Rule #{} applies here.",
    ]
    for num in list(range(1, 51)) + [100, 200, 500]:
        tmpl = rng.choice(hash_templates)
        raw = tmpl.format(num)
        spoken = tmpl.replace("#{}", f"number {n2w(num)}")
        pairs.append((raw, spoken))

    # ── Feet/inches: 6'2" → six feet two inches ──────────────────────────
    height_templates = [
        "He stands {} tall.",
        "She is {}.",
        "The ceiling is {}.",
        "The fence is {} high.",
        "The shelf is {} off the ground.",
        "The player is {}.",
        "The door is {} high.",
        "She measured {} in height.",
        "The wall is {} tall.",
        "The athlete is {}.",
        "The clearance is {}.",
        "The post is {} tall.",
    ]

    # Common heights (4'0" to 7'0" with various inches)
    for feet in range(3, 8):
        for inches in range(0, 12):
            raw_val = f"{feet}'{inches}\""
            if inches == 0:
                spoken_val = f"{n2w(feet)} feet"
            else:
                spoken_val = (f"{n2w(feet)} feet {n2w(inches)} "
                              f"inch{'es' if inches != 1 else ''}")
            n_tmpls = 3 if 5 <= feet <= 6 else 2
            for tmpl in rng.sample(height_templates, min(n_tmpls, len(height_templates))):
                pairs.append((tmpl.format(raw_val), tmpl.format(spoken_val)))

    # Larger foot measurements (no inches) for non-height contexts
    for feet in [8, 10, 12, 15, 20, 25, 30, 40, 50, 100]:
        raw_val = f"{feet}'"
        spoken_val = f"{n2w(feet)} feet"
        tmpl = rng.choice(height_templates)
        pairs.append((tmpl.format(raw_val), tmpl.format(spoken_val)))

    # ── Plus: 5+3 → five plus three ──────────────────────────────────────
    plus_templates = [
        "Calculate {}.",
        "The answer to {} is unknown.",
        "She wrote {} on the board.",
        "What is {}?",
        "The equation {} was solved.",
        "He added {} together.",
        "The sum {} equals the total.",
        "The formula includes {}.",
    ]
    plus_values = (
        [(a, b) for a in range(1, 20) for b in range(1, 20)
         if (a + b) <= 30 and rng.random() < 0.3] +
        [(rng.randint(10, 500), rng.randint(10, 500)) for _ in range(50)]
    )
    for a, b in plus_values:
        raw_val = f"{a}+{b}"
        spoken_val = f"{n2w(a)} plus {n2w(b)}"
        tmpl = rng.choice(plus_templates)
        pairs.append((tmpl.format(raw_val), tmpl.format(spoken_val)))

    # ── Degree: 90° → ninety degrees ─────────────────────────────────────
    # Note: unidecode converts ° to "deg", so after normalization 90° becomes
    # "90deg". We generate these patterns since the training pipeline applies
    # preprocess_text() and the model will see "90deg" as input.
    degree_templates = [
        "The angle is {}.",
        "Rotate it {} clockwise.",
        "It tilted {} to the left.",
        "The turn was {}.",
        "Set the oven to {}.",
        "A {} angle is a right angle.",
        "The slope was {}.",
        "She rotated the image {}.",
        "The compass read {}.",
        "The arc spans {}.",
    ]
    # Angles and temperatures without unit letter (bare degree)
    for deg in list(range(0, 361, 5)) + [1, 15, 22, 33, 45, 60, 72, 90, 120,
                                          135, 150, 180, 225, 270, 315, 360]:
        deg_word = n2w(deg)
        tmpl = rng.choice(degree_templates)
        raw = tmpl.format(f"{deg}\u00b0")
        spoken = tmpl.format(f"{deg_word} degrees")
        pairs.append((raw, spoken))

    return pairs


def gen_letters(rng: random.Random) -> list[tuple[str, str]]:
    """Generate (raw, spoken) pairs for letter-spelling (initialisms/acronyms)."""
    pairs = []

    LETTER_NAMES = {
        "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F",
        "G": "G", "H": "H", "I": "I", "J": "J", "K": "K", "L": "L",
        "M": "M", "N": "N", "O": "O", "P": "P", "Q": "Q", "R": "R",
        "S": "S", "T": "T", "U": "U", "V": "V", "W": "W", "X": "X",
        "Y": "Y", "Z": "Z",
    }

    def spell(s):
        return " ".join(LETTER_NAMES[c] for c in s if c in LETTER_NAMES)

    templates = [
        "The {} report was released.",
        "She works at the {}.",
        "Contact the {} for details.",
        "The {} published new guidelines.",
        "He joined the {} last year.",
        "{} officials confirmed the news.",
        "The {} approved the proposal.",
        "According to the {}, it is safe.",
        "The {} building is downtown.",
        "She studied for her {} exam.",
        "The {} released a statement.",
        "He received a {} certification.",
    ]

    # Common 2-4 letter initialisms that should be spelled
    initialisms = [
        # 3-letter
        "FBI", "CIA", "DNA", "CEO", "CFO", "CTO", "COO", "NBA", "NFL", "NHL",
        "MLB", "CNN", "BBC", "ABC", "CBS", "NBC", "PBS", "NPR", "EPA", "FDA",
        "IRS", "TSA", "FAA", "SEC", "FCC", "CDC", "NIH", "NSA", "DOJ", "DOD",
        "DOS", "DOT", "DHS", "HUD", "SBA", "SSA", "FTC", "FEC", "NRC", "GAO",
        "GPO", "ATF", "DEA", "ICE", "CBP", "TSA", "FAA", "NPS", "BLM", "EPA",
        "HIV", "AIDS", "RNA", "ATP", "CPU", "GPU", "RAM", "ROM", "USB", "LCD",
        "LED", "GPS", "FAQ", "PDF", "URL", "API", "SQL", "PHP", "CSS", "XML",
        "DIY", "CEO", "NGO", "IPO", "VPN", "UFO", "SUV", "RBI", "ERA", "MVP",
        "GDP", "GPA", "SAT", "ACT", "GRE", "MRI", "EKG", "EMT", "CPR", "OTC",
        # 2-letter
        "UN", "EU", "UK", "US", "DC", "LA", "NY", "SF", "AI", "IT", "TV", "DJ",
        "VP", "PR", "HR", "QA", "ID", "OK",
        # 4-letter
        "ACLU", "NASA", "SWAT", "ESPN", "YMCA", "STEM", "HVAC", "JPEG", "HTML",
        "HTTP", "ASAP", "RSVP", "AWOL", "ADHD", "PTSD",
        # Degree abbreviations
        "PhD", "MBA", "MPH", "JD", "MD", "RN", "BS", "BA", "MS", "MA",
    ]

    for init in initialisms:
        spoken = spell(init.upper())
        for _ in range(3):
            tmpl = rng.choice(templates)
            pairs.append((tmpl.format(init), tmpl.format(spoken)))

    # Also generate bare initialisms without sentence context
    bare_templates = [
        "The {} is important.",
        "Check the {}.",
        "She mentioned the {}.",
    ]
    for init in initialisms:
        spoken = spell(init.upper())
        tmpl = rng.choice(bare_templates)
        pairs.append((tmpl.format(init), tmpl.format(spoken)))

    return pairs


def gen_dollar_singular(rng: random.Random) -> list[tuple[str, str | None]]:
    """Category A: Oversample $1 / $1.xx patterns. Normalizer handles expansion."""
    pairs = []

    templates = [
        "The price is {}.",
        "She paid {}.",
        "That costs {}.",
        "He found {} on the ground.",
        "It was only {}.",
        "The fee is {}.",
        "She donated {}.",
        "The toll is {}.",
        "The charge was {}.",
        "He spent {}.",
        "Add {} to the total.",
        "It increased by {}.",
    ]

    for tmpl in templates:
        pairs.append((tmpl.format("$1"), None))

    for cents in range(1, 100):
        tmpl = rng.choice(templates)
        pairs.append((tmpl.format(f"$1.{cents:02d}"), None))

    common_cents = [0, 25, 50, 75, 99, 49, 29, 19, 9, 5, 10, 15, 95, 79, 59, 39]
    for cents in common_cents:
        raw = "$1" if cents == 0 else f"$1.{cents:02d}"
        for tmpl in rng.sample(templates, min(4, len(templates))):
            pairs.append((tmpl.format(raw), None))

    return pairs


GENERATORS = {
    "cardinal": gen_cardinal,
    "ordinal": gen_ordinal,
    "date": gen_date,
    "time": gen_time,
    "money": gen_money,
    "measure": gen_measure,
    "telephone": gen_telephone,
    "address": gen_address,
    "fraction": gen_fraction,
    "roman": gen_roman,
    "score": gen_score,
    "range": gen_range,
    "abbreviation": gen_abbreviation,
    "bare": gen_bare,
    "decimal": gen_decimal,
    "connector": gen_connector,
    "symbol": gen_symbol,
    "letters": gen_letters,
    "dollar_singular": gen_dollar_singular,
}


def main():
    parser = argparse.ArgumentParser(description="Generate augmentation training data")
    parser.add_argument("--output", default="data/g2p_augment.tsv",
                        help="Output TSV (normalized_text\\tphonemes)")
    parser.add_argument("--classes", nargs="*", choices=list(GENERATORS.keys()),
                        help="Only generate specific classes (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats", action="store_true", help="Show planned counts per class")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate pairs but skip phonemization")
    parser.add_argument("--normalizer", default=None,
                        help="Path to normalize_cli binary (auto-detected if not specified)")
    args = parser.parse_args()
    if not args.stats and os.path.exists(args.output):
        parser.error("Output already exists; choose a new dataset path")

    rng = random.Random(args.seed)
    classes = args.classes or list(GENERATORS.keys())

    # Generate all raw pairs first
    all_pairs = []
    class_counts = {}
    for cls in classes:
        gen = GENERATORS[cls]
        pairs = gen(rng)
        class_counts[cls] = len(pairs)
        all_pairs.extend(pairs)

    total = len(all_pairs)
    print(f"Generated {total:,} raw pairs:")
    for cls in sorted(class_counts, key=lambda x: -class_counts[x]):
        print(f"  {cls}: {class_counts[cls]:,}")

    if args.stats:
        return

    # Initialize normalizer (calls the C++ binary — single source of truth)
    normalizer = Normalizer(args.normalizer)
    print(f"Using normalizer: {normalizer.binary_path}")

    # Normalize Category A pairs (spoken=None) through the real normalizer
    cat_a_indices = [i for i, (_, spoken) in enumerate(all_pairs) if spoken is None]
    if cat_a_indices:
        cat_a_raws = [all_pairs[i][0] for i in cat_a_indices]
        print(f"Normalizing {len(cat_a_indices):,} Category A pairs through normalize_cli...")
        # Process in batches to avoid huge pipes
        BATCH = 10000
        normalized = []
        for start in range(0, len(cat_a_raws), BATCH):
            batch = cat_a_raws[start:start + BATCH]
            normalized.extend(normalizer.normalize_batch(batch))
        # Replace spoken=None with normalizer output
        for idx, norm_text in zip(cat_a_indices, normalized):
            raw = all_pairs[idx][0]
            all_pairs[idx] = (raw, norm_text)

    if args.dry_run:
        print(f"\nDry run: would phonemize {total:,} spoken forms")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            for raw, spoken in all_pairs[:200]:
                f.write(f"{raw}\t{spoken}\n")
        print(f"Sample written to {args.output}")
        return

    # Phonemize spoken forms with Misaki
    print("\nInitializing Misaki G2P...")
    misaki = MisakiG2P()

    print(f"Phonemizing {total:,} spoken forms...")
    t0 = time.monotonic()
    ok = 0
    fail = 0
    seen = set()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for i, (raw, spoken) in enumerate(all_pairs):
            # Dedup by spoken text (what goes into column 1)
            if spoken in seen:
                continue
            seen.add(spoken)

            phonemes = misaki.phonemize(spoken)
            if phonemes:
                f.write(f"{spoken}\t{phonemes}\n")
                ok += 1
            else:
                fail += 1

            if (i + 1) % 1000 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                print(f"  [{i+1:,}/{total:,}] {rate:.0f}/s  ETA {eta:.0f}s  ok={ok} fail={fail}",
                      flush=True)

    elapsed = time.monotonic() - t0
    print(f"\nDone: {ok:,} pairs saved to {args.output} ({fail} failed, {len(all_pairs)-len(seen)} dedup)")
    print(f"  {ok/elapsed:.0f} pairs/s in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
