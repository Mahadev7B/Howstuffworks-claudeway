"""Check the arithmetic written in a lesson, exactly.

Finds statements like "1/4 + 3/5 = 17/20", "3 times 4 is 12", "five plus twelve equals seventeen"
or "1 ÷ 4 = 0.25" in the lesson text and recomputes them with exact fractions. Only simple
written sums are checked; anything it can't read is left alone, so it never blocks a normal lesson.
"""
import re
from fractions import Fraction

_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_NUM = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:/\d+)?|\.\d+|" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r")"
_OP = r"(?:\+|-|−|–|×|\*|÷|/|(?<=\d )x(?= \d)|plus|minus|take away|times|multiplied by|divided by)"
_EQ = r"(?:=|equals|equal to|is equal to|is|makes|gives you|gives|you get)"
_APPROX = r"(?:≈|is about|is roughly|is around|about|roughly|around|approximately)"
_EXPR = rf"{_NUM}(?:\s*{_OP}\s*{_NUM})*"
_SEP = rf"\s*(?:{_APPROX}|{_EQ})\s*(?:{_APPROX}\s+)?"
# A chain like "1/4 + 2/8 = 2/8 + 2/8 = 4/8": every part must have the same value.
_STATEMENT = re.compile(
    rf"(?<![\w./]){_EXPR}(?:{_SEP}{_EXPR})+(?P<more>\.\.\.|…)?(?![\w/]|\.\d)",
    re.IGNORECASE,
)
_SPLIT = re.compile(rf"\s*((?:{_APPROX}|{_EQ})\s*(?:{_APPROX}\s+)?)\s*(?=[\d.]|{_NUM})", re.IGNORECASE)
_HAS_OP = re.compile(rf"{_NUM}\s*{_OP}\s*{_NUM}", re.IGNORECASE)
_TOKEN = re.compile(rf"\s*({_NUM}|{_OP})", re.IGNORECASE)
_TEXT_FIELDS = ("title", "subtitle", "explanation", "fun_fact", "narration")


def _number(token: str) -> Fraction:
    t = token.lower().replace(",", "")
    if t in _WORDS:
        return Fraction(_WORDS[t])
    if "/" in t:
        top, bottom = t.split("/")
        return Fraction(int(top), int(bottom))
    return Fraction(t)


def _evaluate(expression: str) -> Fraction:
    tokens = [m.group(1) for m in _TOKEN.finditer(expression)]
    values, ops = [_number(tokens[0])], []
    for op, num in zip(tokens[1::2], tokens[2::2]):
        op = op.lower()
        value = _number(num)
        if op in ("×", "*", "x", "times", "multiplied by"):
            values[-1] *= value
        elif op in ("÷", "/", "divided by"):
            values[-1] /= value
        else:
            ops.append(op)
            values.append(value)
    total = values[0]
    for op, value in zip(ops, values[1:]):
        total = total + value if op in ("+", "plus") else total - value
    return total


def _matches(actual: Fraction, written: str, approx: bool, more: bool) -> bool:
    expected = _number(written)
    if actual == expected:
        return True
    if (approx or more) and "." in written:  # 1/3 = 0.333... or "about 0.33": allow rounding/cut-off.
        places = len(written.split(".")[1])
        step = Fraction(1, 10 ** places)
        return abs(actual - expected) < step
    return False


def problems_in_text(text: str) -> list[str]:
    found = []
    for m in _STATEMENT.finditer(text or ""):
        statement = m.group(0)
        if not _HAS_OP.search(statement):
            continue  # "5 is 5" or "page 3 is 4 lines" — nothing to work out.
        body = statement[: m.start("more") - m.start()] if m.group("more") else statement
        pieces = _SPLIT.split(body)
        parts, approx_flags = pieces[0::2], [False] + [bool(re.search(_APPROX, s, re.I)) for s in pieces[1::2]]
        try:
            first = _evaluate(parts[0])
            for i, (part, approx) in enumerate(zip(parts[1:], approx_flags[1:]), start=1):
                single = _TOKEN.fullmatch(part.strip()) is not None and re.fullmatch(_NUM, part.strip(), re.I)
                last = i == len(parts) - 1
                ok = (_matches(first, part.strip(), approx, bool(m.group("more")) and last) if single
                      else _evaluate(part) == first)
                if not ok:
                    exact = f"{first.numerator}/{first.denominator}" if first.denominator != 1 else str(first.numerator)
                    about = f" (about {float(first):.4g})" if first.denominator != 1 else ""
                    found.append(f'"{statement.strip()}" is wrong: {parts[0].strip()} is {exact}{about}')
                    break
        except (ZeroDivisionError, ValueError, IndexError):
            continue
    return found


_MATH_WORDS = re.compile(
    r"\b(maths?|mathematics|arithmetic|fractions?|decimals?|percent(age)?s?|%|multipl(y|ying|ication|ied)|"
    r"divid(e|ing|ed)|division|subtract(ing|ion)?|addition|adding|sums?|equations?|algebra|geometry|"
    r"angles?|perimeter|area of|volume of|square roots?|prime numbers?|even numbers?|odd numbers?|"
    r"times tables?|place value|number line|long division|denominators?|numerators?|ratios?|averages?|"
    r"probability|negative numbers?|exponents?|squared|cubed|pythagoras|pi\b|integers?|factors?|"
    r"multiples?|remainders?|greatest common|least common|lcm|gcd|calculat(e|ing|ion)|"
    r"how many (sides|corners|degrees|minutes in|seconds in|hours in|days in|weeks in|months in|inches|centimet|millimet|grams in|ounces in))\b",
    re.IGNORECASE,
)
_MATH_EXPRESSION = re.compile(rf"\d\s*(?:\+|-|−|×|\*|÷|/|=|\^)\s*\d|{_NUM}\s+(?:plus|minus|times|divided by)\s+{_NUM}", re.IGNORECASE)


def is_math_question(question: str) -> bool:
    """True when a child's question is about maths, so it can go to the stronger model."""
    q = question or ""
    # "2x2 Rubik's cube", "3*3 grid": a size, not a sum.
    q = re.sub(r"\b\d\s*[x×*]\s*\d(?=\s*(?:rubik|cube|grid|board|puzzle))", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\b\d\s*[x×*]\s*\d\b(?=.*\b(?:rubik|cube)\b)", " ", q, flags=re.IGNORECASE)
    return bool(_MATH_EXPRESSION.search(q) or _MATH_WORDS.search(q))


def check_lesson(lesson: dict) -> list[str]:
    """Every arithmetic mistake found in the lesson's visible text and narration, labelled by slide."""
    problems = [f"Title: {p}" for p in problems_in_text(lesson.get("title", ""))]
    for i, slide in enumerate(lesson.get("slides", []), start=1):
        seen = set()
        for field in _TEXT_FIELDS:
            for p in problems_in_text(slide.get(field, "")):
                if p not in seen:
                    seen.add(p)
                    problems.append(f"Slide {i}: {p}")
    return problems
