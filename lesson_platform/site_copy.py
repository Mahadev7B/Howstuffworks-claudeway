"""Approved homepage copy and the 13 library descriptions.

Descriptions for these 13 lessons are not stored in repo JSON. The lesson
queue lists them among the 30 catalog lessons, and the private-lesson builder
fills a missing summary with "A four-slide private lesson about …". That text
is the published_lessons.summary column, which the site uses as the meta
description, the social description, and the Explore card text.

The startup update changes only that column, only these slugs, and only while
the current value still starts with the placeholder.
"""

from dataclasses import dataclass


PLACEHOLDER_PREFIX = "A four-slide private lesson about"

HOME_TITLE = "Ask Lil Owl — Illustrated lessons for curious kids"
HOME_DESCRIPTION = (
    "How-it-works lessons, illustrated and narrated. The 30-lesson library is free to watch; "
    "families get 3 free custom lessons. Subscriptions aren't open yet."
)
HOME_H1 = "Ask how it works, and watch a short lesson together"
BRAND_IMAGE_PATH = "/static/brand/owl-cutout.png"


@dataclass(frozen=True)
class LibraryDescription:
    slug: str
    title: str
    summary: str


# Titles match the catalog, including U+2019 in "India’s Amazing Languages".
LIBRARY_DESCRIPTIONS: tuple[LibraryDescription, ...] = (
    LibraryDescription(
        "what-is-a-dosa",
        "What Is a Dosa?",
        "Meet a thin, savory food often made from rice and lentil batter, cooked on a hot pan, and enjoyed plain or with fillings, chutney, or sambar.",
    ),
    LibraryDescription(
        "what-is-taekwondo",
        "What Is Taekwondo?",
        "Meet a martial art that developed in Korea, with kicks, blocks, and patterns, colored belts to mark progress, and training that builds control and respect.",
    ),
    LibraryDescription(
        "why-do-owls-say-hoo",
        "Why Do Owls Say Hoo?",
        "Hear why owls hoot, whistle, or screech, how those calls help them communicate, and how eyes, ears, and quiet flight help them hunt when the light is low.",
    ),
    LibraryDescription(
        "why-do-flowers-exist",
        "Why Do Flowers Exist?",
        "See how a flower's colors and scents help attract animals that carry pollen, how a fertilized ovule becomes a seed, and how fruit can carry that seed.",
    ),
    LibraryDescription(
        "what-makes-pigeons-special",
        "What Makes Pigeons Special?",
        "Look closer at a familiar city bird, a relative of the wild rock dove, and see how pigeons find their way home using more than one clue.",
    ),
    LibraryDescription(
        "how-do-aeroplanes-fly",
        "How Do Aeroplanes Fly?",
        "Follow the four forces of flight—weight, lift, thrust, and drag—and see how wings, engines, and control surfaces help an aeroplane take off and turn.",
    ),
    LibraryDescription(
        "how-do-cars-work",
        "How Do Cars Work?",
        "See how a petrol, diesel, or electric car turns its wheels, how steering changes direction, and how brakes and tire grip help the car slow and stop.",
    ),
    LibraryDescription(
        "how-electric-cars-work",
        "How Electric Cars Work",
        "Follow energy from battery to motor, see how plugging in recharges it, how slowing recovers some energy, and why no exhaust is not the same as no impact.",
    ),
    LibraryDescription(
        "how-self-driving-cars-see",
        "How Self-Driving Cars See",
        "See how cameras and other sensors help a driving system gather clues, decide, and react—and why many cars still need a person who watches the road.",
    ),
    LibraryDescription(
        "how-minecraft-was-created",
        "How Minecraft Was Created",
        "Follow how Minecraft began as a world of blocks players could change, grew from a 2009 version, and became a sandbox for building, exploring, and sharing.",
    ),
    LibraryDescription(
        "how-reading-and-writing-began",
        "How Reading and Writing Began",
        "See how marks on clay in ancient Mesopotamia became cuneiform, how other writing systems developed, and how writing can leave a message across time.",
    ),
    LibraryDescription(
        "indias-amazing-languages",
        "India’s Amazing Languages",
        "Hear how many languages are spoken across India, what the Constitution's list of 22 includes and leaves out, and how different scripts write them.",
    ),
    LibraryDescription(
        "the-biggest-countries-on-earth",
        "The Biggest Countries on Earth",
        "See what “biggest country” means by land area, meet Russia and other large countries, and why a list can change if water inside borders is counted too.",
    ),
)

DESCRIPTIONS_BY_SLUG = {item.slug: item for item in LIBRARY_DESCRIPTIONS}

_VALUE_ROWS = ",".join(["(%s::text,%s::text)"] * len(LIBRARY_DESCRIPTIONS))
PLACEHOLDER_UPDATE_SQL = (
    "UPDATE published_lessons AS p "
    "SET summary = v.summary "
    f"FROM (VALUES {_VALUE_ROWS}) AS v(slug, summary) "
    "WHERE p.slug = v.slug "
    "AND p.summary LIKE %s"
)


def placeholder_update_params() -> tuple:
    params: list[str] = []
    for item in LIBRARY_DESCRIPTIONS:
        params.append(item.slug)
        params.append(item.summary)
    params.append(PLACEHOLDER_PREFIX + "%")
    return tuple(params)


def replacement_summary(slug: str, current: str) -> str | None:
    """New summary when this slug is one of the 13 and current is still the placeholder."""
    item = DESCRIPTIONS_BY_SLUG.get(slug)
    if item is None or not isinstance(current, str) or not current.startswith(PLACEHOLDER_PREFIX):
        return None
    return item.summary


def apply_placeholder_summaries(cur) -> int:
    """UPDATE summary only, for these 13 slugs, and only while the placeholder remains."""
    cur.execute(PLACEHOLDER_UPDATE_SQL, placeholder_update_params())
    return cur.rowcount
