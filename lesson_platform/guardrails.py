"""Topic guardrails for incoming questions.

Two layers:
1. Hard-block: explicit/body-part content — never reaches Claude.
2. Soft-check: question must relate to science, technology, nature, math,
   history, inventions, everyday objects, or how the world works.
   Unrelated questions (celebrity gossip, gambling, politics, etc.) get a friendly redirect.
"""
import re

# ---------------------------------------------------------------------------
# Hard-block list — explicit/inappropriate content for kids
# Keep words lowercase; matching is case-insensitive.
# ---------------------------------------------------------------------------
_HARD_BLOCK = {
    "penis", "vagina", "vulva", "anus", "rectum", "testicle", "testicles",
    "scrotum", "clitoris", "uterus", "ovary", "ovaries",
    "sex", "sexual", "sexually", "intercourse", "masturbat",
    "porn", "pornography", "pornographic", "nude", "naked", "nudity",
    "breast", "nipple", "genitals", "genital",
    "erection", "ejaculat", "orgasm", "condom", "contraception",
    "rape", "molest", "abuse", "pedophil",
    "fuck", "shit", "bitch", "cunt", "cock", "dick", "ass", "asshole",
}

# ---------------------------------------------------------------------------
# Science / tech / nature / math — presence of ANY of these passes the filter
# ---------------------------------------------------------------------------
_ALLOWED_TOPICS = {
    # Science general
    "science", "scientist", "experiment", "laboratory", "research",
    # Physics
    "physics", "force", "gravity", "energy", "motion", "speed", "velocity",
    "acceleration", "friction", "momentum", "inertia", "pressure", "heat",
    "temperature", "thermodynamics", "electricity", "magnetism", "magnetic",
    "light", "sound", "wave", "radiation", "nuclear", "atom", "molecule",
    "quantum", "relativity", "optics", "lens", "prism", "refraction",
    # Chemistry
    "chemistry", "chemical", "reaction", "element", "compound", "acid",
    "base", "ph", "periodic", "ion", "bond", "molecule", "gas", "liquid",
    "solid", "plasma", "dissolve", "solution", "mixture",
    # Biology / nature
    "biology", "cell", "dna", "gene", "genetics", "evolution", "species",
    "animal", "plant", "fungus", "bacteria", "virus", "immune", "body",
    "brain", "heart", "lung", "blood", "muscle", "bone", "organ", "digest",
    "photosynthesis", "ecosystem", "food chain", "habitat", "predator",
    "prey", "mammal", "reptile", "insect", "bird", "fish", "amphibian",
    "flower", "seed", "root", "leaf", "leaves", "stem", "bark", "tree", "forest", "ocean", "river",
    "lake", "mountain", "volcano", "earthquake", "dinosaur", "fossil",
    "microbe", "germ", "vaccine", "antibiotic",
    # Earth & space
    "earth", "planet", "solar system", "sun", "moon", "star", "galaxy",
    "universe", "space", "orbit", "gravity", "atmosphere", "weather",
    "climate", "cloud", "rain", "snow", "wind", "tornado", "hurricane",
    "tsunami", "tectonic", "geology", "mineral", "rock", "crystal",
    "comet", "asteroid", "black hole", "nebula", "supernova",
    # Technology & engineering
    "technology", "engineer", "machine", "computer", "robot", "code",
    "program", "software", "hardware", "internet", "network", "satellite",
    "rocket", "airplane", "plane", "helicopter", "car", "engine", "motor",
    "battery", "solar panel", "wind turbine", "electricity", "circuit",
    "sensor", "camera", "laser", "radar", "sonar", "3d print", "bridge",
    "dam", "skyscraper", "elevator", "submarine", "spacecraft", "drone",
    "artificial intelligence", "ai", "machine learning",
    # Math
    "math", "mathematics", "number", "algebra", "geometry", "calculus",
    "fraction", "decimal", "percent", "equation", "graph", "triangle",
    "circle", "sphere", "cube", "symmetry", "prime", "fibonacci",
    "probability", "statistics", "pattern", "sequence",
    # Common "how does X work" subjects that are on-topic
    "how do", "how does", "why is", "why do", "why does",
    "what is", "what are", "what makes",
    "rainbow", "lightning", "thunder", "fire", "flame", "ice", "water",
    "color", "colour", "green", "blue", "red", "yellow", "white", "black",
    "hot", "cold", "dark", "bright", "glow", "shine", "reflect", "absorb",
    "grow", "growth", "alive", "living", "life", "nature", "natural",
    "sky", "ground", "soil", "air", "oxygen", "carbon", "nitrogen",
    "sugar", "starch", "protein", "fat", "vitamin", "mineral",
    "magnet", "compass", "telescope", "microscope", "x-ray", "mri",
    "wifi", "bluetooth", "gps", "microwave", "refrigerator",
    "vaccine", "medicine", "doctor",
    # History of science and inventions
    "invention", "inventor", "invent", "invented", "discover", "discovery",
    "newton", "einstein", "darwin", "curie", "galileo", "tesla", "edison",
    "history", "historical", "ancient", "origin", "origins", "come from",
    "where did", "who made", "who invented", "how was", "how were",
    "when was", "when were", "why did", "why do people",
    # Everyday objects and life
    "shoe", "shoes", "pencil", "pen", "paper", "glass", "bread", "food",
    "book", "wheel", "clock", "watch", "chair", "table", "phone", "telephone",
    "bicycle", "bike", "train", "ship", "boat", "umbrella", "hat", "clothes",
    "clothing", "money", "coin", "school", "toy", "toys", "map", "flag",
    "house", "building", "road", "bridge", "tool", "tools", "knife", "fork",
    "spoon", "cup", "bowl", "door", "window", "wall", "key", "lock",
    "button", "zipper", "mirror", "candle", "lamp", "light bulb", "radio",
    "television", "tv", "movie", "music", "instrument", "piano", "guitar",
    "sport", "sports", "ball", "game", "games", "playground",
    # Common question starters — allow broad child questions through
    "how did", "how do people", "what makes", "where does", "where do",
    "who was", "who were", "tell me", "explain",
}

# Stopwords that don't help classify — strip before checking allowed topics
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "but", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "its", "it", "i", "my", "me", "we",
    "our", "you", "your",
}

_HARD_BLOCK_MSG = (
    "This app is for science and technology questions for kids. "
    "That topic isn't something we cover here."
)

_OFFTOPIC_MSG = (
    "This app explores science, nature, history, inventions, and how the world works. "
    "Try asking something like \"How do rockets fly?\", \"Why is the sky blue?\", "
    "or \"How was the wheel invented?\""
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def check_question(question: str) -> str | None:
    """Return an error message if the question should be blocked, else None.

    None means the question is allowed through.
    """
    q_lower = question.lower()
    tokens = _tokenize(q_lower)

    # Layer 1 — hard block: substring match for explicit terms
    for term in _HARD_BLOCK:
        if term in q_lower:
            return _HARD_BLOCK_MSG

    # Layer 2 — soft topic check
    # Build a set of meaningful words + bigrams from the question
    words = set(tokens) - _STOPWORDS
    bigrams = {tokens[i] + " " + tokens[i + 1] for i in range(len(tokens) - 1)}
    candidates = words | bigrams

    # Check if any allowed topic keyword appears in the question
    for topic in _ALLOWED_TOPICS:
        if topic in q_lower or topic in candidates:
            return None  # on-topic, allow

    # Nothing matched — off-topic redirect
    return _OFFTOPIC_MSG
