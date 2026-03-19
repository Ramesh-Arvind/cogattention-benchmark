"""
Base module for CogAttention procedural task generation.
Shared entity pools, utilities, and seed management.
"""

import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ─── Entity Pools ───────────────────────────────────────────────────
# Diverse, uncommon names to minimize training-data overlap
FIRST_NAMES = [
    "Amara", "Bashir", "Colette", "Dmitri", "Elara", "Femi", "Greta",
    "Haruto", "Ines", "Joaquin", "Kaia", "Leif", "Maren", "Nico",
    "Olena", "Priya", "Qadir", "Runa", "Soren", "Tala", "Ugo",
    "Vesna", "Wren", "Xander", "Yara", "Zain", "Adaeze", "Bram",
    "Celine", "Dariush", "Elio", "Freya", "Gael", "Hana", "Idris",
    "Joelle", "Kenji", "Lumi", "Magnus", "Nalini", "Orla", "Paloma",
    "Ravi", "Sigrid", "Tariq", "Uma", "Viktor", "Willa", "Yuki", "Zora",
]

COLORS = [
    "red", "blue", "green", "silver", "golden", "amber", "ivory",
    "copper", "jade", "coral", "crimson", "cobalt", "ochre", "slate",
    "teal", "bronze", "indigo", "pearl", "russet", "onyx",
]

OBJECTS = [
    "key", "book", "coin", "compass", "lantern", "ring", "flask",
    "scroll", "pendant", "dagger", "mask", "chalice", "mirror",
    "bell", "feather", "stone", "shell", "candle", "locket", "dice",
]

CITIES = [
    "Valetta", "Kumasi", "Trieste", "Plovdiv", "Oulu", "Tbilisi",
    "Cartagena", "Gdansk", "Mandalay", "Recife", "Ulaanbaatar",
    "Cusco", "Bruges", "Tallinn", "Zanzibar", "Luang Prabang",
    "Kotor", "Jaipur", "Reykjavik", "Fez",
]

ANIMALS = [
    "eagle", "fox", "heron", "lynx", "otter", "falcon", "ibis",
    "marten", "osprey", "badger", "crane", "ermine", "gazelle",
    "jackal", "kingfisher", "lemur", "newt", "puffin", "quail", "stoat",
]

INSTRUMENTS = [
    "violin", "flute", "cello", "harp", "oboe", "lute", "tabla",
    "sitar", "dulcimer", "zither", "bassoon", "mandolin", "timpani",
    "balalaika", "theremin", "hurdy-gurdy", "mbira", "koto", "erhu", "oud",
]

METALS = [
    "iron", "copper", "tin", "zinc", "lead", "nickel", "cobalt",
    "tungsten", "titanium", "chromium", "manganese", "vanadium",
    "molybdenum", "platinum", "palladium", "iridium", "osmium",
    "rhodium", "ruthenium", "niobium",
]

CATEGORY_POOLS = {
    "birds": ["eagle", "heron", "falcon", "ibis", "osprey", "crane",
              "puffin", "quail", "kingfisher", "robin", "wren", "finch",
              "sparrow", "magpie", "starling", "swift", "tern", "dove",
              "raven", "woodpecker"],
    "near_birds": ["bat", "butterfly", "dragonfly", "flying squirrel",
                   "flying fish", "moth", "beetle", "wasp", "pterodactyl",
                   "sugar glider"],
    "metals": METALS,
    "near_metals": ["glass", "ceramic", "wood", "plastic", "rubber",
                    "concrete", "granite", "marble", "chalk", "sand"],
    "instruments": INSTRUMENTS,
    "near_instruments": ["microphone", "speaker", "metronome", "music stand",
                         "tuning fork", "pitch pipe", "amplifier", "mixer",
                         "headphones", "record player"],
    "rivers": ["Danube", "Mekong", "Tigris", "Volga", "Zambezi", "Loire",
               "Indus", "Nile", "Rhine", "Ganges", "Amazon", "Yangtze",
               "Mississippi", "Congo", "Murray", "Euphrates", "Don",
               "Elbe", "Oder", "Tagus"],
    "near_rivers": ["Lake Baikal", "Caspian Sea", "Dead Sea", "Lake Victoria",
                    "Strait of Gibraltar", "Panama Canal", "Suez Canal",
                    "Bay of Bengal", "Black Sea", "Aral Sea"],
}

# ─── Filler paragraph templates ─────────────────────────────────────
FILLER_TEMPLATES = [
    "The market square in {city} was busier than usual that morning. Vendors had set up their stalls before dawn, and the smell of roasted chestnuts mixed with diesel fumes from the delivery trucks.",
    "A thin rain began to fall just as {name} reached the old quarter. The cobblestones gleamed under the streetlights, and a cat watched from a window ledge with studied indifference.",
    "The workshop on {street} had been there for decades, its walls darkened by time and soot. Inside, the hum of machinery was constant, a low vibration you felt in your teeth.",
    "Trade negotiations between the two districts had stalled over a minor tariff dispute. The delegates from {city} insisted on maintaining their position, while the merchants grew impatient.",
    "The library held over thirty thousand volumes, most of them catalogued by hand in a system that only {name} fully understood. The wooden shelves bowed slightly under the weight.",
    "Fog rolled in from the harbour, thick enough to muffle the sound of the bells. Ships sat motionless at their moorings, waiting for the visibility to improve before attempting the channel.",
    "The annual inspection of the bridge supports revealed nothing unusual. The engineers noted minor erosion along the northern pylons but deemed it within acceptable parameters.",
    "A street musician played something melancholy on a worn accordion. The tune drifted through the alley, mixing with the clatter of dishes from a restaurant kitchen.",
    "The old postal route between {city} and the coastal villages had not been used in years. Weeds pushed through the gravel, and the mile markers were barely legible.",
    "Construction on the new civic building proceeded on schedule despite the weather. The foreman reviewed the blueprints each morning, marking progress with a red pencil.",
    "The clock tower had been silent for three months while repairs were made to the mechanism. Residents had grown accustomed to the quiet and were divided on whether to restore it.",
    "A shipment of textiles arrived from the south, packed in crates stamped with unfamiliar markings. The customs officer consulted her reference manual before clearing them.",
    "The botanical garden maintained a collection of over four hundred species, each labelled with its Latin name and region of origin. A narrow gravel path wound between the beds.",
    "Evening fell quickly in the valley. The mountains blocked the last hour of sunlight, and by five o'clock the streetlights were already on.",
    "The ferry crossed the strait twice daily, weather permitting. On calm days the journey took forty minutes; in rough seas it could take over an hour.",
]


@dataclass
class TaskInstance:
    """A single benchmark item with all metadata."""
    task_id: str
    task_type: str
    difficulty: str
    prompt: str
    gold_answer: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    canary: str = ""

    def __post_init__(self):
        # Generate canary string for contamination detection
        h = hashlib.md5(f"{self.task_id}-{self.prompt[:100]}".encode()).hexdigest()[:12]
        self.canary = f"<!-- CogAttention-v1-{h} -->"


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    # numpy seed set separately if needed


def sample_unique(pool: list, n: int, rng: random.Random = None) -> list:
    """Sample n unique items from pool."""
    if rng:
        return rng.sample(pool, min(n, len(pool)))
    return random.sample(pool, min(n, len(pool)))


def make_item_descriptor(rng: random.Random = None) -> str:
    """Generate a unique [color] [object] descriptor."""
    c = rng.choice(COLORS) if rng else random.choice(COLORS)
    o = rng.choice(OBJECTS) if rng else random.choice(OBJECTS)
    return f"{c} {o}"


def generate_filler_paragraph(rng: random.Random = None) -> str:
    """Generate a single filler paragraph with randomized entities."""
    r = rng if rng else random
    template = r.choice(FILLER_TEMPLATES)
    return template.format(
        city=r.choice(CITIES),
        name=r.choice(FIRST_NAMES),
        street=f"{r.choice(FIRST_NAMES)} Street",
    )


DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard", "Expert", "Frontier"]
