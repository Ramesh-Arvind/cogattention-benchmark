"""
Task B-Novel: Interleaved Stream Segregation (Sustained + Selective Attention)

NOVEL CONTRIBUTION: No existing LLM benchmark tests word-level stream segregation.
Adapts the Dichotic Listening / Cocktail Party paradigm for text.

Two conversations are interleaved at sentence level. The model must follow
only one stream and answer questions about it, ignoring the other.
Includes a "breakthrough" detection variant (detecting a keyword in the ignored stream).
"""

import random
from typing import List
from .base import TaskInstance, DIFFICULTY_LEVELS, FIRST_NAMES

ITEMS_PER_DIFFICULTY = 8

# Topic pairs for the two streams (deliberately different domains)
TOPIC_PAIRS = [
    ("cooking_recipe", "travel_plan"),
    ("repair_instructions", "event_planning"),
    ("gardening_guide", "financial_discussion"),
    ("sports_commentary", "medical_advice"),
]

# Topic triples for 3-stream Frontier difficulty
TOPIC_TRIPLES = [
    ("cooking_recipe", "travel_plan", "financial_discussion"),
    ("repair_instructions", "event_planning", "medical_advice"),
    ("gardening_guide", "sports_commentary", "travel_plan"),
    ("cooking_recipe", "repair_instructions", "medical_advice"),
]

# Sentence generators for each topic
TOPIC_SENTENCES = {
    "cooking_recipe": [
        "First, preheat the oven to {temp} degrees.",
        "Dice the {vegetable} into small cubes.",
        "Add {amount} tablespoons of olive oil to the pan.",
        "Season with salt, pepper, and a pinch of {spice}.",
        "Let the mixture simmer for {minutes} minutes.",
        "Stir occasionally until the sauce thickens.",
        "Remove from heat and let it cool for {minutes} minutes.",
        "Garnish with fresh {herb} before serving.",
        "The total cooking time should be about {total_minutes} minutes.",
        "Serve on a warm plate alongside {side_dish}.",
    ],
    "travel_plan": [
        "The flight departs at {time} from terminal {terminal}.",
        "Book a hotel near {landmark} for the best location.",
        "The train from the airport takes about {minutes} minutes.",
        "Budget approximately ${budget} per day for meals.",
        "The museum on {street} is open until {closing_time}.",
        "Pack {item} — the weather forecast shows {weather}.",
        "The rental car pickup is at {location}.",
        "Exchange currency at the airport — the rate is {rate} to the dollar.",
        "The guided tour starts at {tour_time} near the main square.",
        "Check out is at {checkout_time} — leave bags at reception.",
    ],
    "repair_instructions": [
        "First, disconnect the power supply completely.",
        "Remove the {n_screws} screws from the back panel.",
        "Locate the {component} — it should be near the {location}.",
        "Use a {tool_size}mm wrench to loosen the bolt.",
        "Replace the worn {part} with the new one from the kit.",
        "Apply {adhesive} to both surfaces before joining.",
        "Let the joint set for at least {hours} hours.",
        "Reattach the panel and tighten screws to {torque} Nm.",
        "Test the operation before restoring power.",
        "If the issue persists, check the {secondary_part}.",
    ],
    "event_planning": [
        "The venue holds up to {capacity} guests.",
        "Catering quotes range from ${low} to ${high} per person.",
        "The band can play from {start_time} to {end_time}.",
        "Reserve {n_tables} round tables with {chairs} chairs each.",
        "Flowers should arrive by {delivery_time} on the day.",
        "The photographer charges ${photo_rate} per hour.",
        "Send invitations at least {weeks} weeks in advance.",
        "The cake needs to be ordered {days} days ahead.",
        "Parking is available for {n_cars} vehicles.",
        "Set up begins at {setup_time} — the venue opens at {open_time}.",
    ],
    "gardening_guide": [
        "Plant the {plant} seeds {depth} inches deep.",
        "Water thoroughly every {days} days during {season}.",
        "The soil pH should be between {ph_low} and {ph_high}.",
        "Add {fertilizer} fertilizer once every {weeks} weeks.",
        "Prune the {plant} back to {inches} inches in {month}.",
        "Space each plant at least {spacing} inches apart.",
        "Expect germination in {germ_days} to {germ_days_high} days.",
        "Harvest when the {plant} reaches {height} inches tall.",
        "Mulch with {mulch_type} to retain moisture.",
        "Watch for {pest} — treat with {treatment} if spotted.",
    ],
    "financial_discussion": [
        "The quarterly revenue increased by {percent}% year-over-year.",
        "Operating costs are projected at ${amount} million.",
        "The debt-to-equity ratio stands at {ratio}.",
        "Dividends per share will be ${dividend}.",
        "Market capitalization reached ${market_cap} billion.",
        "The board approved a ${buyback} million share buyback.",
        "Revenue from the {region} region grew {growth}%.",
        "Net profit margin improved to {margin}%.",
        "Capital expenditure is budgeted at ${capex} million.",
        "The stock trades at a P/E ratio of {pe}.",
    ],
    "sports_commentary": [
        "The score is {home}-{away} at the end of the {period}.",
        "{player} scored from {distance} yards out.",
        "The referee issued a {card} card for the foul.",
        "Possession has been {home_poss}%-{away_poss}% so far.",
        "Substitution: {sub_in} replaces {sub_out}.",
        "The attendance tonight is {attendance} spectators.",
        "Injury time will be {added_time} minutes.",
        "The corner kick is taken by {player}.",
        "{player} makes a save from close range.",
        "The match has been played in {weather} conditions.",
    ],
    "medical_advice": [
        "Take {dosage}mg of {medication} twice daily.",
        "The follow-up appointment is in {weeks} weeks.",
        "Blood pressure reading was {systolic}/{diastolic}.",
        "Limit sodium intake to {sodium}mg per day.",
        "The recommended daily water intake is {liters} liters.",
        "Avoid {food} for at least {days} days post-procedure.",
        "Exercise for at least {minutes} minutes daily.",
        "The test results will be available in {days} business days.",
        "Apply the {cream} cream {times} times per day.",
        "Schedule a follow-up if symptoms persist beyond {days} days.",
    ],
}


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill a template with random plausible values."""
    replacements = {
        "{temp}": str(rng.randint(150, 450)),
        "{vegetable}": rng.choice(["onion", "carrot", "celery", "pepper", "zucchini"]),
        "{amount}": str(rng.randint(1, 4)),
        "{spice}": rng.choice(["cumin", "paprika", "turmeric", "oregano"]),
        "{minutes}": str(rng.randint(5, 45)),
        "{total_minutes}": str(rng.randint(30, 90)),
        "{herb}": rng.choice(["basil", "parsley", "cilantro", "dill"]),
        "{side_dish}": rng.choice(["rice", "bread", "salad", "potatoes"]),
        "{time}": f"{rng.randint(6,21)}:{rng.choice(['00','15','30','45'])}",
        "{terminal}": str(rng.randint(1, 5)),
        "{landmark}": rng.choice(["the cathedral", "the old market", "the central park"]),
        "{budget}": str(rng.randint(30, 150)),
        "{street}": f"{rng.choice(FIRST_NAMES)} Street",
        "{closing_time}": f"{rng.randint(17,21)}:00",
        "{item}": rng.choice(["an umbrella", "sunscreen", "a warm jacket"]),
        "{weather}": rng.choice(["rain", "sunshine", "cold winds"]),
        "{location}": rng.choice(["the east exit", "gate B", "the main lobby"]),
        "{rate}": f"{rng.uniform(0.5, 1.5):.2f}",
        "{tour_time}": f"{rng.randint(9,14)}:00",
        "{checkout_time}": f"{rng.randint(10,12)}:00",
        "{n_screws}": str(rng.randint(2, 8)),
        "{component}": rng.choice(["capacitor", "relay switch", "thermal fuse"]),
        "{tool_size}": str(rng.choice([6, 8, 10, 12])),
        "{part}": rng.choice(["gasket", "bearing", "filter", "seal"]),
        "{adhesive}": rng.choice(["epoxy", "silicone", "contact cement"]),
        "{hours}": str(rng.randint(2, 24)),
        "{torque}": str(rng.randint(5, 25)),
        "{secondary_part}": rng.choice(["control board", "wiring harness", "sensor"]),
        "{capacity}": str(rng.randint(50, 300)),
        "{low}": str(rng.randint(25, 50)),
        "{high}": str(rng.randint(60, 120)),
        "{start_time}": f"{rng.randint(17,19)}:00",
        "{end_time}": f"{rng.randint(22,23)}:00",
        "{n_tables}": str(rng.randint(5, 30)),
        "{chairs}": str(rng.choice([6, 8, 10])),
        "{delivery_time}": f"{rng.randint(8,11)}:00",
        "{photo_rate}": str(rng.randint(100, 400)),
        "{weeks}": str(rng.randint(2, 8)),
        "{days}": str(rng.randint(2, 14)),
        "{n_cars}": str(rng.randint(20, 100)),
        "{setup_time}": f"{rng.randint(8,12)}:00",
        "{open_time}": f"{rng.randint(14,17)}:00",
        "{plant}": rng.choice(["tomato", "basil", "lettuce", "sunflower"]),
        "{depth}": str(rng.choice([0.25, 0.5, 1, 2])),
        "{season}": rng.choice(["spring", "summer", "autumn"]),
        "{ph_low}": f"{rng.uniform(5.5, 6.5):.1f}",
        "{ph_high}": f"{rng.uniform(6.5, 7.5):.1f}",
        "{fertilizer}": rng.choice(["nitrogen-rich", "balanced", "potassium"]),
        "{inches}": str(rng.randint(4, 18)),
        "{month}": rng.choice(["March", "April", "October"]),
        "{spacing}": str(rng.randint(6, 24)),
        "{germ_days}": str(rng.randint(5, 10)),
        "{germ_days_high}": str(rng.randint(12, 21)),
        "{height}": str(rng.randint(6, 36)),
        "{mulch_type}": rng.choice(["straw", "wood chips", "leaf compost"]),
        "{pest}": rng.choice(["aphids", "slugs", "caterpillars"]),
        "{treatment}": rng.choice(["neem oil", "diatomaceous earth", "insecticidal soap"]),
        "{percent}": str(rng.randint(2, 25)),
        "{amount}": str(rng.randint(10, 500)),
        "{ratio}": f"{rng.uniform(0.3, 2.5):.2f}",
        "{dividend}": f"{rng.uniform(0.5, 5.0):.2f}",
        "{market_cap}": str(rng.randint(5, 500)),
        "{buyback}": str(rng.randint(50, 500)),
        "{region}": rng.choice(["Asia-Pacific", "European", "North American"]),
        "{growth}": str(rng.randint(3, 30)),
        "{margin}": f"{rng.uniform(5, 25):.1f}",
        "{capex}": str(rng.randint(20, 200)),
        "{pe}": f"{rng.uniform(8, 35):.1f}",
        "{home}": str(rng.randint(0, 4)),
        "{away}": str(rng.randint(0, 4)),
        "{period}": rng.choice(["first half", "second half", "third quarter"]),
        "{player}": rng.choice(FIRST_NAMES),
        "{distance}": str(rng.randint(5, 35)),
        "{card}": rng.choice(["yellow", "red"]),
        "{home_poss}": str(rng.randint(35, 65)),
        "{away_poss}": str(100 - rng.randint(35, 65)),
        "{sub_in}": rng.choice(FIRST_NAMES),
        "{sub_out}": rng.choice(FIRST_NAMES),
        "{attendance}": f"{rng.randint(10, 80)},{rng.randint(100, 999)}",
        "{added_time}": str(rng.randint(1, 5)),
        "{dosage}": str(rng.choice([100, 200, 250, 500])),
        "{medication}": rng.choice(["amoxicillin", "ibuprofen", "metformin"]),
        "{systolic}": str(rng.randint(110, 160)),
        "{diastolic}": str(rng.randint(60, 100)),
        "{sodium}": str(rng.randint(1500, 2500)),
        "{liters}": f"{rng.uniform(1.5, 3.0):.1f}",
        "{food}": rng.choice(["dairy", "gluten", "alcohol", "caffeine"]),
        "{cream}": rng.choice(["hydrocortisone", "antibiotic", "moisturizing"]),
        "{times}": str(rng.choice([2, 3])),
    }
    result = template
    for key, val in replacements.items():
        result = result.replace(key, val, 1)
    return result


def _generate_stream(topic: str, n_sentences: int, rng: random.Random) -> List[str]:
    """Generate n sentences for a topic stream."""
    templates = TOPIC_SENTENCES.get(topic, TOPIC_SENTENCES["cooking_recipe"])
    selected = [rng.choice(templates) for _ in range(n_sentences)]
    return [_fill_template(t, rng) for t in selected]


DIFFICULTY_CONFIG_STREAM = {
    "Easy":   {"n_sentences": 4, "has_breakthrough": False},
    "Medium": {"n_sentences": 6, "has_breakthrough": False},
    "Hard":   {"n_sentences": 8, "has_breakthrough": True},
    "Expert":   {"n_sentences": 10, "has_breakthrough": True},
    "Frontier": {"n_sentences": 14, "has_breakthrough": True, "n_streams": 3},
}


def generate_stream_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 1049 + DIFFICULTY_LEVELS.index(difficulty) * 2909)
    config = DIFFICULTY_CONFIG_STREAM[difficulty]

    n = config["n_sentences"]
    n_streams = config.get("n_streams", 2)

    if n_streams == 3:
        # 3-stream Frontier mode
        topic_a, topic_b, topic_c = rng.choice(TOPIC_TRIPLES)
        stream_a = _generate_stream(topic_a, n, rng)
        stream_b = _generate_stream(topic_b, n, rng)
        stream_c = _generate_stream(topic_c, n, rng)

        # Insert breakthrough keyword in stream C
        breakthrough_keyword = None
        breakthrough_position = None
        if config["has_breakthrough"]:
            breakthrough_keyword = "ALERT"
            breakthrough_position = rng.randint(2, n - 1)
            stream_c[breakthrough_position] = (
                stream_c[breakthrough_position].rstrip('.') +
                f". ALERT: this is a priority notice."
            )

        # Interleave 3 streams
        interleaved = []
        for i in range(n):
            interleaved.append(f"[A] {stream_a[i]}")
            interleaved.append(f"[B] {stream_b[i]}")
            interleaved.append(f"[C] {stream_c[i]}")

        passage = "\n".join(interleaved)

        import re
        first_number = None
        for sent in stream_a:
            nums = re.findall(r'\b\d+(?:\.\d+)?\b', sent)
            if nums:
                first_number = nums[0]
                break

        question_about_a = f"Based ONLY on conversation A (about {topic_a.replace('_', ' ')}), " \
                           f"what is the first specific number or measurement mentioned?"

        lines = [
            "Below are THREE interleaved conversations marked [A], [B], and [C].",
            f"Answer questions ONLY about conversation A (about {topic_a.replace('_', ' ')}).",
            "Completely ignore conversations B and C.",
            "",
            passage,
            "",
            f"Question 1: {question_about_a}",
        ]
        if config["has_breakthrough"]:
            lines.append(f'Question 2: Was the word "ALERT" mentioned anywhere in conversation C? (yes/no)')

        lines.append("\nANSWER:")
        lines.append("1. [your answer]")
        if config["has_breakthrough"]:
            lines.append("2. [yes/no]")

        prompt = "\n".join(lines)

        gold = {"1": first_number or "unknown"}
        if config["has_breakthrough"]:
            gold["2"] = "yes"

        return TaskInstance(
            task_id=f"stream_{difficulty.lower()}_{instance_idx:03d}",
            task_type="stream_segregation",
            difficulty=difficulty,
            prompt=prompt,
            gold_answer=gold,
            metadata={
                "topic_a": topic_a,
                "topic_b": topic_b,
                "topic_c": topic_c,
                "n_streams": 3,
                "n_sentences_per_stream": n,
                "has_breakthrough": config["has_breakthrough"],
                "breakthrough_keyword": breakthrough_keyword,
                "first_number_in_a": first_number,
                "stream_a_sentences": stream_a,
            },
        )

    # Original 2-stream path
    topic_a, topic_b = rng.choice(TOPIC_PAIRS)

    stream_a = _generate_stream(topic_a, n, rng)
    stream_b = _generate_stream(topic_b, n, rng)

    # Insert breakthrough keyword in stream B if applicable
    breakthrough_keyword = None
    breakthrough_position = None
    if config["has_breakthrough"]:
        breakthrough_keyword = "ALERT"
        breakthrough_position = rng.randint(2, n - 1)
        stream_b[breakthrough_position] = (
            stream_b[breakthrough_position].rstrip('.') +
            f". ALERT: this is a priority notice."
        )

    # Interleave streams
    interleaved = []
    for i in range(n):
        interleaved.append(f"[A] {stream_a[i]}")
        interleaved.append(f"[B] {stream_b[i]}")

    passage = "\n".join(interleaved)

    # Generate question about stream A
    question_about_a = f"Based ONLY on conversation A (about {topic_a.replace('_', ' ')}), " \
                       f"what is the first specific number or measurement mentioned?"
    # Find the first number in stream A
    import re
    first_number = None
    for sent in stream_a:
        nums = re.findall(r'\b\d+(?:\.\d+)?\b', sent)
        if nums:
            first_number = nums[0]
            break

    # Build prompt
    lines = [
        "Below are two interleaved conversations marked [A] and [B].",
        f"Answer questions ONLY about conversation A (about {topic_a.replace('_', ' ')}).",
        "Completely ignore conversation B.",
        "",
        passage,
        "",
        f"Question 1: {question_about_a}",
    ]
    if config["has_breakthrough"]:
        lines.append(f'Question 2: Was the word "ALERT" mentioned anywhere in conversation B? (yes/no)')

    lines.append("\nANSWER:")
    lines.append("1. [your answer]")
    if config["has_breakthrough"]:
        lines.append("2. [yes/no]")

    prompt = "\n".join(lines)

    gold = {"1": first_number or "unknown"}
    if config["has_breakthrough"]:
        gold["2"] = "yes"

    return TaskInstance(
        task_id=f"stream_{difficulty.lower()}_{instance_idx:03d}",
        task_type="stream_segregation",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=gold,
        metadata={
            "topic_a": topic_a,
            "topic_b": topic_b,
            "n_streams": 2,
            "n_sentences_per_stream": n,
            "has_breakthrough": config["has_breakthrough"],
            "breakthrough_keyword": breakthrough_keyword,
            "first_number_in_a": first_number,
            "stream_a_sentences": stream_a,
        },
    )


def generate_stream_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_stream_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_stream_dataset()
    print(f"Generated {len(dataset)} Stream Segregation instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        s = subset[0]
        print(f"  {diff}: {s.metadata['n_sentences_per_stream']} sentences/stream, "
              f"breakthrough={s.metadata['has_breakthrough']}")
        print(f"    Topics: {s.metadata['topic_a']} vs {s.metadata['topic_b']}")
