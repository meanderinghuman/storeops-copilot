"""Readable synthetic product names for M5 item codes (M5 only has codes like HOUSEHOLD_1_118).

Two sources:
- "teacher": the teacher API names each item; results are cached in `store/product_names.json`
  (committed, so every rebuild gets identical names without calling the API again).
- "offline": a deterministic generator from small per-department vocabularies. Used for tests
  and before the teacher API is configured. Offline names are never written to the cache.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CACHE_PATH = Path(__file__).with_name("product_names.json")

DEPT_THEMES = {
    "FOODS_1": "snacks, breakfast and baking",
    "FOODS_2": "dairy, frozen and refrigerated",
    "FOODS_3": "pantry staples, produce and everyday groceries",
    "HOUSEHOLD_1": "cleaning, paper goods and laundry",
    "HOUSEHOLD_2": "kitchenware, storage and home basics",
    "HOBBIES_1": "toys, games and crafts",
    "HOBBIES_2": "seasonal, party and outdoor",
}

# (product, variants). Variants are sizes or pack counts appended to the product.
_VOCAB: dict[str, list[tuple[str, list[str]]]] = {
    "FOODS_1": [
        ("Tortilla chips", ["11 oz", "family size 18 oz"]),
        ("Pretzel twists", ["16 oz", "snack packs 12-pack"]),
        ("Microwave popcorn", ["3-pack", "butter 6-pack", "light 6-pack"]),
        ("Granola bars", ["oats & honey 6-pack", "chocolate chip 12-pack"]),
        ("Rolled oats", ["18 oz", "42 oz"]),
        ("Corn flakes cereal", ["12 oz", "family size 24 oz"]),
        ("Honey nut cereal", ["10.8 oz", "family size 19 oz"]),
        ("Pancake mix", ["32 oz", "buttermilk 2 lb"]),
        ("Maple syrup", ["12 oz", "24 oz"]),
        ("All-purpose flour", ["2 lb", "5 lb"]),
        ("Granulated sugar", ["4 lb", "10 lb"]),
        ("Chocolate chip cookies", ["13 oz", "family size 18 oz"]),
        ("Saltine crackers", ["16 oz", "4-sleeve box"]),
        ("Peanut butter", ["creamy 16 oz", "crunchy 16 oz", "creamy 40 oz"]),
        ("Strawberry jam", ["18 oz", "32 oz"]),
        ("Trail mix", ["10 oz", "26 oz"]),
        ("Mixed nuts", ["lightly salted 10 oz", "unsalted 30 oz"]),
        ("Fruit snacks", ["10-pack", "40-pack"]),
        ("Brownie mix", ["18 oz", "family size 3-pack"]),
        ("Ground coffee", ["medium roast 12 oz", "dark roast 30 oz"]),
    ],
    "FOODS_2": [
        ("Whole milk", ["half gallon", "1 gallon"]),
        ("2% milk", ["half gallon", "1 gallon"]),
        ("Large eggs", ["dozen", "18-count"]),
        ("Salted butter", ["4 sticks", "2 lb"]),
        ("Cheddar cheese", ["shredded 8 oz", "block 16 oz"]),
        ("Mozzarella cheese", ["shredded 16 oz", "string cheese 12-pack"]),
        ("Greek yogurt", ["plain 32 oz", "vanilla 4-pack"]),
        ("Cream cheese", ["8 oz", "whipped 12 oz"]),
        ("Sour cream", ["16 oz", "24 oz"]),
        ("Orange juice", ["52 oz", "no pulp 89 oz"]),
        ("Frozen pizza", ["pepperoni 12 in", "cheese 4-pack"]),
        ("Frozen mixed vegetables", ["12 oz", "32 oz"]),
        ("Frozen broccoli florets", ["12 oz", "steam bags 4-pack"]),
        ("Vanilla ice cream", ["48 oz", "1 gallon"]),
        ("Frozen waffles", ["10-count", "24-count"]),
        ("Chicken nuggets", ["frozen 2 lb", "frozen 5 lb"]),
        ("Fish sticks", ["frozen 24-count", "frozen 44-count"]),
        ("Hummus", ["classic 10 oz", "family size 17 oz"]),
        ("Deli turkey slices", ["9 oz", "16 oz"]),
        ("Coffee creamer", ["French vanilla 32 oz", "original 64 oz"]),
    ],
    "FOODS_3": [
        ("Long grain white rice", ["2 lb", "5 lb", "20 lb"]),
        ("Spaghetti", ["16 oz", "32 oz"]),
        ("Penne pasta", ["16 oz", "32 oz"]),
        ("Pasta sauce", ["marinara 24 oz", "tomato basil 45 oz"]),
        ("Canned black beans", ["15 oz", "15 oz 4-pack"]),
        ("Canned diced tomatoes", ["14.5 oz", "28 oz"]),
        ("Chicken noodle soup", ["10.75 oz", "4-pack"]),
        ("Canned tuna", ["chunk light 5 oz", "4-pack"]),
        ("Vegetable oil", ["48 oz", "1 gallon"]),
        ("Olive oil", ["16.9 oz", "51 oz"]),
        ("Sandwich bread", ["white 20 oz", "whole wheat 24 oz"]),
        ("Flour tortillas", ["10-count", "20-count"]),
        ("Bananas", ["per lb", "bunch 3 lb"]),
        ("Gala apples", ["3 lb bag", "per lb"]),
        ("Russet potatoes", ["5 lb bag", "10 lb bag"]),
        ("Yellow onions", ["3 lb bag", "per lb"]),
        ("Baby carrots", ["16 oz", "32 oz"]),
        ("Romaine hearts", ["3-pack", "6-pack"]),
        ("Bottled water", ["24-pack 16.9 oz", "1 gallon"]),
        ("Cola", ["12-pack cans", "2 liter"]),
        ("Sparkling water", ["lime 8-pack", "unflavored 12-pack"]),
        ("Ground beef 80/20", ["1 lb", "3 lb"]),
        ("Chicken breasts", ["boneless 2.5 lb", "family pack 5 lb"]),
        ("Macaroni and cheese", ["7.25 oz", "5-pack"]),
        ("Ketchup", ["20 oz", "38 oz"]),
        ("Yellow mustard", ["14 oz", "20 oz"]),
        ("Mayonnaise", ["15 oz", "30 oz"]),
        ("Table salt", ["26 oz", "iodized 26 oz"]),
    ],
    "HOUSEHOLD_1": [
        ("Paper towels", ["2-pack", "6-pack", "12-pack"]),
        ("Toilet paper", ["4 double rolls", "12 double rolls", "24 mega rolls"]),
        ("Facial tissues", ["3-pack", "6-pack"]),
        ("Paper napkins", ["250-count", "500-count"]),
        ("Liquid laundry detergent", ["50 oz", "100 oz"]),
        ("Laundry pods", ["42-count", "81-count"]),
        ("Fabric softener", ["64 oz", "dryer sheets 120-count"]),
        ("Bleach", ["regular 81 oz", "splash-less 55 oz"]),
        ("Dish soap", ["lemon 19 oz", "original 38 oz"]),
        ("Dishwasher detergent pods", ["32-count", "70-count"]),
        ("All-purpose cleaner", ["spray 32 oz", "refill 64 oz"]),
        ("Glass cleaner", ["spray 23 oz", "refill 67 oz"]),
        ("Disinfecting wipes", ["35-count", "3-pack"]),
        ("Kitchen trash bags", ["13 gal 45-count", "13 gal 90-count"]),
        ("Outdoor trash bags", ["33 gal 20-count", "39 gal 32-count"]),
        ("Sponges", ["3-pack", "non-scratch 6-pack"]),
        ("Hand soap", ["7.5 oz", "refill 56 oz"]),
        ("Toilet bowl cleaner", ["24 oz", "2-pack"]),
        ("Air freshener spray", ["8 oz", "3-pack"]),
        ("Floor cleaner", ["32 oz", "concentrate 90 oz"]),
    ],
    "HOUSEHOLD_2": [
        ("Aluminum foil", ["75 sq ft", "200 sq ft"]),
        ("Plastic wrap", ["200 sq ft", "400 sq ft"]),
        ("Zip storage bags", ["quart 48-count", "gallon 38-count"]),
        ("Sandwich bags", ["100-count", "300-count"]),
        ("Food storage containers", ["10-piece set", "24-piece set"]),
        ("Paper plates", ["9 in 50-count", "10 in 100-count"]),
        ("Plastic cups", ["16 oz 50-count", "18 oz 100-count"]),
        ("Plastic forks", ["48-count", "100-count"]),
        ("AA batteries", ["8-pack", "24-pack"]),
        ("AAA batteries", ["8-pack", "20-pack"]),
        ("LED light bulbs", ["60W equivalent 4-pack", "100W equivalent 2-pack"]),
        ("Bath towel", ["white", "gray"]),
        ("Hand towel", ["white 2-pack", "gray 2-pack"]),
        ("Hangers", ["plastic 18-pack", "velvet 30-pack"]),
        ("Storage bin", ["16 qt", "66 qt"]),
        ("Candles", ["vanilla 3-wick", "unscented tealights 50-count"]),
        ("Extension cord", ["6 ft", "15 ft"]),
        ("Duct tape", ["1.88 in x 20 yd", "1.88 in x 60 yd"]),
        ("Kitchen towels", ["2-pack", "6-pack"]),
        ("Can opener", ["manual", "electric"]),
    ],
    "HOBBIES_1": [
        ("Jigsaw puzzle", ["500 pieces", "1000 pieces"]),
        ("Playing cards", ["2-deck pack", "jumbo index"]),
        ("Crayons", ["24-count", "64-count"]),
        ("Colored pencils", ["12-count", "50-count"]),
        ("Washable markers", ["10-count", "40-count"]),
        ("Modeling clay", ["4-pack", "12-pack"]),
        ("Construction paper", ["200 sheets", "500 sheets"]),
        ("Building bricks set", ["300 pieces", "900 pieces"]),
        ("Toy race cars", ["5-pack", "20-pack"]),
        ("Bubble solution", ["32 oz", "wand 3-pack"]),
        ("Sidewalk chalk", ["20-count", "52-count"]),
        ("Board game", ["family classic", "strategy edition"]),
        ("Craft glue", ["4 oz", "glue sticks 12-pack"]),
        ("Yarn", ["worsted 7 oz", "baby soft 5 oz"]),
        ("Plush teddy bear", ["12 in", "24 in"]),
        ("Foam darts refill", ["30-pack", "100-pack"]),
        ("Watercolor paint set", ["16 colors", "36 colors"]),
        ("Sketchbook", ["9 x 12 in", "5.5 x 8.5 in"]),
    ],
    "HOBBIES_2": [
        ("Party balloons", ["assorted 25-count", "metallic 50-count"]),
        ("Birthday candles", ["24-count", "number candle"]),
        ("Gift wrap", ["holiday 3-roll", "everyday 2-roll"]),
        ("Gift bags", ["medium 4-pack", "large 6-pack"]),
        ("Greeting cards", ["birthday 6-pack", "assorted 12-pack"]),
        ("Plastic tablecloth", ["rectangle 54 x 108 in", "round 84 in"]),
        ("Pool noodles", ["2-pack", "6-pack"]),
        ("Beach towel", ["striped", "oversized"]),
        ("Sunscreen", ["SPF 30 8 oz", "SPF 50 spray 6 oz"]),
        ("Charcoal briquettes", ["8 lb", "16 lb"]),
        ("Water guns", ["2-pack", "super soaker"]),
        ("String lights", ["warm white 100-count", "multicolor 200-count"]),
        ("Halloween candy", ["assorted 50-count", "family bag 100-count"]),
        ("Costume mask", ["kids", "adult"]),
        ("Garden gloves", ["1 pair", "3-pack"]),
        ("Flower seeds", ["wildflower mix", "sunflower 3-pack"]),
        ("Picnic plates", ["paper 40-count", "reusable 8-count"]),
        ("Cooler", ["24 qt", "48 qt"]),
    ],
}


def offline_names(dept_id: str, item_ids: list[str], seed: int = 42) -> dict[str, str]:
    """Deterministically assign a unique vocabulary name to each item in one department."""
    vocab = _VOCAB.get(dept_id) or _VOCAB[f"{dept_id.split('_')[0]}_1"]
    pool = [f"{product} {variant}" for product, variants in vocab for variant in variants]
    random.Random(f"{seed}-{dept_id}").shuffle(pool)
    names = {}
    for i, item_id in enumerate(sorted(item_ids)):
        base = pool[i % len(pool)]
        names[item_id] = base if i < len(pool) else f"{base} ({i // len(pool) + 1})"
    return names


def load_cache(path: Path = CACHE_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())["names"]


def save_cache(names: dict[str, str], path: Path = CACHE_PATH) -> None:
    payload = {
        "note": "Synthetic product names generated by the teacher model. Not real Walmart products.",
        "names": dict(sorted(names.items())),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")


def teacher_names(
    dept_id: str, item_ids: list[str], taken: set[str], batch_size: int = 40, attempts: int = 3
) -> dict[str, str]:
    """Ask the teacher model to name items, in batches, rejecting duplicates of `taken`."""
    from datagen.teacher import Teacher, parse_json_object

    teacher = Teacher()
    theme = DEPT_THEMES.get(dept_id, "general merchandise")
    taken_lower = {n.lower() for n in taken}
    names: dict[str, str] = {}
    todo = sorted(item_ids)
    for _ in range(attempts):
        if not todo:
            break
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            avoid = sorted(n for n in names.values())[-60:]
            prompt = (
                f"You are naming products for a mock US supermarket catalog, department {dept_id} "
                f"({theme}).\nGive each item ID below a realistic, distinct, generic product name "
                "of 2 to 6 words that includes a size or pack count (for example "
                '"Paper towels 6-pack" or "Long grain white rice 5 lb"). No real brand names. '
                "Spread the names across the whole department, not one product type.\n"
                + (f"Do not reuse any of these names: {json.dumps(avoid)}\n" if avoid else "")
                + f"Item IDs: {json.dumps(batch)}\n"
                "Reply with only a JSON object mapping each item ID to its name."
            )
            reply = teacher.chat(
                [{"role": "user", "content": prompt}], temperature=0.8, max_tokens=6000
            )
            try:
                proposed = parse_json_object(reply.get("content") or "")
            except ValueError:
                continue
            for item_id in batch:
                name = str(proposed.get(item_id, "")).strip()
                if 2 <= len(name) <= 60 and name.lower() not in taken_lower:
                    names[item_id] = name
                    taken_lower.add(name.lower())
        todo = [i for i in todo if i not in names]
    if todo:
        raise RuntimeError(f"teacher failed to name {len(todo)} items in {dept_id}: {todo[:5]}...")
    return names
