"""
Core recipe generation logic. This talks to the Claude API to suggest
a recipe given constraints (cuisine, time, ingredients on hand).
"""

import json
import anthropic

MODEL = "claude-sonnet-4-6"


def generate_recipe(api_key: str, constraints: dict) -> dict:
    """
    constraints: dict like:
        {
            "cuisine": "Italian" or "any",
            "time_minutes": 30,
            "ingredients_on_hand": "chicken, rice, broccoli"
        }

    Returns: {"title": str, "recipe_text": str}
    """
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Suggest one specific recipe given these constraints:
- Cuisine preference: {constraints.get('cuisine', 'any')}
- Time available: {constraints.get('time_minutes', 30)} minutes
- Ingredients on hand / must use: {constraints.get('ingredients_on_hand', 'none specified')}

Respond ONLY with valid JSON, no markdown fences, no preamble, in this exact format:
{{"title": "Recipe Name", "recipe_text": "Full recipe with ingredients list and numbered steps, formatted in markdown."}}
"""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )

    text = resp.content[0].text.strip()
    # Strip markdown fences if the model adds them despite instructions
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


# --- Feature extraction: turns a recipe's free text into structured data ---
# the classifier in Week 2 Session 2 will train on these features.

FEATURE_SCHEMA = {
    "cuisine": "string, e.g. 'Italian', 'Thai', 'Mexican'",
    "protein": "string, e.g. 'chicken', 'tofu', 'none'",
    "spice_level": "integer 1-5, 1=none 5=very spicy",
    "cook_time_minutes": "integer, estimated total time",
    "difficulty": "integer 1-5, 1=trivial 5=advanced technique",
    "richness": "integer 1-5, 1=light 5=very rich/heavy",
}


def extract_features(api_key: str, recipe_text: str) -> dict:
    """
    Extracts structured features from a recipe's text using the LLM.
    Returns a dict matching FEATURE_SCHEMA.
    """
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Analyze this recipe and extract structured features.

Recipe:
{recipe_text}

Respond ONLY with valid JSON (no markdown fences, no preamble) matching exactly this schema:
{json.dumps(FEATURE_SCHEMA, indent=2)}
"""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


# --- LLM-native preference approach: instead of a trained classifier, this
# summarizes taste in plain language and asks the LLM directly to predict a
# rating. Used in Week 4 to compare against the classifier's approach.

def summarize_taste_profile(api_key: str, rated_recipes: list) -> str:
    """
    rated_recipes: list of dicts with at least {"title": str, "features": dict,
                    "rating": int, "note": str (optional)}
    Returns a short natural-language summary of the person's taste.
    """
    if not rated_recipes:
        return "No ratings yet — no taste profile available."

    client = anthropic.Anthropic(api_key=api_key)
    lines = []
    for r in rated_recipes:
        note = r.get("note") or ""
        lines.append(f"- {r['title']} | rating: {r['rating']}/5 | note: {note} | features: {r['features']}")

    prompt = f"""Here is a history of recipes suggested to a user, with their ratings (1-5) and notes:

{chr(10).join(lines)}

In 3-4 sentences, summarize this person's food preferences: what they tend to like,
what they tend to dislike, and any patterns in cuisine, spice level, cook time, or
richness. Be specific and concise, written for use as context in a future prompt."""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def predict_rating_llm(api_key: str, taste_summary: str, features: dict) -> float:
    """
    Asks the LLM directly to predict a 1-5 rating for a recipe (given its features)
    based on a taste profile summary — the "LLM-native" alternative to the trained
    classifier in classifier.py.
    """
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""A person's food preferences, based on their rating history:
{taste_summary}

Given these preferences, predict how much they would rate a NEW recipe with these features (1-5 scale):
{json.dumps(features, indent=2)}

Respond ONLY with a single number between 1 and 5 (can include one decimal, e.g. 3.5). No other text."""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    try:
        return max(1.0, min(5.0, float(text)))
    except ValueError:
        return 3.0  # fallback if the model doesn't return a clean number


if __name__ == "__main__":
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable first.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
    else:
        result = generate_recipe(api_key, {
            "cuisine": "Thai",
            "time_minutes": 30,
            "ingredients_on_hand": "chicken thighs, coconut milk, basil",
        })
        print(f"\nTITLE: {result['title']}\n")
        print(result["recipe_text"])
