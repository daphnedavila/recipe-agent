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


if __name__ == "__main__":
    # This block only runs when you execute this file directly
    # (python recipe_engine.py) — not when app.py imports generate_recipe from it.
    # Quick local test — run: python recipe_engine.py
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
