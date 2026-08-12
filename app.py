"""
Palate-learning cooking agent — Streamlit app.

Run locally with:
    streamlit run app.py

Requires ANTHROPIC_API_KEY to be set as an environment variable,
or entered in Streamlit secrets when deployed.
"""

import os
import uuid
import streamlit as st

from db import (
    init_db, save_recipe, save_rating, get_all_rated_recipes,
    check_and_increment_usage,
)
from recipe_engine import generate_recipe

st.set_page_config(page_title="Palate Agent", page_icon="🍳", layout="centered")

init_db()

DAILY_LIMIT_PER_VISITOR = 15  # generations per visitor per day, to protect API credits

# --- Per-visitor session ID for rate limiting (not tied to identity, just this browser session) ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- API key resolution: local env var first, then Streamlit secrets (for deployment) ---
def get_api_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None

API_KEY = get_api_key()

st.title("🍳 Palate Agent")
st.caption("An agent that learns your food preferences over time.")

if not API_KEY:
    st.error(
        "No API key found. Set the ANTHROPIC_API_KEY environment variable "
        "before running `streamlit run app.py`."
    )
    st.stop()

# --- Session state: holds the current unrated recipe, if any ---
if "current_recipe" not in st.session_state:
    st.session_state.current_recipe = None
if "current_recipe_id" not in st.session_state:
    st.session_state.current_recipe_id = None

# --- Input form ---
st.subheader("What do you want to cook?")

col1, col2 = st.columns(2)
with col1:
    cuisine = st.text_input("Cuisine (or 'any')", value="any")
with col2:
    time_minutes = st.number_input("Time available (minutes)", min_value=5, max_value=240, value=30, step=5)

ingredients = st.text_input("Ingredients on hand (optional)", placeholder="e.g. chicken thighs, rice, broccoli")

generate_clicked = st.button("Generate recipe", type="primary")

if generate_clicked:
    allowed, remaining = check_and_increment_usage(st.session_state.session_id, DAILY_LIMIT_PER_VISITOR)
    if not allowed:
        st.warning(
            f"You've hit the daily limit of {DAILY_LIMIT_PER_VISITOR} recipe generations. "
            "This keeps the site's API costs manageable — please come back tomorrow!"
        )
        st.stop()

    with st.spinner("Thinking of something good..."):
        try:
            result = generate_recipe(API_KEY, {
                "cuisine": cuisine,
                "time_minutes": int(time_minutes),
                "ingredients_on_hand": ingredients or "none specified",
            })
            recipe_id = save_recipe(
                title=result["title"],
                recipe_text=result["recipe_text"],
                cuisine=cuisine,
                time_minutes=int(time_minutes),
                ingredients_on_hand=ingredients,
            )
            st.session_state.current_recipe = result
            st.session_state.current_recipe_id = recipe_id
        except Exception as e:
            st.error(f"Something went wrong generating the recipe: {e}")

# --- Show current recipe + rating UI ---
if st.session_state.current_recipe:
    recipe = st.session_state.current_recipe
    st.divider()
    st.subheader(recipe["title"])
    st.markdown(recipe["recipe_text"])

    st.divider()
    st.subheader("Rate this recipe")
    rating = st.slider("How much do you like this?", 1, 5, 3)
    note = st.text_input("Optional note (e.g. 'too spicy', 'loved the texture')")

    if st.button("Submit rating"):
        save_rating(st.session_state.current_recipe_id, rating, note)
        st.success("Rating saved!")
        st.session_state.current_recipe = None
        st.session_state.current_recipe_id = None
        st.rerun()

# --- History ---
st.divider()
with st.expander("Your rating history"):
    history = get_all_rated_recipes()
    if not history:
        st.write("No ratings yet.")
    else:
        for item in history:
            st.write(f"**{item['title']}** — {item['rating']}/5"
                      + (f" _{item['note']}_" if item['note'] else ""))
