"""
Week 4 evaluation: compares two approaches for predicting how much a user
will like a recipe, using leave-one-out cross-validation on their real
rating history:

  1. Classifier approach (classifier.py) — a gradient boosting model trained
     on structured features (cuisine, spice level, etc.)
  2. LLM-native approach (recipe_engine.py) — a natural-language taste
     summary fed back into the LLM, which predicts a rating directly.

For each rated recipe, both approaches are trained/summarized on ALL OTHER
ratings, then asked to predict the held-out recipe's rating. This simulates
"how well would this have predicted a new recipe" rather than testing on
data the model already saw.

Run with:
    python eval.py <username>

Requires ANTHROPIC_API_KEY to be set (the LLM-native approach makes real
API calls — expect roughly 2 calls per recipe, so keep N reasonable,
e.g. under ~40 recipes to keep cost and runtime low).
"""

import sys
import os
import statistics

from db import init_db, get_all_rated_recipes
from classifier import PreferenceModel, MIN_RATINGS_TO_TRAIN
from recipe_engine import summarize_taste_profile, predict_rating_llm


def evaluate_classifier(rated_recipes):
    """Leave-one-out evaluation of the trained classifier."""
    errors = []
    predictions = []

    for i in range(len(rated_recipes)):
        held_out = rated_recipes[i]
        training_set = rated_recipes[:i] + rated_recipes[i + 1:]

        if len(training_set) < MIN_RATINGS_TO_TRAIN:
            continue  # not enough data yet to train a fair held-out prediction

        model = PreferenceModel()
        trained = model.fit(training_set)
        if not trained:
            continue

        predicted = model.predict(held_out["features"])
        actual = held_out["rating"]
        error = abs(predicted - actual)
        errors.append(error)
        predictions.append((held_out["title"], actual, predicted))

    return errors, predictions


def evaluate_llm_native(api_key, rated_recipes):
    """Leave-one-out evaluation of the LLM-native taste-profile approach."""
    errors = []
    predictions = []

    for i in range(len(rated_recipes)):
        held_out = rated_recipes[i]
        training_set = rated_recipes[:i] + rated_recipes[i + 1:]

        if len(training_set) < MIN_RATINGS_TO_TRAIN:
            continue

        summary = summarize_taste_profile(api_key, training_set)
        predicted = predict_rating_llm(api_key, summary, held_out["features"])
        actual = held_out["rating"]
        error = abs(predicted - actual)
        errors.append(error)
        predictions.append((held_out["title"], actual, predicted))

    return errors, predictions


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval.py <username>")
        sys.exit(1)

    username = sys.argv[1]
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable first.")
        sys.exit(1)

    init_db()
    rated_recipes = get_all_rated_recipes(username)

    print(f"Found {len(rated_recipes)} rated recipes for '{username}'.")
    if len(rated_recipes) < MIN_RATINGS_TO_TRAIN + 1:
        print(f"Need at least {MIN_RATINGS_TO_TRAIN + 1} ratings to run a meaningful "
              f"leave-one-out evaluation. Rate a few more recipes first.")
        sys.exit(1)

    print("\n--- Evaluating classifier (structured features + gradient boosting) ---")
    clf_errors, clf_predictions = evaluate_classifier(rated_recipes)
    for title, actual, predicted in clf_predictions:
        print(f"  {title[:40]:40s} actual={actual}  predicted={predicted:.1f}")

    print("\n--- Evaluating LLM-native (taste summary + direct LLM prediction) ---")
    print("(this makes real API calls, may take a minute...)")
    llm_errors, llm_predictions = evaluate_llm_native(api_key, rated_recipes)
    for title, actual, predicted in llm_predictions:
        print(f"  {title[:40]:40s} actual={actual}  predicted={predicted:.1f}")

    print("\n=== RESULTS ===")
    if clf_errors:
        print(f"Classifier   — Mean Absolute Error: {statistics.mean(clf_errors):.2f} "
              f"(n={len(clf_errors)} held-out predictions)")
    else:
        print("Classifier   — not enough data to evaluate.")

    if llm_errors:
        print(f"LLM-native   — Mean Absolute Error: {statistics.mean(llm_errors):.2f} "
              f"(n={len(llm_errors)} held-out predictions)")
    else:
        print("LLM-native   — not enough data to evaluate.")

    if clf_errors and llm_errors:
        better = "Classifier" if statistics.mean(clf_errors) < statistics.mean(llm_errors) else "LLM-native"
        print(f"\n{better} performed better on this data (lower MAE = better).")

    print("\nMAE (Mean Absolute Error) = average |predicted - actual| rating, on a 1-5 scale.")
    print("Lower is better. E.g. MAE of 0.8 means predictions are off by under 1 star on average.")


if __name__ == "__main__":
    main()
