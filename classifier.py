"""
Preference model: predicts how much the user will like a recipe,
based on structured features (cuisine, spice level, cook time, etc.)
extracted by recipe_engine.extract_features().

This is intentionally simple — a small gradient boosting model —
since the point isn't state-of-the-art ML, it's a legible, explainable
pipeline you can describe end-to-end in an interview.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer

MIN_RATINGS_TO_TRAIN = 5  # below this, predictions aren't meaningful


class PreferenceModel:
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        self.is_trained = False

    def _prepare_features(self, features: dict) -> dict:
        """
        Flattens a features dict into a form DictVectorizer can handle.
        Categorical fields (cuisine, protein) become one-hot via DictVectorizer
        automatically when they're strings; numeric fields stay as numbers.
        """
        clean = {}
        for key, value in features.items():
            if isinstance(value, list):
                # e.g. flavor_profile -> one boolean feature per flavor
                for v in value:
                    clean[f"{key}__{v}"] = 1
            else:
                clean[key] = value
        return clean

    def fit(self, rated_recipes: list):
        """
        rated_recipes: list of dicts with at least {"features": dict, "rating": int}
        """
        if len(rated_recipes) < MIN_RATINGS_TO_TRAIN:
            self.is_trained = False
            return False

        X_dicts = [self._prepare_features(r["features"]) for r in rated_recipes]
        y = np.array([r["rating"] for r in rated_recipes])

        X = self.vectorizer.fit_transform(X_dicts)
        self.model.fit(X, y)
        self.is_trained = True
        return True

    def predict(self, features: dict) -> float:
        """Returns predicted rating (1-5 scale, not rounded)."""
        if not self.is_trained:
            return None
        X_dict = self._prepare_features(features)
        X = self.vectorizer.transform([X_dict])
        pred = self.model.predict(X)[0]
        return float(np.clip(pred, 1, 5))
