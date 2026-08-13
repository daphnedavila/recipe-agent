# 🍳 Sous Agent

A kitchen helper that thinks about your next meal so you don't have to. Sous Agent generates recipe suggestions with Claude, learns your taste from your ratings, and predicts how much you'll like new recipes before you even cook them.

**Live app:** [sous-agent.streamlit.app](https://sous-agent.streamlit.app/)

---

## What it does

1. **Generate** — enter a cuisine, time budget, and ingredients on hand; Claude suggests a specific recipe.
2. **Extract features** — a second Claude call turns the recipe into structured data (cuisine, protein, spice level, cook time, difficulty, richness).
3. **Rate** — rate the recipe 1-5 stars, with an optional note.
4. **Learn** — once you've rated a handful of recipes, a small trained model starts predicting how much you'll like the *next* recipe before you rate it.
5. **Evaluate** — a built-in evaluation compares that trained model against a second, LLM-native approach, using your real rating history.

Multiple people can use the same deployed app — each account (username + password) has its own private recipes, ratings, and taste predictions.

---

## Why this project

Most "AI recipe generator" demos stop at generation. The interesting problem is *personalization*: can a system actually learn what a specific person likes, and can you prove it? This project treats that as a real ML question rather than a vibe — it compares two different approaches to preference prediction and evaluates both against real held-out data.

---

## Architecture

```
app.py            Streamlit UI — login, recipe generation, rating, history, evaluation
recipe_engine.py  Claude API calls: recipe generation, feature extraction,
                   taste-profile summarization, LLM-native rating prediction
classifier.py     Gradient boosting model (scikit-learn) trained on structured
                   recipe features to predict a user's rating
db.py             SQLite data layer: users, recipes, ratings, rate limiting
eval.py           Standalone leave-one-out evaluation script (also available
                   in-app under "How well does it know your taste?")
```

### Two competing approaches to preference prediction

| | **Learned model** | **Plain-language guess** |
|---|---|---|
| How it works | Structured features (cuisine, spice level, cook time, etc.) → trained `GradientBoostingRegressor` | Rating history → natural-language taste summary → Claude predicts a rating directly from that summary |
| Where it lives | `classifier.py` | `recipe_engine.py` (`summarize_taste_profile`, `predict_rating_llm`) |

Both are evaluated the same way: **leave-one-out cross-validation**. For each rated recipe, the approach is trained/summarized on every *other* rating, then asked to predict the held-out one. This simulates predicting a genuinely new recipe rather than testing on data the model already saw.

---

## Results

On real rating data collected through the app, the learned model consistently outperformed the plain-language guess:

| Approach | Mean Absolute Error (stars) |
|---|---|
| Learned model | ~0.8 |
| Plain-language guess | ~1.1 |

**Takeaway:** turning preferences into structured features and training a small model on them captures taste patterns more reliably than describing preferences in a sentence and asking an LLM to judge directly. This held even with a fairly small dataset (well under 30 ratings) — a useful reminder that structure + a simple model can beat a more "AI-native"-sounding approach for a well-defined prediction task.

*Caveat: this result is based on one person's rating history at modest scale. It's a directional finding, not a claim that would generalize to arbitrary users or larger datasets without further testing.*

---

## Tech stack

- **Claude API** (`claude-sonnet-4-6`) — recipe generation, feature extraction, taste summarization, LLM-native prediction
- **Streamlit** — UI and public deployment (Streamlit Community Cloud)
- **scikit-learn** — `GradientBoostingRegressor` + `DictVectorizer` for the learned preference model
- **SQLite** — data storage (see [Known limitations](#known-limitations))

---

## Running it locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-your-key-here   # PowerShell: $env:ANTHROPIC_API_KEY="..."
streamlit run app.py
```

Run the standalone evaluation script against a local account:
```bash
python eval.py <username>
```

---

## Known limitations

- **Storage is not currently persistent across deployments.** The app uses local SQLite, and Streamlit Community Cloud's filesystem resets on reboot/redeploy — meaning rating history can be lost when the app is redeployed. Planned fix: migrate to a hosted SQLite-compatible database (e.g. Turso) or Postgres, so data survives redeploys.
- **Small evaluation sample.** Results above are based on limited ratings from active development/testing. More data would sharpen the comparison between approaches.
- **No verification of ingredient safety/availability** — recipes are generated, not sourced from a vetted database.

---

## Possible next steps

- Persistent hosted database (removes the reboot-wipes-data issue)
- Larger-scale evaluation once more ratings accumulate
- A third baseline (e.g. simple average-by-cuisine) for a fuller comparison
- Export ratings/recipes as JSON for portability between accounts
