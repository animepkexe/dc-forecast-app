import io
import numpy as np
import pandas as pd
import streamlit as st
import penaltyblog as pb

st.set_page_config(page_title="Dixon–Coles Forecasting (penaltyblog)", layout="wide")
st.title("⚽ Dixon–Coles Forecasting Tool (penaltyblog)")
st.caption(
    "Upload historical results to fit a Dixon–Coles model, then upload fixtures to get probabilities + implied odds."
)

REQUIRED_RESULTS_COLS = {"team_home", "team_away", "goals_home", "goals_away"}
REQUIRED_FIXTURE_COLS = {"team_home", "team_away"}

def implied_decimal_odds(p: float) -> float:
    p = float(p)
    if p <= 0:
        return float("inf")
    return 1.0 / p

def validate_columns(df: pd.DataFrame, required: set, name: str):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

@st.cache_data(show_spinner=False)
def read_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(content))

@st.cache_resource(show_spinner=True)
def fit_dc_model(results_df: pd.DataFrame):
    clf = pb.models.DixonColesGoalModel(
        results_df["goals_home"],
        results_df["goals_away"],
        results_df["team_home"],
        results_df["team_away"],
    )
    clf.fit()
    return clf

def get_btts_probs(prob_grid):
    """
    Version-safe BTTS extraction from penaltyblog FootballProbabilityGrid.
    Tries a few common attribute names across versions.
    """
    # Most common in examples/docs
    if hasattr(prob_grid, "both_teams_to_score"):
        p_yes = float(getattr(prob_grid, "both_teams_to_score"))
        return p_yes, 1.0 - p_yes

    # Some versions use explicit yes/no attributes
    if hasattr(prob_grid, "btts_yes") and hasattr(prob_grid, "btts_no"):
        return float(prob_grid.btts_yes), float(prob_grid.btts_no)

    # As a last resort, try a callable market method if present
    if hasattr(prob_grid, "btts"):
        val = prob_grid.btts
        if callable(val):
            p_yes = float(val("yes"))
            return p_yes, 1.0 - p_yes

    raise AttributeError(
        "BTTS market not found on FootballProbabilityGrid. "
        "Tried: both_teams_to_score, btts_yes/btts_no, btts('yes')."
    )

def forecast_one(clf, home: str, away: str, ou_line: float, ah_line: float):
    probs = clf.predict(home, away)

    # 1X2
    p_home, p_draw, p_away = probs.home_draw_away

    # Over/Under total goals
    p_over = float(probs.total_goals("over", ou_line))
    p_under = 1.0 - p_over

    # BTTS
    p_btts_yes, p_btts_no = get_btts_probs(probs)

    # Asian handicap (home)
    p_ah_home = float(probs.asian_handicap("home", ah_line))
    p_ah_away = 1.0 - p_ah_home

    return {
        "team_home": home,
        "team_away": away,

        "p_home": float(p_home),
        "p_draw": float(p_draw),
        "p_away": float(p_away),
        "odds_home": implied_decimal_odds(p_home),
        "odds_draw": implied_decimal_odds(p_draw),
        "odds_away": implied_decimal_odds(p_away),

        f"p_over_{ou_line}": p_over,
        f"p_under_{ou_line}": p_under,
        f"odds_over_{ou_line}": implied_decimal_odds(p_over),
        f"odds_under_{ou_line}": implied_decimal_odds(p_under),

        "p_btts_yes": float(p_btts_yes),
        "p_btts_no": float(p_btts_no),
        "odds_btts_yes": implied_decimal_odds(p_btts_yes),
        "odds_btts_no": implied_decimal_odds(p_btts_no),

        f"p_ah_home_{ah_line}": float(p_ah_home),
        f"p_ah_away_{ah_line}": float(p_ah_away),
        f"odds_ah_home_{ah_line}": implied_decimal_odds(p_ah_home),
        f"odds_ah_away_{ah_line}": implied_decimal_odds(p_ah_away),
    }

with st.sidebar:
    st.header("1) Upload historical results")
    results_file = st.file_uploader("results.csv", type=["csv"])

    st.header("2) Market settings")
    ou_line = st.selectbox("Over/Under line", [0.5, 1.5, 2.5, 3.5], index=2)
    ah_line = st.selectbox("Asian handicap (home)", [-1.5, -0.5, 0.5, 1.5], index=2)

    st.header("3) Upload fixtures")
    fixtures_file = st.file_uploader("fixtures.csv", type=["csv"])

# --- Load and validate results ---
if not results_file:
    st.info("Upload **results.csv** (historical results) to fit the model.")
    st.stop()

try:
    results_df = read_csv(results_file)
    validate_columns(results_df, REQUIRED_RESULTS_COLS, "results.csv")
except Exception as e:
    st.error(f"Could not read results.csv: {e}")
    st.stop()

# Coerce goal columns to numeric and clean
for c in ["goals_home", "goals_away"]:
    results_df[c] = pd.to_numeric(results_df[c], errors="coerce")

results_df = results_df.dropna(subset=["team_home", "team_away", "goals_home", "goals_away"]).copy()
results_df["team_home"] = results_df["team_home"].astype(str)
results_df["team_away"] = results_df["team_away"].astype(str)

teams = sorted(set(results_df["team_home"]).union(set(results_df["team_away"])))

# Fit model
with st.spinner("Fitting Dixon–Coles model..."):
    try:
        clf = fit_dc_model(results_df)
    except Exception as e:
        st.error(f"Model fit failed: {e}")
        st.stop()

st.success("Model fitted ✅")
st.caption(f"Teams in training data: {len(teams)}")
with st.expander("Show team list"):
    st.write(teams)

# --- Load fixtures ---
if not fixtures_file:
    st.info("Upload **fixtures.csv** to generate forecasts.")
    st.stop()

try:
    fixtures_df = read_csv(fixtures_file)
    validate_columns(fixtures_df, REQUIRED_FIXTURE_COLS, "fixtures.csv")
except Exception as e:
    st.error(f"Could not read fixtures.csv: {e}")
    st.stop()

fixtures_df["team_home"] = fixtures_df["team_home"].astype(str)
fixtures_df["team_away"] = fixtures_df["team_away"].astype(str)

# Warn about unknown teams
unknown = sorted(set(
    [t for t in fixtures_df["team_home"].tolist() + fixtures_df["team_away"].tolist() if t not in teams]
))
if unknown:
    st.warning(
        "These teams appear in fixtures but not in training data (name mismatch or missing history):\n\n"
        + ", ".join(unknown)
    )

# Forecast
rows, errors = [], []
for i, r in fixtures_df.iterrows():
    home, away = r["team_home"], r["team_away"]
    try:
        rows.append(forecast_one(clf, home, away, float(ou_line), float(ah_line)))
    except Exception as e:
        errors.append((i, home, away, str(e)))

out_df = pd.DataFrame(rows)

st.subheader("Forecasts")
st.dataframe(out_df, use_container_width=True)

st.download_button(
    "Download forecasts CSV",
    data=out_df.to_csv(index=False).encode("utf-8"),
    file_name="forecasts.csv",
    mime="text/csv",
)

if errors:
    st.subheader("Errors")
    st.dataframe(pd.DataFrame(errors, columns=["row", "team_home", "team_away", "error"]))

