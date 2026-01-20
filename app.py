import io
import numpy as np
import pandas as pd
import streamlit as st
import penaltyblog as pb

st.set_page_config(page_title="Dixon–Coles Forecasting", layout="wide")

# -------------------------
# Helpers
# -------------------------
REQUIRED_RESULTS_COLS = {"team_home", "team_away", "goals_home", "goals_away"}
REQUIRED_FIXTURE_COLS = {"team_home", "team_away"}

def implied_decimal_odds(p: float) -> float:
    p = float(p)
    if p <= 0:
        return float("inf")
    return 1.0 / p

def fmt_prob(p: float) -> str:
    if pd.isna(p):
        return ""
    return f"{100.0 * float(p):.1f}%"

def fmt_odds(o: float) -> str:
    if o == float("inf") or pd.isna(o):
        return ""
    return f"{float(o):.2f}"

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
    if hasattr(prob_grid, "both_teams_to_score"):
        p_yes = float(prob_grid.both_teams_to_score)
        return p_yes, 1.0 - p_yes
    if hasattr(prob_grid, "btts_yes") and hasattr(prob_grid, "btts_no"):
        return float(prob_grid.btts_yes), float(prob_grid.btts_no)
    if hasattr(prob_grid, "btts"):
        val = prob_grid.btts
        if callable(val):
            p_yes = float(val("yes"))
            return p_yes, 1.0 - p_yes
    raise AttributeError(
        "BTTS market not found on FootballProbabilityGrid. "
        "Tried: both_teams_to_score, btts_yes/btts_no, btts('yes')."
    )

def forecast_one(clf, home: str, away: str, ou_line: float, ah_line: float,
                 include_1x2: bool, include_ou: bool, include_btts: bool, include_ah: bool):
    probs = clf.predict(home, away)

    out = {"team_home": home, "team_away": away}

    if include_1x2:
        p_home, p_draw, p_away = probs.home_draw_away
        out.update({
            "p_home": float(p_home), "p_draw": float(p_draw), "p_away": float(p_away),
            "odds_home": implied_decimal_odds(p_home),
            "odds_draw": implied_decimal_odds(p_draw),
            "odds_away": implied_decimal_odds(p_away),
        })

    if include_ou:
        p_over = float(probs.total_goals("over", ou_line))
        p_under = 1.0 - p_over
        out.update({
            f"p_over_{ou_line}": p_over,
            f"p_under_{ou_line}": p_under,
            f"odds_over_{ou_line}": implied_decimal_odds(p_over),
            f"odds_under_{ou_line}": implied_decimal_odds(p_under),
        })

    if include_btts:
        p_yes, p_no = get_btts_probs(probs)
        out.update({
            "p_btts_yes": float(p_yes),
            "p_btts_no": float(p_no),
            "odds_btts_yes": implied_decimal_odds(p_yes),
            "odds_btts_no": implied_decimal_odds(p_no),
        })

    if include_ah:
        p_ah_home = float(probs.asian_handicap("home", ah_line))
        p_ah_away = 1.0 - p_ah_home
        out.update({
            f"p_ah_home_{ah_line}": float(p_ah_home),
            f"p_ah_away_{ah_line}": float(p_ah_away),
            f"odds_ah_home_{ah_line}": implied_decimal_odds(p_ah_home),
            f"odds_ah_away_{ah_line}": implied_decimal_odds(p_ah_away),
        })

    return out

def sample_results_csv() -> bytes:
    s = """team_home,team_away,goals_home,goals_away
Alpha FC,Beta FC,2,1
Beta FC,Alpha FC,0,1
Alpha FC,Gamma FC,1,1
Gamma FC,Beta FC,2,2
Beta FC,Gamma FC,1,0
Gamma FC,Alpha FC,0,2
"""
    return s.encode("utf-8")

def sample_fixtures_csv() -> bytes:
    s = """team_home,team_away
Alpha FC,Beta FC
Gamma FC,Alpha FC
"""
    return s.encode("utf-8")

# -------------------------
# Header
# -------------------------
st.title("⚽ Dixon–Coles Forecasting (Penaltyblog)")
st.caption("Browser-based UI: upload historical results + fixtures → probabilities + implied odds.")

# -------------------------
# Sidebar: Stepper-style controls
# -------------------------
with st.sidebar:
    st.subheader("Setup")
    st.download_button("Download sample results.csv", data=sample_results_csv(),
                       file_name="results_sample.csv", mime="text/csv")
    st.download_button("Download sample fixtures.csv", data=sample_fixtures_csv(),
                       file_name="fixtures_sample.csv", mime="text/csv")
    st.divider()

    st.markdown("**Step 1 — Upload historical results**")
    results_file = st.file_uploader("results.csv", type=["csv"], label_visibility="collapsed")

    st.markdown("**Step 2 — Choose markets**")
    colA, colB = st.columns(2)
    with colA:
        include_1x2 = st.checkbox("1X2", value=True)
        include_btts = st.checkbox("BTTS", value=True)
    with colB:
        include_ou = st.checkbox("Over/Under", value=True)
        include_ah = st.checkbox("Asian handicap", value=True)

    ou_line = st.selectbox("O/U line", [0.5, 1.5, 2.5, 3.5], index=2, disabled=not include_ou)
    ah_line = st.selectbox("AH line (home)", [-1.5, -0.5, 0.5, 1.5], index=2, disabled=not include_ah)

    st.markdown("**Step 3 — Upload fixtures**")
    fixtures_file = st.file_uploader("fixtures.csv", type=["csv"], label_visibility="collapsed")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["1) Upload & Fit", "2) Forecast", "Help"])

# -------------------------
# Tab 1: Upload & Fit
# -------------------------
with tab1:
    st.subheader("Upload & fit model")

    if not results_file:
        st.info("Upload **results.csv** in the sidebar to fit the model. (You can download a sample in the sidebar.)")
        st.stop()

    try:
        results_df = read_csv(results_file)
        validate_columns(results_df, REQUIRED_RESULTS_COLS, "results.csv")
    except Exception as e:
        st.error(f"Could not read results.csv: {e}")
        st.stop()

    for c in ["goals_home", "goals_away"]:
        results_df[c] = pd.to_numeric(results_df[c], errors="coerce")

    results_df = results_df.dropna(subset=["team_home", "team_away", "goals_home", "goals_away"]).copy()
    results_df["team_home"] = results_df["team_home"].astype(str)
    results_df["team_away"] = results_df["team_away"].astype(str)

    teams = sorted(set(results_df["team_home"]).union(set(results_df["team_away"])))

    with st.spinner("Fitting Dixon–Coles model..."):
        try:
            clf = fit_dc_model(results_df)
        except Exception as e:
            st.error(f"Model fit failed: {e}")
            st.stop()

    st.success("Model fitted ✅")
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", f"{len(results_df):,}")
    c2.metric("Teams", f"{len(teams):,}")
    c3.metric("Markets selected", f"{sum([include_1x2, include_ou, include_btts, include_ah])}")

    with st.expander("Preview uploaded results"):
        st.dataframe(results_df.head(50), use_container_width=True)

    with st.expander("Team list"):
        st.write(teams)

    # store in session so tab2 can use it
    st.session_state["clf"] = clf
    st.session_state["teams"] = teams

# -------------------------
# Tab 2: Forecast
# -------------------------
with tab2:
    st.subheader("Forecast fixtures")

    if "clf" not in st.session_state:
        st.info("Fit the model first in **Upload & Fit**.")
        st.stop()

    clf = st.session_state["clf"]
    teams = st.session_state["teams"]

    if not fixtures_file:
        st.info("Upload **fixtures.csv** in the sidebar to generate forecasts.")
        st.stop()

    try:
        fixtures_df = read_csv(fixtures_file)
        validate_columns(fixtures_df, REQUIRED_FIXTURE_COLS, "fixtures.csv")
    except Exception as e:
        st.error(f"Could not read fixtures.csv: {e}")
        st.stop()

    fixtures_df["team_home"] = fixtures_df["team_home"].astype(str)
    fixtures_df["team_away"] = fixtures_df["team_away"].astype(str)

    unknown = sorted(set(
        [t for t in fixtures_df["team_home"].tolist() + fixtures_df["team_away"].tolist() if t not in teams]
    ))
    if unknown:
        st.warning("Team names in fixtures not found in results training data:\n\n" + ", ".join(unknown))

    rows, errors = [], []
    progress = st.progress(0, text="Forecasting...")
    n = len(fixtures_df)

    for i, r in fixtures_df.iterrows():
        home, away = r["team_home"], r["team_away"]
        try:
            rows.append(
                forecast_one(
                    clf, home, away, float(ou_line), float(ah_line),
                    include_1x2, include_ou, include_btts, include_ah
                )
            )
        except Exception as e:
            errors.append((i, home, away, str(e)))
        progress.progress(int(100 * (i + 1) / max(n, 1)), text=f"Forecasting {i+1}/{n}")

    progress.empty()

    out_df = pd.DataFrame(rows)

    # Pretty display: show formatted columns alongside raw if desired
    pretty = out_df.copy()
    for col in pretty.columns:
        if col.startswith("p_"):
            pretty[col] = pretty[col].apply(fmt_prob)
        if col.startswith("odds_"):
            pretty[col] = pretty[col].apply(fmt_odds)

    st.markdown("### Results (formatted)")
    st.dataframe(pretty, use_container_width=True)

    st.markdown("### Raw output (numbers)")
    st.dataframe(out_df, use_container_width=True)

    st.download_button(
        "Download forecasts.csv",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="forecasts.csv",
        mime="text/csv",
    )

    if errors:
        st.subheader("Errors")
        st.dataframe(pd.DataFrame(errors, columns=["row", "team_home", "team_away", "error"]))

# -------------------------
# Tab 3: Help
# -------------------------
with tab3:
    st.subheader("CSV formats")

    st.markdown("**results.csv required columns**")
    st.code("team_home, team_away, goals_home, goals_away", language="text")

    st.markdown("**fixtures.csv required columns**")
    st.code("team_home, team_away", language="text")

    st.markdown("**Common issues**")
    st.markdown(
        "- Team names must match exactly between both files.\n"
        "- Goals must be numeric.\n"
        "- If you use different column names, rename them to the required ones."
    )


