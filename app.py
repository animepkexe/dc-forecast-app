import io
import numpy as np
import pandas as pd
import streamlit as st
import penaltyblog as pb

st.set_page_config(page_title="Dixon–Coles Forecasting", layout="wide")

REQUIRED_RESULTS_COLS = {"team_home", "team_away", "goals_home", "goals_away"}
REQUIRED_FIXTURE_COLS = {"team_home", "team_away"}

# -------------------------
# Helpers
# -------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def implied_decimal_odds(p: float) -> float:
    p = safe_float(p)
    if np.isnan(p) or p <= 0:
        return float("inf")
    return 1.0 / p

def fmt_prob(p: float, decimals: int = 1) -> str:
    p = safe_float(p)
    if np.isnan(p):
        return ""
    return f"{100.0 * p:.{decimals}f}%"

def fmt_odds(o: float, decimals: int = 2) -> str:
    o = safe_float(o)
    if np.isnan(o) or o == float("inf"):
        return ""
    return f"{o:.{decimals}f}"

def validate_columns(df: pd.DataFrame, required: set, name: str):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

@st.cache_data(show_spinner=False)
def read_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(content))

def parse_date_series_dayfirst(s: pd.Series) -> pd.Series:
    # Your format is 17/08/2024 => dayfirst=True
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def has_date_column(df: pd.DataFrame) -> bool:
    return "date" in df.columns

# -------------------------
# penaltyblog model fitting (supports weights in newer versions)
# -------------------------
@st.cache_resource(show_spinner=True)
def fit_dc_model(results_df: pd.DataFrame, weights: np.ndarray | None, use_gradient: bool):
    try:
        model = pb.models.DixonColesGoalModel(
            results_df["goals_home"],
            results_df["goals_away"],
            results_df["team_home"],
            results_df["team_away"],
            weights=weights,
        )
    except TypeError:
        # Older signature: weights positional or not supported
        if weights is None:
            model = pb.models.DixonColesGoalModel(
                results_df["goals_home"],
                results_df["goals_away"],
                results_df["team_home"],
                results_df["team_away"],
            )
        else:
            model = pb.models.DixonColesGoalModel(
                results_df["goals_home"],
                results_df["goals_away"],
                results_df["team_home"],
                results_df["team_away"],
                weights,
            )

    try:
        model.fit(use_gradient=use_gradient)
    except TypeError:
        model.fit()

    return model

# -------------------------
# FootballProbabilityGrid market helpers (version-safe)
# -------------------------
def get_btts_probs(grid):
    if hasattr(grid, "btts_yes") and hasattr(grid, "btts_no"):
        return safe_float(grid.btts_yes), safe_float(grid.btts_no)
    if hasattr(grid, "both_teams_to_score"):
        p_yes = safe_float(grid.both_teams_to_score)
        return p_yes, 1.0 - p_yes
    if hasattr(grid, "btts") and callable(grid.btts):
        p_yes = safe_float(grid.btts("yes"))
        return p_yes, 1.0 - p_yes
    return np.nan, np.nan

def get_double_chance(grid):
    out = {}
    if hasattr(grid, "double_chance_1x"): out["1X"] = safe_float(grid.double_chance_1x)
    if hasattr(grid, "double_chance_x2"): out["X2"] = safe_float(grid.double_chance_x2)
    if hasattr(grid, "double_chance_12"): out["12"] = safe_float(grid.double_chance_12)
    return out

def get_dnb(grid):
    out = {}
    if hasattr(grid, "draw_no_bet_home"): out["Home DNB"] = safe_float(grid.draw_no_bet_home)
    if hasattr(grid, "draw_no_bet_away"): out["Away DNB"] = safe_float(grid.draw_no_bet_away)
    return out

def exact_score_prob(grid, hg: int, ag: int) -> float:
    if hasattr(grid, "exact_score"):
        return safe_float(grid.exact_score(hg, ag))
    return np.nan

def win_to_nil_probs(grid):
    if not hasattr(grid, "exact_score"):
        return np.nan, np.nan
    max_goals = 10
    p_home = 0.0
    p_away = 0.0
    ok = False
    for hg in range(1, max_goals + 1):
        p = exact_score_prob(grid, hg, 0)
        if not np.isnan(p):
            p_home += p
            ok = True
    for ag in range(1, max_goals + 1):
        p = exact_score_prob(grid, 0, ag)
        if not np.isnan(p):
            p_away += p
            ok = True
    if not ok:
        return np.nan, np.nan
    return p_home, p_away

# -------------------------
# EV calculation (per £1 stake)
# Betfair commission reduces profit, not stake.
# net_return = 1 + (odds-1)*(1-commission)
# EV = p*net_return - 1
# EV% = EV*100
# -------------------------
def ev_percent(p: float, betfair_odds: float, commission: float) -> float:
    p = safe_float(p)
    o = safe_float(betfair_odds)
    c = safe_float(commission)
    if np.isnan(p) or np.isnan(o) or o <= 1 or p <= 0 or p >= 1 or c < 0 or c >= 1:
        return np.nan
    net_return = 1.0 + (o - 1.0) * (1.0 - c)
    ev = p * net_return - 1.0
    return 100.0 * ev

# -------------------------
# App header
# -------------------------
st.title("⚽ Dixon–Coles Forecasting (Penaltyblog)")
st.caption("Upload results → fit Dixon–Coles → pick teams → get model odds → type Betfair odds → EV% & value.")

# -------------------------
# Sidebar: model + markets + display
# -------------------------
with st.sidebar:
    st.subheader("Step 1 — Upload historical results")
    results_file = st.file_uploader("results.csv", type=["csv"], label_visibility="collapsed")

    st.divider()
    st.subheader("Model options")
    use_time_decay = st.checkbox("Use time-decay weighting (requires 'date')", value=True)
    xi = st.slider("Decay factor (xi)", min_value=0.0, max_value=0.01, value=0.001, step=0.0005)
    use_gradient = st.checkbox("Use gradient optimisation (if supported)", value=True)

    st.divider()
    st.subheader("Markets")
    include_1x2 = st.checkbox("1X2", value=True)
    include_double_chance = st.checkbox("Double Chance (1X / X2 / 12)", value=True)
    include_dnb = st.checkbox("Draw No Bet (Home/Away)", value=True)

    include_ou = st.checkbox("Over/Under", value=True)
    ou_line = st.selectbox("O/U line", [0.5, 1.5, 2.5, 3.5], index=2, disabled=not include_ou)

    include_btts = st.checkbox("BTTS", value=True)

    include_ah = st.checkbox("Asian Handicap", value=True)
    ah_line = st.selectbox("AH line (home)", [-1.5, -0.5, 0.0, 0.5, 1.5], index=3, disabled=not include_ah)

    include_exact_score = st.checkbox("Exact Score (Top N)", value=True)
    top_n_scores = st.slider("Top N scorelines", 5, 30, 10, disabled=not include_exact_score)

    include_win_to_nil = st.checkbox("Win to Nil (Home/Away)", value=True)

    st.divider()
    st.subheader("Betfair EV")
    commission = st.slider("Betfair commission", 0.0, 0.20, 0.05, 0.01)
    show_value_only = st.checkbox("Show value only", value=False)

    st.divider()
    st.subheader("Display")
    prob_decimals = st.selectbox("Probability % decimals", [0, 1, 2], index=1)
    odds_decimals = st.selectbox("Odds decimals", [2, 3], index=0)

# -------------------------
# Tabs
# -------------------------
tab_fit, tab_forecast, tab_help = st.tabs(["1) Upload & Fit", "2) Forecast + Betfair EV", "Help"])

# -------------------------
# Fit tab
# -------------------------
with tab_fit:
    if not results_file:
        st.info("Upload **results.csv** from the sidebar to begin.")
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

    # Date handling
    date_ok = False
    if has_date_column(results_df):
        results_df["date"] = parse_date_series_dayfirst(results_df["date"])
        if results_df["date"].notna().any():
            date_ok = True

    if use_time_decay and not date_ok:
        st.warning("Time-decay is enabled but no usable 'date' column was found. It will be ignored.")

    # Date filter
    if date_ok:
        min_d = results_df["date"].min()
        max_d = results_df["date"].max()
        st.markdown("### Training date range")
        d1, d2 = st.slider(
            "Filter historical matches used for fitting",
            min_value=min_d.to_pydatetime(),
            max_value=max_d.to_pydatetime(),
            value=(min_d.to_pydatetime(), max_d.to_pydatetime()),
        )
        mask = (results_df["date"] >= pd.Timestamp(d1)) & (results_df["date"] <= pd.Timestamp(d2))
        train_df = results_df.loc[mask].copy()
    else:
        train_df = results_df

    teams = sorted(set(train_df["team_home"]).union(set(train_df["team_away"])))

    weights = None
    if use_time_decay and date_ok:
        # weights helper expects dates series and xi
        weights = pb.models.dixon_coles_weights(train_df["date"], xi)

    with st.spinner("Fitting Dixon–Coles model..."):
        try:
            clf = fit_dc_model(train_df, weights=weights, use_gradient=use_gradient)
        except Exception as e:
            st.error(f"Model fit failed: {e}")
            st.stop()

    st.success("Model fitted ✅")
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches used", f"{len(train_df):,}")
    c2.metric("Teams", f"{len(teams):,}")
    c3.metric("Time-decay", "On" if (use_time_decay and date_ok) else "Off")

    with st.expander("Preview training data"):
        st.dataframe(train_df.head(50), use_container_width=True)

    st.session_state["clf"] = clf
    st.session_state["teams"] = teams

# -------------------------
# Forecast + EV tab
# -------------------------
with tab_forecast:
    if "clf" not in st.session_state:
        st.info("Fit the model first in **Upload & Fit**.")
        st.stop()

    clf = st.session_state["clf"]
    teams = st.session_state["teams"]

    st.markdown("### Pick teams")
    colA, colB, colC = st.columns([4, 4, 2])
    with colA:
        home_team = st.selectbox("Home", teams, index=0)
    with colB:
        away_team = st.selectbox("Away", teams, index=1 if len(teams) > 1 else 0)
    with colC:
        run = st.button("Run", type="primary", use_container_width=True)

    if not run:
        st.stop()

    if home_team == away_team:
        st.error("Home and Away cannot be the same team.")
        st.stop()

    # Predict
    try:
        try:
            grid = clf.predict(home_team, away_team, max_goals=15, normalize=True)
        except TypeError:
            grid = clf.predict(home_team, away_team)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    fixture_label = f"{home_team} vs {away_team}"

    # Build market table
    rows = []

    # 1X2
    if include_1x2:
        p_home, p_draw, p_away = grid.home_draw_away
        for sel, p in [("Home", p_home), ("Draw", p_draw), ("Away", p_away)]:
            p = safe_float(p)
            rows.append([fixture_label, "1X2", sel, p, implied_decimal_odds(p)])

    # Double Chance
    if include_double_chance:
        dc = get_double_chance(grid)
        for sel, p in dc.items():
            rows.append([fixture_label, "Double Chance", sel, p, implied_decimal_odds(p)])

    # DNB
    if include_dnb:
        dnb = get_dnb(grid)
        for sel, p in dnb.items():
            rows.append([fixture_label, "Draw No Bet", sel, p, implied_decimal_odds(p)])

    # Over/Under
    if include_ou:
        p_over = safe_float(grid.total_goals("over", float(ou_line)))
        p_under = 1.0 - p_over
        rows.append([fixture_label, f"Totals {ou_line}", f"Over {ou_line}", p_over, implied_decimal_odds(p_over)])
        rows.append([fixture_label, f"Totals {ou_line}", f"Under {ou_line}", p_under, implied_decimal_odds(p_under)])

    # BTTS
    if include_btts:
        p_yes, p_no = get_btts_probs(grid)
        if not np.isnan(p_yes):
            rows.append([fixture_label, "BTTS", "Yes", p_yes, implied_decimal_odds(p_yes)])
        if not np.isnan(p_no):
            rows.append([fixture_label, "BTTS", "No", p_no, implied_decimal_odds(p_no)])

    # Asian handicap
    if include_ah:
        p_ah_home = safe_float(grid.asian_handicap("home", float(ah_line)))
        p_ah_away = 1.0 - p_ah_home
        rows.append([fixture_label, f"AH {ah_line}", f"Home {float(ah_line):+}", p_ah_home, implied_decimal_odds(p_ah_home)])
        rows.append([fixture_label, f"AH {ah_line}", f"Away {-float(ah_line):+}", p_ah_away, implied_decimal_odds(p_ah_away)])

    # Win to nil
    if include_win_to_nil:
        p_hwtn, p_awtn = win_to_nil_probs(grid)
        if not np.isnan(p_hwtn):
            rows.append([fixture_label, "Win to Nil", "Home Win to Nil", p_hwtn, implied_decimal_odds(p_hwtn)])
        if not np.isnan(p_awtn):
            rows.append([fixture_label, "Win to Nil", "Away Win to Nil", p_awtn, implied_decimal_odds(p_awtn)])

    # Exact score Top N
    if include_exact_score and hasattr(grid, "exact_score"):
        candidates = []
        max_goals = 8
        for hg in range(0, max_goals + 1):
            for ag in range(0, max_goals + 1):
                p = exact_score_prob(grid, hg, ag)
                if not np.isnan(p) and p > 0:
                    candidates.append((hg, ag, p))
        candidates.sort(key=lambda x: x[2], reverse=True)
        for hg, ag, p in candidates[: int(top_n_scores)]:
            rows.append([fixture_label, "Correct Score", f"{hg}-{ag}", p, implied_decimal_odds(p)])

    market_df = pd.DataFrame(rows, columns=["fixture", "market", "selection", "model_prob", "model_odds"])

    # Prepare editable Betfair odds table
    editor_df = market_df.copy()
    editor_df["betfair_odds"] = np.nan

    # If we already have previous edits for same fixture, reuse them
    key = f"bf_odds_{fixture_label}"
    if key in st.session_state:
        prev = st.session_state[key]
        # merge on market+selection
        editor_df = editor_df.merge(
            prev[["market", "selection", "betfair_odds"]],
            on=["market", "selection"],
            how="left",
            suffixes=("", "_prev"),
        )
        editor_df["betfair_odds"] = editor_df["betfair_odds_prev"].combine_first(editor_df["betfair_odds"])
        editor_df = editor_df.drop(columns=["betfair_odds_prev"])

    st.markdown("### Model outputs + enter Betfair odds")
    st.caption("Type Betfair decimal odds into the last column. EV% accounts for commission on winnings.")

    edited = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "model_prob": st.column_config.NumberColumn("Model prob", format="%.6f", disabled=True),
            "model_odds": st.column_config.NumberColumn("Model odds (fair)", format="%.3f", disabled=True),
            "betfair_odds": st.column_config.NumberColumn("Betfair odds", format="%.3f"),
        },
        disabled=["fixture", "market", "selection"],
        key=f"editor_{fixture_label}",
    )

    # Store odds edits in session
    st.session_state[key] = edited[["market", "selection", "betfair_odds"]].copy()

    # Compute EV + value
    out = edited.copy()
    out["EV_%"] = out.apply(lambda r: ev_percent(r["model_prob"], r["betfair_odds"], commission), axis=1)
    out["Value"] = out["EV_%"].apply(lambda x: (not np.isnan(x)) and (x > 0.0))

    # Pretty display
    pretty = out.copy()
    pretty["model_prob"] = pretty["model_prob"].apply(lambda x: fmt_prob(x, prob_decimals))
    pretty["model_odds"] = pretty["model_odds"].apply(lambda x: fmt_odds(x, odds_decimals))
    pretty["betfair_odds"] = pretty["betfair_odds"].apply(lambda x: "" if np.isnan(safe_float(x)) else fmt_odds(x, odds_decimals))
    pretty["EV_%"] = pretty["EV_%"].apply(lambda x: "" if np.isnan(safe_float(x)) else f"{safe_float(x):.2f}%")

    if show_value_only:
        pretty = pretty[pretty["Value"] == True]

    st.markdown("### Value view")
    st.dataframe(pretty[["fixture", "market", "selection", "model_prob", "model_odds", "betfair_odds", "EV_%", "Value"]], use_container_width=True)

    # Quick summary
    n_value = int(out["Value"].sum())
    st.metric("Value selections", n_value)

    # Downloads
    st.download_button(
        "Download full table (with Betfair odds + EV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="model_vs_betfair_ev.csv",
        mime="text/csv",
    )

# -------------------------
# Help tab
# -------------------------
with tab_help:
    st.markdown("### CSV formats")
    st.markdown("**results.csv required columns**")
    st.code("team_home, team_away, goals_home, goals_away", language="text")
    st.markdown("Optional (recommended):")
    st.code("date  (your format: 17/08/2024)", language="text")
    st.markdown("**fixtures.csv required columns** (only used if you later add batch mode again)")
    st.code("team_home, team_away", language="text")

    st.markdown("### EV% definition")
    st.markdown(
        "For £1 stake, with Betfair commission on winnings:\n\n"
        "- net_return = 1 + (odds − 1) × (1 − commission)\n"
        "- EV = p × net_return − 1\n"
        "- EV% = 100 × EV\n\n"
        "Value = EV% > 0"
    )


