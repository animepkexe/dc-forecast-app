import io
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import penaltyblog as pb

st.set_page_config(page_title="Dixon–Coles Forecasting", layout="wide")

# -------------------------
# Required columns
# -------------------------
REQUIRED_RESULTS_COLS = {"team_home", "team_away", "goals_home", "goals_away"}
REQUIRED_FIXTURE_COLS = {"team_home", "team_away"}

# -------------------------
# Formatting helpers
# -------------------------
def implied_decimal_odds(p: float) -> float:
    p = float(p)
    if p <= 0:
        return float("inf")
    return 1.0 / p

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def fmt_prob(p: float, decimals: int = 1) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return f"{100.0 * float(p):.{decimals}f}%"

def fmt_odds(o: float, decimals: int = 2) -> str:
    if o is None or o == float("inf") or (isinstance(o, float) and np.isnan(o)):
        return ""
    return f"{float(o):.{decimals}f}"

# -------------------------
# Data helpers
# -------------------------
def validate_columns(df: pd.DataFrame, required: set, name: str):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

@st.cache_data(show_spinner=False)
def read_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(content))

def parse_date_series(s: pd.Series) -> pd.Series:
    # Handle common date formats; results become pandas datetime
    return pd.to_datetime(s, errors="coerce", utc=False)

def has_date_column(df: pd.DataFrame) -> bool:
    return "date" in df.columns

# -------------------------
# penaltyblog model fitting (supports weights)
# -------------------------
@st.cache_resource(show_spinner=True)
def fit_dc_model(results_df: pd.DataFrame, weights: np.ndarray | None, use_gradient: bool):
    """
    Fits a Dixon–Coles model. Uses keyword 'weights' when possible (newer versions),
    otherwise falls back to positional (older versions).
    """
    try:
        model = pb.models.DixonColesGoalModel(
            results_df["goals_home"],
            results_df["goals_away"],
            results_df["team_home"],
            results_df["team_away"],
            weights=weights,
        )
    except TypeError:
        # Older signature used positional weights
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

    # Fit options (newer penaltyblog supports use_gradient/minimizer_options, but keep safe)
    try:
        model.fit(use_gradient=use_gradient)
    except TypeError:
        model.fit()

    return model

# -------------------------
# FootballProbabilityGrid market helpers (version-safe)
# -------------------------
def get_btts_probs(grid):
    # Newer grid: btts_yes/btts_no; docs also show both_teams_to_score in some contexts :contentReference[oaicite:2]{index=2}
    if hasattr(grid, "btts_yes") and hasattr(grid, "btts_no"):
        return safe_float(grid.btts_yes), safe_float(grid.btts_no)
    if hasattr(grid, "both_teams_to_score"):
        p_yes = safe_float(grid.both_teams_to_score)
        return p_yes, 1.0 - p_yes
    if hasattr(grid, "btts"):
        val = grid.btts
        if callable(val):
            p_yes = safe_float(val("yes"))
            return p_yes, 1.0 - p_yes
    raise AttributeError("BTTS not available on this FootballProbabilityGrid.")

def get_double_chance(grid):
    # From penaltyblog v1.5.0 example :contentReference[oaicite:3]{index=3}
    out = {}
    if hasattr(grid, "double_chance_1x"):
        out["1X"] = safe_float(grid.double_chance_1x)
    if hasattr(grid, "double_chance_x2"):
        out["X2"] = safe_float(grid.double_chance_x2)
    if hasattr(grid, "double_chance_12"):
        out["12"] = safe_float(grid.double_chance_12)
    return out

def get_dnb(grid):
    # From penaltyblog v1.5.0 example :contentReference[oaicite:4]{index=4}
    out = {}
    if hasattr(grid, "draw_no_bet_home"):
        out["Home DNB"] = safe_float(grid.draw_no_bet_home)
    if hasattr(grid, "draw_no_bet_away"):
        out["Away DNB"] = safe_float(grid.draw_no_bet_away)
    return out

def exact_score_prob(grid, hg: int, ag: int) -> float:
    # From penaltyblog v1.5.0 example :contentReference[oaicite:5]{index=5}
    if hasattr(grid, "exact_score"):
        return safe_float(grid.exact_score(hg, ag))
    # If not available, we can't reliably compute without the matrix API.
    return np.nan

def goal_distributions(grid):
    # From penaltyblog v1.5.0 example :contentReference[oaicite:6]{index=6}
    out = {}
    if hasattr(grid, "home_goal_distribution"):
        out["home"] = np.array(grid.home_goal_distribution(), dtype=float)
    if hasattr(grid, "away_goal_distribution"):
        out["away"] = np.array(grid.away_goal_distribution(), dtype=float)
    if hasattr(grid, "total_goals_distribution"):
        out["total"] = np.array(grid.total_goals_distribution(), dtype=float)
    return out

def win_to_nil_probs(grid):
    """
    Win to Nil:
      - Home win & away scores 0
      - Away win & home scores 0
    We compute using exact scores if available, otherwise return NaN.
    """
    max_goals = 10
    p_home_wtn = 0.0
    p_away_wtn = 0.0
    any_valid = False

    if hasattr(grid, "exact_score"):
        for hg in range(1, max_goals + 1):
            p = exact_score_prob(grid, hg, 0)
            if not np.isnan(p):
                p_home_wtn += p
                any_valid = True
        for ag in range(1, max_goals + 1):
            p = exact_score_prob(grid, 0, ag)
            if not np.isnan(p):
                p_away_wtn += p
                any_valid = True

    if not any_valid:
        return np.nan, np.nan

    return p_home_wtn, p_away_wtn

# -------------------------
# App header
# -------------------------
st.title("⚽ Dixon–Coles Forecasting (Penaltyblog)")
st.caption(
    "Upload historical results, optionally apply time-decay, then forecast markets with a clean betting-style output."
)

# -------------------------
# Sidebar: Inputs
# -------------------------
with st.sidebar:
    st.subheader("Step 1 — Upload historical results")
    results_file = st.file_uploader("results.csv", type=["csv"], label_visibility="collapsed")

    st.divider()
    st.subheader("Step 2 — Model options")

    use_time_decay = st.checkbox("Use time-decay weighting (requires a 'date' column)", value=True)
    xi = st.slider("Decay factor (xi)", min_value=0.0, max_value=0.01, value=0.001, step=0.0005,
                   help="Higher xi down-weights older matches more strongly.")
    use_gradient = st.checkbox("Use gradient optimisation (if supported)", value=True)

    st.divider()
    st.subheader("Step 3 — Markets")

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

    include_distributions = st.checkbox("Goal distributions (Home/Away/Total)", value=False)
    include_win_to_nil = st.checkbox("Win to Nil (Home/Away)", value=True)

    st.divider()
    st.subheader("Display")
    prob_decimals = st.selectbox("Probability % decimals", [0, 1, 2], index=1)
    odds_decimals = st.selectbox("Odds decimals", [2, 3], index=0)
    show_wide_table = st.checkbox("Also show wide table", value=False)

# -------------------------
# Tabs
# -------------------------
tab_fit, tab_forecast, tab_help = st.tabs(["1) Upload & Fit", "2) Forecast", "Help"])

# -------------------------
# Tab: Upload & Fit
# -------------------------
with tab_fit:
    if not results_file:
        st.info("Upload **results.csv** from the sidebar to begin.")
        st.stop()

    # Read and validate
    try:
        results_df = read_csv(results_file)
        validate_columns(results_df, REQUIRED_RESULTS_COLS, "results.csv")
    except Exception as e:
        st.error(f"Could not read results.csv: {e}")
        st.stop()

    # Coerce goals + clean
    for c in ["goals_home", "goals_away"]:
        results_df[c] = pd.to_numeric(results_df[c], errors="coerce")

    results_df = results_df.dropna(subset=["team_home", "team_away", "goals_home", "goals_away"]).copy()
    results_df["team_home"] = results_df["team_home"].astype(str)
    results_df["team_away"] = results_df["team_away"].astype(str)

    # Date handling + filter
    date_ok = False
    if has_date_column(results_df):
        results_df["date"] = parse_date_series(results_df["date"])
        if results_df["date"].notna().any():
            date_ok = True

    if use_time_decay and not date_ok:
        st.warning("Time-decay is enabled but your results.csv has no usable 'date' column. Time-decay will be ignored.")

    # If date exists, allow filtering the training window
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
        filtered_df = results_df.loc[mask].copy()
    else:
        filtered_df = results_df

    teams = sorted(set(filtered_df["team_home"]).union(set(filtered_df["team_away"])))

    # Build weights if possible and requested
    weights = None
    if use_time_decay and date_ok:
        # penaltyblog provides pb.models.dixon_coles_weights(date, xi) :contentReference[oaicite:7]{index=7}
        weights = pb.models.dixon_coles_weights(filtered_df["date"], xi)

    # Fit model
    with st.spinner("Fitting Dixon–Coles model..."):
        try:
            clf = fit_dc_model(filtered_df, weights=weights, use_gradient=use_gradient)
        except Exception as e:
            st.error(f"Model fit failed: {e}")
            st.stop()

    st.success("Model fitted ✅")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches used", f"{len(filtered_df):,}")
    c2.metric("Teams", f"{len(teams):,}")
    c3.metric("Time-decay", "On" if (use_time_decay and date_ok) else "Off")
    c4.metric("xi", f"{xi:.4f}" if (use_time_decay and date_ok) else "—")

    with st.expander("Preview training data"):
        st.dataframe(filtered_df.head(50), use_container_width=True)

    with st.expander("Team list"):
        st.write(teams)

    # Save to session for Forecast tab
    st.session_state["clf"] = clf
    st.session_state["teams"] = teams

# -------------------------
# Tab: Forecast
# -------------------------
with tab_forecast:
    if "clf" not in st.session_state:
        st.info("Fit the model first in **Upload & Fit**.")
        st.stop()

    clf = st.session_state["clf"]
    teams = st.session_state["teams"]

    st.markdown("### Step A — Provide fixtures")
    mode = st.radio("Choose input method", ["Upload fixtures.csv", "Build fixtures here"], horizontal=True)

    fixtures_df = None

    if mode == "Upload fixtures.csv":
        fixtures_file = st.file_uploader("fixtures.csv", type=["csv"])
        if not fixtures_file:
            st.info("Upload **fixtures.csv** (columns: team_home, team_away), or switch to 'Build fixtures here'.")
            st.stop()
        try:
            fixtures_df = read_csv(fixtures_file)
            validate_columns(fixtures_df, REQUIRED_FIXTURE_COLS, "fixtures.csv")
        except Exception as e:
            st.error(f"Could not read fixtures.csv: {e}")
            st.stop()
        fixtures_df["team_home"] = fixtures_df["team_home"].astype(str)
        fixtures_df["team_away"] = fixtures_df["team_away"].astype(str)

    else:
        # Manual builder stored in session
        if "fixture_rows" not in st.session_state:
            st.session_state["fixture_rows"] = []

        col1, col2, col3 = st.columns([4, 4, 2])
        with col1:
            home_sel = st.selectbox("Home team", teams, index=0)
        with col2:
            away_sel = st.selectbox("Away team", teams, index=1 if len(teams) > 1 else 0)
        with col3:
            if st.button("Add", use_container_width=True):
                if home_sel == away_sel:
                    st.warning("Home and Away cannot be the same.")
                else:
                    st.session_state["fixture_rows"].append({"team_home": home_sel, "team_away": away_sel})

        if st.session_state["fixture_rows"]:
            st.dataframe(pd.DataFrame(st.session_state["fixture_rows"]), use_container_width=True)
            if st.button("Clear fixtures"):
                st.session_state["fixture_rows"] = []
        else:
            st.info("Add at least one fixture using the dropdowns above.")

        if not st.session_state["fixture_rows"]:
            st.stop()

        fixtures_df = pd.DataFrame(st.session_state["fixture_rows"])

    # Warn about unknown teams (name mismatch)
    unknown = sorted(set(
        [t for t in fixtures_df["team_home"].tolist() + fixtures_df["team_away"].tolist() if t not in teams]
    ))
    if unknown:
        st.warning("These teams are not in training data (likely a naming mismatch): " + ", ".join(unknown))

    st.markdown("### Step B — Forecast")

    run = st.button("Run forecast", type="primary")
    if not run:
        st.stop()

    # Build "betting-like" output rows:
    market_rows = []
    wide_rows = []

    progress = st.progress(0, text="Forecasting...")
    n = len(fixtures_df)

    for i, r in fixtures_df.iterrows():
        home, away = r["team_home"], r["team_away"]

        try:
            # Use normalize=True by default in newer penaltyblog; keep signature safe
            try:
                grid = clf.predict(home, away, max_goals=15, normalize=True)
            except TypeError:
                grid = clf.predict(home, away)

            fixture_label = f"{home} vs {away}"

            # --- 1X2 ---
            if include_1x2:
                p_home, p_draw, p_away = grid.home_draw_away
                for sel, p in [("Home", p_home), ("Draw", p_draw), ("Away", p_away)]:
                    p = safe_float(p)
                    market_rows.append([fixture_label, "1X2", sel, p, implied_decimal_odds(p)])
                wide = {
                    "fixture": fixture_label,
                    "p_home": safe_float(p_home),
                    "p_draw": safe_float(p_draw),
                    "p_away": safe_float(p_away),
                }
            else:
                wide = {"fixture": fixture_label}

            # --- Double chance ---
            if include_double_chance:
                dc = get_double_chance(grid)
                for sel, p in dc.items():
                    market_rows.append([fixture_label, "Double Chance", sel, p, implied_decimal_odds(p)])
                wide.update({f"p_dc_{k.lower()}": v for k, v in dc.items()})

            # --- Draw No Bet ---
            if include_dnb:
                dnb = get_dnb(grid)
                for sel, p in dnb.items():
                    market_rows.append([fixture_label, "Draw No Bet", sel, p, implied_decimal_odds(p)])
                wide.update({f"p_{k.lower().replace(' ', '_')}": v for k, v in dnb.items()})

            # --- Over/Under ---
            if include_ou:
                p_over = safe_float(grid.total_goals("over", float(ou_line)))
                p_under = 1.0 - p_over
                market_rows.append([fixture_label, f"Totals {ou_line}", f"Over {ou_line}", p_over, implied_decimal_odds(p_over)])
                market_rows.append([fixture_label, f"Totals {ou_line}", f"Under {ou_line}", p_under, implied_decimal_odds(p_under)])
                wide.update({f"p_over_{ou_line}": p_over, f"p_under_{ou_line}": p_under})

                # If totals() exists, include push-aware info for integer lines
                if hasattr(grid, "totals"):
                    try:
                        u, push, o = grid.totals(float(ou_line))
                        wide.update({f"p_totals_under_{ou_line}": safe_float(u), f"p_totals_push_{ou_line}": safe_float(push), f"p_totals_over_{ou_line}": safe_float(o)})
                    except Exception:
                        pass

            # --- BTTS ---
            if include_btts:
                p_yes, p_no = get_btts_probs(grid)
                market_rows.append([fixture_label, "BTTS", "Yes", p_yes, implied_decimal_odds(p_yes)])
                market_rows.append([fixture_label, "BTTS", "No", p_no, implied_decimal_odds(p_no)])
                wide.update({"p_btts_yes": p_yes, "p_btts_no": p_no})

            # --- Asian handicap ---
            if include_ah:
                p_ah_home = safe_float(grid.asian_handicap("home", float(ah_line)))
                p_ah_away = 1.0 - p_ah_home
                market_rows.append([fixture_label, f"AH {ah_line}", f"Home {ah_line:+}", p_ah_home, implied_decimal_odds(p_ah_home)])
                market_rows.append([fixture_label, f"AH {ah_line}", f"Away {-float(ah_line):+}", p_ah_away, implied_decimal_odds(p_ah_away)])
                wide.update({f"p_ah_home_{ah_line}": p_ah_home, f"p_ah_away_{ah_line}": p_ah_away})

                # Push-aware breakdown if available (asian_handicap_probs)
                if hasattr(grid, "asian_handicap_probs"):
                    try:
                        w, push, l = grid.asian_handicap_probs("home", float(ah_line))
                        wide.update({f"p_ah_home_win_{ah_line}": safe_float(w),
                                     f"p_ah_home_push_{ah_line}": safe_float(push),
                                     f"p_ah_home_lose_{ah_line}": safe_float(l)})
                    except Exception:
                        pass

            # --- Win to Nil ---
            if include_win_to_nil:
                p_home_wtn, p_away_wtn = win_to_nil_probs(grid)
                if not np.isnan(p_home_wtn):
                    market_rows.append([fixture_label, "Win to Nil", "Home Win to Nil", p_home_wtn, implied_decimal_odds(p_home_wtn)])
                if not np.isnan(p_away_wtn):
                    market_rows.append([fixture_label, "Win to Nil", "Away Win to Nil", p_away_wtn, implied_decimal_odds(p_away_wtn)])
                wide.update({"p_home_win_to_nil": p_home_wtn, "p_away_win_to_nil": p_away_wtn})

            # --- Exact score Top N ---
            if include_exact_score and hasattr(grid, "exact_score"):
                # enumerate a reasonable grid of scorelines, then take top N
                candidates = []
                max_goals = 8
                for hg in range(0, max_goals + 1):
                    for ag in range(0, max_goals + 1):
                        p = exact_score_prob(grid, hg, ag)
                        if not np.isnan(p) and p > 0:
                            candidates.append((hg, ag, p))
                candidates.sort(key=lambda x: x[2], reverse=True)
                top = candidates[: int(top_n_scores)]
                for hg, ag, p in top:
                    market_rows.append([fixture_label, "Correct Score", f"{hg}-{ag}", p, implied_decimal_odds(p)])

            # --- Distributions ---
            if include_distributions:
                dists = goal_distributions(grid)
                # Just store a few headline distribution values in wide output (optional)
                if "total" in dists and len(dists["total"]) > 0:
                    # e.g., P(Total Goals = 0..5)
                    for k in range(min(6, len(dists["total"]))):
                        wide[f"p_total_goals_eq_{k}"] = safe_float(dists["total"][k])

            wide_rows.append(wide)

        except Exception as e:
            market_rows.append([f"{home} vs {away}", "ERROR", str(e), np.nan, np.nan])

        progress.progress(int(100 * (i + 1) / max(n, 1)), text=f"Forecasting {i+1}/{n}")

    progress.empty()

    # Build dataframes
    market_df = pd.DataFrame(market_rows, columns=["fixture", "market", "selection", "prob", "odds"])
    wide_df = pd.DataFrame(wide_rows)

    # Pretty view
    pretty = market_df.copy()
    pretty["prob_%"] = pretty["prob"].apply(lambda x: fmt_prob(x, prob_decimals))
    pretty["odds_dec"] = pretty["odds"].apply(lambda x: fmt_odds(x, odds_decimals))
    pretty = pretty.drop(columns=["prob", "odds"])

    st.markdown("## Market prices (betting-style)")
    st.dataframe(pretty, use_container_width=True)

    st.download_button(
        "Download market_prices.csv",
        data=market_df.to_csv(index=False).encode("utf-8"),
        file_name="market_prices.csv",
        mime="text/csv",
    )

    if show_wide_table:
        st.markdown("## Wide table (model-friendly)")
        st.dataframe(wide_df, use_container_width=True)
        st.download_button(
            "Download forecasts_wide.csv",
            data=wide_df.to_csv(index=False).encode("utf-8"),
            file_name="forecasts_wide.csv",
            mime="text/csv",
        )

# -------------------------
# Tab: Help
# -------------------------
with tab_help:
    st.markdown("### CSV formats")

    st.markdown("**results.csv required columns**")
    st.code("team_home, team_away, goals_home, goals_away", language="text")
    st.markdown("Optional but recommended for time-decay:")
    st.code("date", language="text")

    st.markdown("**fixtures.csv required columns**")
    st.code("team_home, team_away", language="text")

    st.markdown("### Notes")
    st.markdown(
        "- If you enable **time-decay**, your results file needs a usable `date` column. "
        "The app will also let you filter the training window when `date` is present.\n"
        "- If any fixtures error, it’s almost always a **team name mismatch** between results and fixtures.\n"
        "- Exact Score / distributions require a newer `FootballProbabilityGrid` that exposes those methods."
    )


