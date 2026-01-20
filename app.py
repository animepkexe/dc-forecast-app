import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import penaltyblog as pb

st.set_page_config(page_title="Dixon–Coles Forecasting", layout="wide")

# -------------------------
# Core helpers
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

def parse_date_series_dayfirst(s: pd.Series) -> pd.Series:
    # Your format is 17/08/2024 -> dayfirst=True
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def ev_percent(p: float, betfair_odds: float, commission: float) -> float:
    """
    For £1 stake, with Betfair commission on winnings:
      net_return = 1 + (odds-1)*(1-commission)
      EV = p*net_return - 1
      EV% = 100*EV
    """
    p = safe_float(p)
    o = safe_float(betfair_odds)
    c = safe_float(commission)
    if np.isnan(p) or np.isnan(o) or o <= 1 or p <= 0 or p >= 1 or c < 0 or c >= 1:
        return np.nan
    net_return = 1.0 + (o - 1.0) * (1.0 - c)
    ev = p * net_return - 1.0
    return 100.0 * ev

# -------------------------
# CSV IO
# -------------------------
@st.cache_data(show_spinner=False)
def read_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(content))

# -------------------------
# penaltyblog fit (supports weights in newer versions)
# -------------------------
@st.cache_resource(show_spinner=True)
def fit_dc_model(home_goals, away_goals, home_teams, away_teams, weights=None, use_gradient=True):
    try:
        model = pb.models.DixonColesGoalModel(home_goals, away_goals, home_teams, away_teams, weights=weights)
    except TypeError:
        # older signature
        if weights is None:
            model = pb.models.DixonColesGoalModel(home_goals, away_goals, home_teams, away_teams)
        else:
            model = pb.models.DixonColesGoalModel(home_goals, away_goals, home_teams, away_teams, weights)

    try:
        model.fit(use_gradient=use_gradient)
    except TypeError:
        model.fit()

    return model

# -------------------------
# Probability grid market helpers (version-safe)
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
# Backtest utilities
# -------------------------
def odds_to_probs_1x2(oh, od, oa):
    """
    Convert 1X2 odds -> implied probs and remove overround.
    Returns (pH, pD, pA) or (nan,nan,nan) if invalid.
    """
    oh, od, oa = safe_float(oh), safe_float(od), safe_float(oa)
    if any(np.isnan(x) for x in (oh, od, oa)) or oh <= 1 or od <= 1 or oa <= 1:
        return np.nan, np.nan, np.nan
    pH, pD, pA = 1.0/oh, 1.0/od, 1.0/oa
    s = pH + pD + pA
    if s <= 0:
        return np.nan, np.nan, np.nan
    return pH/s, pD/s, pA/s

def log_loss_multiclass(p_vec, y_idx, eps=1e-15):
    p = np.array(p_vec, dtype=float)
    p = np.clip(p, eps, 1 - eps)
    p = p / p.sum()
    return -np.log(p[int(y_idx)])

def brier_multiclass(p_vec, y_idx):
    p = np.array(p_vec, dtype=float)
    p = p / p.sum()
    y = np.zeros_like(p)
    y[int(y_idx)] = 1.0
    return float(np.sum((p - y) ** 2))

def result_to_index(ftr: str):
    # Football-Data: "H","D","A"
    if ftr == "H":
        return 0
    if ftr == "D":
        return 1
    if ftr == "A":
        return 2
    return None

@st.cache_data(show_spinner=True)
def run_walk_forward_backtest(df: pd.DataFrame, xi: float, use_time_decay: bool, use_gradient: bool,
                              min_train_matches: int, max_goals: int):
    """
    Expanding-window walk-forward backtest:
      For each match i, fit on matches < i, predict i, compare vs Max odds implied probs.
    Returns per-match dataframe + summary.
    """
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    rows = []
    # Precompute weights if possible (for full set) then slice; penaltyblog helper expects date series
    # If not available in installed pb version, we fall back to None.
    weights_full = None
    if use_time_decay:
        try:
            weights_full = pb.models.dixon_coles_weights(df["date"], xi)
        except Exception:
            weights_full = None

    for i in range(len(df)):
        if i < min_train_matches:
            continue

        train = df.iloc[:i]
        test = df.iloc[i]

        # Training arrays
        home_goals = train["FTHG"].astype(float)
        away_goals = train["FTAG"].astype(float)
        home_teams = train["HomeTeam"].astype(str)
        away_teams = train["AwayTeam"].astype(str)

        weights = None
        if use_time_decay and weights_full is not None:
            weights = np.array(weights_full[:i], dtype=float)

        # Fit model
        try:
            model = fit_dc_model(home_goals, away_goals, home_teams, away_teams, weights=weights, use_gradient=use_gradient)
        except Exception:
            # If optimisation fails for some prefix, skip this test point
            continue

        # Predict
        try:
            try:
                grid = model.predict(str(test["HomeTeam"]), str(test["AwayTeam"]), max_goals=max_goals, normalize=True)
            except TypeError:
                grid = model.predict(str(test["HomeTeam"]), str(test["AwayTeam"]))
            pH_m, pD_m, pA_m = grid.home_draw_away
            pH_m, pD_m, pA_m = safe_float(pH_m), safe_float(pD_m), safe_float(pA_m)
        except Exception:
            continue

        # Market probs from Max odds
        pH_k, pD_k, pA_k = odds_to_probs_1x2(test["MaxH"], test["MaxD"], test["MaxA"])

        y = result_to_index(str(test["FTR"]))
        if y is None:
            continue
        if any(np.isnan(x) for x in (pH_k, pD_k, pA_k)):
            # If no odds, skip (since comparison to market is requested)
            continue

        # Scores
        ll_model = log_loss_multiclass([pH_m, pD_m, pA_m], y)
        ll_mkt = log_loss_multiclass([pH_k, pD_k, pA_k], y)
        br_model = brier_multiclass([pH_m, pD_m, pA_m], y)
        br_mkt = brier_multiclass([pH_k, pD_k, pA_k], y)

        rows.append({
            "date": test["date"],
            "HomeTeam": test["HomeTeam"],
            "AwayTeam": test["AwayTeam"],
            "FTR": test["FTR"],
            "pH_model": pH_m, "pD_model": pD_m, "pA_model": pA_m,
            "pH_mkt": pH_k, "pD_mkt": pD_k, "pA_mkt": pA_k,
            "ll_model": ll_model, "ll_mkt": ll_mkt,
            "brier_model": br_model, "brier_mkt": br_mkt,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {}

    summary = {
        "n": int(len(out)),
        "logloss_model": float(out["ll_model"].mean()),
        "logloss_market": float(out["ll_mkt"].mean()),
        "brier_model": float(out["brier_model"].mean()),
        "brier_market": float(out["brier_mkt"].mean()),
    }
    summary["logloss_diff_model_minus_market"] = summary["logloss_model"] - summary["logloss_market"]
    summary["brier_diff_model_minus_market"] = summary["brier_model"] - summary["brier_market"]

    return out, summary

def calibration_table(out_df: pd.DataFrame, which: str, bins: int = 10):
    """
    Simple calibration for HOME win probability:
    bucket predicted p_home into bins, compare avg p vs actual frequency.
    which: 'model' or 'mkt'
    """
    col = "pH_model" if which == "model" else "pH_mkt"
    df = out_df.copy()
    df["home_win"] = (df["FTR"] == "H").astype(int)
    df = df.dropna(subset=[col])
    df["bin"] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
    g = df.groupby("bin").agg(
        n=("home_win", "size"),
        avg_p=(col, "mean"),
        freq=("home_win", "mean"),
    ).reset_index()
    g["avg_p"] = g["avg_p"].astype(float)
    g["freq"] = g["freq"].astype(float)
    return g

# -------------------------
# App header
# -------------------------
st.title("⚽ Dixon–Coles Forecasting (Penaltyblog)")
st.caption("Forecast a match + enter Betfair odds for EV, and backtest vs Max market odds (1X2).")

# session state for forecast persistence
if "last_forecast" not in st.session_state:
    st.session_state["last_forecast"] = None
if "bf_inputs" not in st.session_state:
    st.session_state["bf_inputs"] = {}

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.subheader("Upload data")
    st.write("Use your historical CSV (e.g., Football-Data E0).")
    results_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.divider()
    st.subheader("Model options")
    use_time_decay = st.checkbox("Use time-decay weighting (requires Date)", value=True)
    xi = st.slider("Decay factor (xi)", 0.0, 0.01, 0.001, 0.0005)
    use_gradient = st.checkbox("Use gradient optimisation (if supported)", value=True)

    st.divider()
    st.subheader("Markets (Forecast tab)")
    include_1x2 = st.checkbox("1X2", value=True)
    include_double_chance = st.checkbox("Double Chance", value=True)
    include_dnb = st.checkbox("Draw No Bet", value=True)

    include_ou = st.checkbox("Over/Under", value=True)
    ou_line = st.selectbox("O/U line", [0.5, 1.5, 2.5, 3.5], index=2, disabled=not include_ou)

    include_btts = st.checkbox("BTTS", value=True)

    include_ah = st.checkbox("Asian Handicap", value=True)
    ah_line = st.selectbox("AH line (home)", [-1.5, -0.5, 0.0, 0.5, 1.5], index=3, disabled=not include_ah)

    include_exact_score = st.checkbox("Exact Score (Top N)", value=True)
    top_n_scores = st.slider("Top N scorelines", 5, 30, 10, disabled=not include_exact_score)

    include_win_to_nil = st.checkbox("Win to Nil", value=True)

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
tab_fit, tab_forecast, tab_backtest, tab_help = st.tabs(
    ["1) Fit", "2) Forecast + Betfair EV", "3) Backtest vs Max Odds", "Help"]
)

# -------------------------
# Fit tab: load CSV, standardize columns, fit "global" model for forecasting UI
# -------------------------
with tab_fit:
    if not results_file:
        st.info("Upload your historical CSV in the sidebar to begin.")
        st.stop()

    try:
        raw = read_csv(results_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Map Football-Data names -> internal names (for forecast tab fit)
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(raw.columns)
    if missing:
        st.error(f"CSV missing required columns: {sorted(missing)}")
        st.stop()

    df = raw.copy()
    df["date"] = parse_date_series_dayfirst(df["Date"])
    df = df.dropna(subset=["date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()

    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"]).copy()

    # Optional date filter for training
    min_d = df["date"].min()
    max_d = df["date"].max()

    d1, d2 = st.slider(
        "Training date range",
        min_value=min_d.to_pydatetime(),
        max_value=max_d.to_pydatetime(),
        value=(min_d.to_pydatetime(), max_d.to_pydatetime()),
    )
    train_df = df[(df["date"] >= pd.Timestamp(d1)) & (df["date"] <= pd.Timestamp(d2))].copy()
    train_df = train_df.sort_values("date").reset_index(drop=True)

    weights = None
    if use_time_decay:
        try:
            weights = pb.models.dixon_coles_weights(train_df["date"], xi)
        except Exception:
            weights = None
            st.warning("Time-decay weights not available in this environment; fitting without weights.")

    with st.spinner("Fitting Dixon–Coles model for forecasting..."):
        try:
            clf = fit_dc_model(
                train_df["FTHG"].astype(float),
                train_df["FTAG"].astype(float),
                train_df["HomeTeam"].astype(str),
                train_df["AwayTeam"].astype(str),
                weights=weights,
                use_gradient=use_gradient
            )
        except Exception as e:
            st.error(f"Model fit failed: {e}")
            st.stop()

    teams = sorted(set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"])))
    st.success("Model fitted ✅")
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches used", f"{len(train_df):,}")
    c2.metric("Teams", f"{len(teams):,}")
    c3.metric("Time-decay", "On" if (use_time_decay and weights is not None) else "Off")

    st.session_state["clf"] = clf
    st.session_state["teams"] = teams
    st.session_state["hist_df_for_backtest"] = df  # store cleaned df for backtest

# -------------------------
# Forecast + EV tab (single match)
# -------------------------
with tab_forecast:
    if "clf" not in st.session_state or "teams" not in st.session_state:
        st.info("Fit the model first in the **Fit** tab.")
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
        run_clicked = st.button("Run", type="primary", use_container_width=True)

    if home_team == away_team:
        st.error("Home and Away cannot be the same team.")
        st.stop()

    # Rerun logic: cache the simulation so typing odds doesn't bounce back
    cached = st.session_state["last_forecast"]
    settings_sig = {
        "include_1x2": include_1x2,
        "include_double_chance": include_double_chance,
        "include_dnb": include_dnb,
        "include_ou": include_ou,
        "ou_line": float(ou_line),
        "include_btts": include_btts,
        "include_ah": include_ah,
        "ah_line": float(ah_line),
        "include_exact_score": include_exact_score,
        "top_n_scores": int(top_n_scores),
        "include_win_to_nil": include_win_to_nil,
    }

    need_rerun = (
        run_clicked
        or cached is None
        or cached.get("home") != home_team
        or cached.get("away") != away_team
        or cached.get("settings") != settings_sig
    )

    if need_rerun:
        st.session_state["bf_inputs"] = {}

        try:
            try:
                grid = clf.predict(home_team, away_team, max_goals=15, normalize=True)
            except TypeError:
                grid = clf.predict(home_team, away_team)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        fixture_label = f"{home_team} vs {away_team}"
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
            maxg = 8
            for hg in range(0, maxg + 1):
                for ag in range(0, maxg + 1):
                    p = exact_score_prob(grid, hg, ag)
                    if not np.isnan(p) and p > 0:
                        candidates.append((hg, ag, p))
            candidates.sort(key=lambda x: x[2], reverse=True)
            for hg, ag, p in candidates[: int(top_n_scores)]:
                rows.append([fixture_label, "Correct Score", f"{hg}-{ag}", p, implied_decimal_odds(p)])

        market_df = pd.DataFrame(rows, columns=["fixture", "market", "selection", "model_prob", "model_odds"])
        st.session_state["last_forecast"] = {
            "home": home_team,
            "away": away_team,
            "settings": settings_sig,
            "market_df": market_df,
        }

    cached = st.session_state["last_forecast"]
    market_df = cached["market_df"].copy()

    st.markdown("### Model outputs + enter Betfair odds")
    st.caption("Edits rerun the app, but your latest simulation stays on screen.")

    # populate betfair odds from session map
    def get_bf(mkt, sel):
        return st.session_state["bf_inputs"].get((mkt, sel), np.nan)

    editor_df = market_df.copy()
    editor_df["betfair_odds"] = editor_df.apply(lambda r: get_bf(r["market"], r["selection"]), axis=1)

    edited = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        disabled=["fixture", "market", "selection", "model_prob", "model_odds"],
        column_config={
            "model_prob": st.column_config.NumberColumn("Model prob", format="%.6f", disabled=True),
            "model_odds": st.column_config.NumberColumn("Model odds (fair)", format="%.3f", disabled=True),
            "betfair_odds": st.column_config.NumberColumn("Betfair odds", format="%.3f"),
        },
        key="bf_editor",
    )

    # store edits
    new_map = {}
    for _, r in edited.iterrows():
        new_map[(r["market"], r["selection"])] = safe_float(r["betfair_odds"])
    st.session_state["bf_inputs"] = new_map

    out = edited.copy()
    out["EV_%"] = out.apply(lambda r: ev_percent(r["model_prob"], r["betfair_odds"], commission), axis=1)
    out["Value"] = out["EV_%"].apply(lambda x: (not np.isnan(safe_float(x))) and (safe_float(x) > 0.0))

    pretty = out.copy()
    pretty["model_prob"] = pretty["model_prob"].apply(lambda x: fmt_prob(x, prob_decimals))
    pretty["model_odds"] = pretty["model_odds"].apply(lambda x: fmt_odds(x, odds_decimals))
    pretty["betfair_odds"] = pretty["betfair_odds"].apply(lambda x: "" if np.isnan(safe_float(x)) else fmt_odds(x, odds_decimals))
    pretty["EV_%"] = pretty["EV_%"].apply(lambda x: "" if np.isnan(safe_float(x)) else f"{safe_float(x):.2f}%")

    if show_value_only:
        pretty = pretty[pretty["Value"] == True]

    st.markdown("### Value view")
    st.dataframe(
        pretty[["fixture", "market", "selection", "model_prob", "model_odds", "betfair_odds", "EV_%", "Value"]],
        use_container_width=True
    )

    st.metric("Value selections", int(out["Value"].sum()))

    st.download_button(
        "Download full table (with Betfair odds + EV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="model_vs_betfair_ev.csv",
        mime="text/csv",
    )

# -------------------------
# Backtest tab (1X2 vs MaxH/MaxD/MaxA)
# -------------------------
with tab_backtest:
    if "hist_df_for_backtest" not in st.session_state:
        st.info("Upload and fit first in the **Fit** tab.")
        st.stop()

    df = st.session_state["hist_df_for_backtest"].copy()

    # Validate required backtest columns
    needed = {"date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "MaxH", "MaxD", "MaxA"}
    missing = needed - set(df.columns)
    if missing:
        st.error(f"Your CSV is missing required backtest columns: {sorted(missing)}")
        st.stop()

    df["MaxH"] = pd.to_numeric(df["MaxH"], errors="coerce")
    df["MaxD"] = pd.to_numeric(df["MaxD"], errors="coerce")
    df["MaxA"] = pd.to_numeric(df["MaxA"], errors="coerce")

    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

    df = df.dropna(subset=["date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "MaxH", "MaxD", "MaxA"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    st.markdown("### Backtest settings (1X2 vs Max odds)")
    c1, c2, c3 = st.columns(3)
    with c1:
        min_train = st.number_input("Minimum training matches", min_value=200, max_value=5000, value=760, step=50,
                                    help="Roughly 2 seasons for Premier League (380*2).")
    with c2:
        max_goals = st.number_input("Max goals in model grid", min_value=8, max_value=20, value=15, step=1)
    with c3:
        bins = st.number_input("Calibration bins (home win)", min_value=5, max_value=20, value=10, step=1)

    # Date range selection
    min_d = df["date"].min()
    max_d = df["date"].max()
    d1, d2 = st.slider(
        "Evaluation date range",
        min_value=min_d.to_pydatetime(),
        max_value=max_d.to_pydatetime(),
        value=(min_d.to_pydatetime(), max_d.to_pydatetime()),
        help="This filters which matches are evaluated; training still uses all prior matches within the uploaded set."
    )

    eval_df = df[(df["date"] >= pd.Timestamp(d1)) & (df["date"] <= pd.Timestamp(d2))].copy()
    if eval_df.empty:
        st.warning("No matches in that evaluation range.")
        st.stop()

    run_bt = st.button("Run backtest", type="primary")
    if not run_bt:
        st.stop()

    # Run (cached) walk-forward on the *full* df but we'll display within eval range
    out_all, summary = run_walk_forward_backtest(
        df=df,
        xi=float(xi),
        use_time_decay=bool(use_time_decay),
        use_gradient=bool(use_gradient),
        min_train_matches=int(min_train),
        max_goals=int(max_goals),
    )

    if out_all.empty:
        st.error("Backtest produced no results. Try lowering minimum training matches or check your CSV content.")
        st.stop()

    out = out_all[(out_all["date"] >= pd.Timestamp(d1)) & (out_all["date"] <= pd.Timestamp(d2))].copy()
    if out.empty:
        st.warning("No evaluated matches after filtering (might be due to missing odds or min_train setting).")
        st.stop()

    st.subheader("Summary (lower is better)")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Matches evaluated", f"{len(out):,}")
    s2.metric("Log loss (Model)", f"{out['ll_model'].mean():.4f}")
    s3.metric("Log loss (Market Max)", f"{out['ll_mkt'].mean():.4f}")
    s4.metric("Model − Market", f"{(out['ll_model'].mean() - out['ll_mkt'].mean()):+.4f}")

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Brier (Model)", f"{out['brier_model'].mean():.4f}")
    t2.metric("Brier (Market Max)", f"{out['brier_mkt'].mean():.4f}")
    t3.metric("Model − Market", f"{(out['brier_model'].mean() - out['brier_mkt'].mean()):+.4f}")
    t4.metric("Time-decay", "On" if use_time_decay else "Off")

    st.markdown("### Cumulative log loss (Model vs Market)")
    plot_df = out.sort_values("date").copy()
    plot_df["cum_ll_model"] = plot_df["ll_model"].cumsum()
    plot_df["cum_ll_mkt"] = plot_df["ll_mkt"].cumsum()

    fig = plt.figure()
    plt.plot(plot_df["date"], plot_df["cum_ll_model"], label="Model")
    plt.plot(plot_df["date"], plot_df["cum_ll_mkt"], label="Market (Max)")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative log loss")
    st.pyplot(fig)

    st.markdown("### Calibration (Home win probability)")
    cal_model = calibration_table(out, which="model", bins=int(bins))
    cal_mkt = calibration_table(out, which="mkt", bins=int(bins))

    colL, colR = st.columns(2)
    with colL:
        st.write("Model")
        show = cal_model.copy()
        show["avg_p"] = show["avg_p"].apply(lambda x: fmt_prob(x, 1))
        show["freq"] = show["freq"].apply(lambda x: fmt_prob(x, 1))
        st.dataframe(show, use_container_width=True)
    with colR:
        st.write("Market (Max)")
        show = cal_mkt.copy()
        show["avg_p"] = show["avg_p"].apply(lambda x: fmt_prob(x, 1))
        show["freq"] = show["freq"].apply(lambda x: fmt_prob(x, 1))
        st.dataframe(show, use_container_width=True)

    st.markdown("### Per-match details")
    st.dataframe(
        plot_df[[
            "date","HomeTeam","AwayTeam","FTR",
            "pH_model","pD_model","pA_model",
            "pH_mkt","pD_mkt","pA_mkt",
            "ll_model","ll_mkt","brier_model","brier_mkt"
        ]],
        use_container_width=True
    )

    st.download_button(
        "Download backtest results CSV",
        data=plot_df.to_csv(index=False).encode("utf-8"),
        file_name="backtest_1x2_vs_max.csv",
        mime="text/csv",
    )

# -------------------------
# Help tab
# -------------------------
with tab_help:
    st.markdown("### Expected CSV (Football-Data style)")
    st.code("Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, MaxH, MaxD, MaxA", language="text")

    st.markdown("### What the backtest does")
    st.markdown(
        "- Walk-forward expanding window: for each match, fit on all prior matches.\n"
        "- Compare your model vs market-implied probabilities from Max odds (overround removed).\n"
        "- Report log loss and Brier score (lower is better).\n"
    )

    st.markdown("### Interpreting results")
    st.markdown(
        "- If **Model − Market** is negative (log loss), your model is beating the Max-odds benchmark on that sample.\n"
        "- If it’s positive, the market is better calibrated (still useful: it shows where to improve).\n"
    )


