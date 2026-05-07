from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse the model, LP solver, evaluation tools and plot functions from your main script.
# This file should be placed in the same folder as compute_average_cost_optimal_policy.py.
from compute_average_cost_optimal_policy import (
    # Parameters
    SHELF_LIFE,
    INVENTORY_CAP,
    MAX_ORDER,
    PRODUCTION_DAYS,
    DEMAND_SHEET_NAME,
    WEEKDAYS,
    PRINT_TOP_ROWS,
    C_OUTDATE,
    C_SHORTAGE,
    C_HOLDING,
    C_PRODUCTION,
    # State/action helpers
    State,
    load_demand_probabilities,
    enumerate_states,
    feasible_actions,
    extract_min_demand_by_day,
    compute_max_total_inventory_by_day,
    structurally_feasible_state,
    reachable_state_filter,
    assert_transition_closed,
    step_dynamics,
    # Optimization/evaluation
    solve_average_cost_lp,
    build_transition_matrix_under_policy,
    stationary_distribution_of_policy,
    build_compact_summary,
    build_state_probability_table,
    build_visited_state_table,
    build_weekday_recommendation_table,
    # Plot functions
    plot_policy_heatmap_avg_order,
    plot_policy_heatmap_weighted_order,
    plot_stationary_probability_by_stock,
    plot_order_distribution_by_weekday,
)


# ============================================================
# USER PARAMETERS
# ============================================================

# Baseline/assumed demand. Keep this name exactly as in your current workflow.
ASSUMED_DEMAND_XLSX_PATH = Path("weekday_demand_probabilities.xlsx")
ASSUMED_SCENARIO_NAME = "baseline"

# True demand scenario used for model-risk evaluation.
# Generate this with a separate demand generator, e.g. generate_demand_matrix_plus10.py.
TRUE_DEMAND_XLSX_PATH = Path("weekday_demand_probabilities_plus10.xlsx")
TRUE_SCENARIO_NAME = "plus10"

# Output folders/files
OUTPUT_DIR = Path("data/Model risk")
OUTPUT_XLSX_PATH = OUTPUT_DIR / f"model_risk_{ASSUMED_SCENARIO_NAME}_policy_under_{TRUE_SCENARIO_NAME}.xlsx"
PLOTS_DIR = OUTPUT_DIR / "plots_and_tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# COMMON STATE SPACE HELPERS
# ============================================================

def make_support_union_pmf(demand_pmfs: list[np.ndarray]) -> np.ndarray:
    """
    Construct an artificial demand PMF whose positive support is the union of
    the positive supports of all supplied demand PMFs.

    This is only used for state-space filtering and transition-closure checks.
    It is NOT used for optimization or cost evaluation.
    """
    if not demand_pmfs:
        raise ValueError("At least one demand PMF is required.")

    n_days = demand_pmfs[0].shape[0]
    max_k = max(pmf.shape[1] for pmf in demand_pmfs) - 1

    support_pmf = np.zeros((n_days, max_k + 1), dtype=float)

    for day in range(n_days):
        support = set()
        for pmf in demand_pmfs:
            positive = np.where(pmf[day, :] > 1e-12)[0]
            support.update(int(k) for k in positive)

        if not support:
            raise ValueError(f"No positive demand support found for {WEEKDAYS[day]}.")

        for k in support:
            support_pmf[day, k] = 1.0 / len(support)

    return support_pmf


def build_common_state_space(demand_pmfs: list[np.ndarray]) -> tuple[List[State], Dict[State, List[int]], int]:
    """
    Build one common state space that is valid for both the assumed and true
    demand models. This is essential for model-risk evaluation.
    """
    support_union_pmf = make_support_union_pmf(demand_pmfs)
    K_union = support_union_pmf.shape[1] - 1

    min_demand_by_day = extract_min_demand_by_day(support_union_pmf)
    max_total_by_day = compute_max_total_inventory_by_day(min_demand_by_day)

    print("Minimum demand with positive probability by weekday, union support:")
    print({WEEKDAYS[d]: k for d, k in min_demand_by_day.items()})
    print("Computed upper bound on total inventory by weekday, union support:")
    print({WEEKDAYS[d]: max_total_by_day[d] for d in range(7)})

    states = enumerate_states(INVENTORY_CAP, SHELF_LIFE)
    print(f"States before filtering: {len(states)}")

    states = [s for s in states if structurally_feasible_state(s, max_total_by_day)]
    print(f"States after structural filtering: {len(states)}")

    initial_state = (0,) + (0,) * (SHELF_LIFE - 1)

    states = reachable_state_filter(
        candidate_states=states,
        demand_pmf=support_union_pmf,
        inventory_cap=INVENTORY_CAP,
        max_order=MAX_ORDER,
        production_days=PRODUCTION_DAYS,
        shelf_life=SHELF_LIFE,
        initial_states=[initial_state],
    )
    print(f"States after reachability filtering: {len(states)}")

    # Check closure under the union support. This implies closure under each individual scenario.
    assert_transition_closed(
        states=states,
        demand_pmf=support_union_pmf,
        inventory_cap=INVENTORY_CAP,
        max_order=MAX_ORDER,
        production_days=PRODUCTION_DAYS,
        shelf_life=SHELF_LIFE,
    )

    actions_by_state = {
        s: feasible_actions(s, INVENTORY_CAP, MAX_ORDER, PRODUCTION_DAYS)
        for s in states
    }
    num_state_action_pairs = sum(len(v) for v in actions_by_state.values())

    return states, actions_by_state, num_state_action_pairs


# ============================================================
# POLICY EVALUATION AND OUTPUT HELPERS
# ============================================================

def evaluate_policy_under_demand(
    states: List[State],
    det_policy: Dict[State, int],
    policy_df: pd.DataFrame,
    demand_pmf: np.ndarray,
) -> tuple[float, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a fixed deterministic policy under a given demand distribution.
    Returns cost per day, stationary distribution, policy table with stationary probabilities,
    compact summary, visited states, and weekday recommendation table.
    """
    P, c_vec, _, _ = build_transition_matrix_under_policy(
        states=states,
        det_policy=det_policy,
        demand_pmf=demand_pmf,
        shelf_life=SHELF_LIFE,
    )
    pi = stationary_distribution_of_policy(P)
    average_cost_per_day = float(np.dot(pi, c_vec))

    evaluated_policy_df = policy_df.copy()
    evaluated_policy_df["stationary_probability"] = pi
    evaluated_policy_df = evaluated_policy_df.sort_values(
        ["weekday", "stationary_probability", "total_stock"],
        ascending=[True, False, True],
    )

    compact_df = build_compact_summary(evaluated_policy_df)
    state_prob_df = build_state_probability_table(states, pi, det_policy)
    visited_state_df = build_visited_state_table(state_prob_df, cutoff=1e-6)
    weekday_plan_df = build_weekday_recommendation_table(states, pi, det_policy)

    return average_cost_per_day, pi, evaluated_policy_df, compact_df, visited_state_df, weekday_plan_df



def compute_policy_cost_breakdown_under_demand(
    states: List[State],
    det_policy: Dict[State, int],
    demand_pmf: np.ndarray,
    stationary_probabilities: np.ndarray,
    case_name: str,
    optimized_under_scenario: str,
    evaluated_under_scenario: str,
    reference_average_cost_per_day: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decompose the long-run average immediate cost of a fixed policy under a
    specified demand model.

    The weighting is done with the stationary distribution of the Markov chain
    induced by the fixed policy under the evaluation demand model.
    """
    if len(states) != len(stationary_probabilities):
        raise ValueError("states and stationary_probabilities must have the same length.")

    units = {
        "holding": 0.0,
        "production": 0.0,
        "outdate": 0.0,
        "shortage": 0.0,
    }

    for s, state_probability in zip(states, stationary_probabilities):
        if state_probability <= 0.0:
            continue

        day = s[0]
        action = det_policy[s]
        probs = demand_pmf[day, :]

        for demand, demand_probability in enumerate(probs):
            if demand_probability <= 1e-12:
                continue

            _, shortage, outdate, holding = step_dynamics(
                state=s,
                action=action,
                demand=int(demand),
                shelf_life=SHELF_LIFE,
            )

            weight = float(state_probability) * float(demand_probability)
            units["holding"] += weight * holding
            units["production"] += weight * float(action)
            units["outdate"] += weight * outdate
            units["shortage"] += weight * shortage

    unit_costs = {
        "holding": C_HOLDING,
        "production": C_PRODUCTION,
        "outdate": C_OUTDATE,
        "shortage": C_SHORTAGE,
    }

    rows = []
    component_costs_per_day = {
        component: units[component] * unit_costs[component]
        for component in units
    }
    total_cost_per_day = float(sum(component_costs_per_day.values()))

    for component in ["holding", "production", "outdate", "shortage"]:
        avg_units_per_day = float(units[component])
        avg_cost_per_day = float(component_costs_per_day[component])
        rows.append({
            "case": case_name,
            "optimized_under_scenario": optimized_under_scenario,
            "evaluated_under_scenario": evaluated_under_scenario,
            "component": component,
            "unit_cost": float(unit_costs[component]),
            "average_units_per_day": avg_units_per_day,
            "average_units_per_week": 7.0 * avg_units_per_day,
            "average_cost_per_day": avg_cost_per_day,
            "average_cost_per_week": 7.0 * avg_cost_per_day,
            "share_of_case_total_cost": (
                avg_cost_per_day / total_cost_per_day
                if abs(total_cost_per_day) > 1e-12 else np.nan
            ),
        })

    rows.append({
        "case": case_name,
        "optimized_under_scenario": optimized_under_scenario,
        "evaluated_under_scenario": evaluated_under_scenario,
        "component": "total_immediate_cost",
        "unit_cost": np.nan,
        "average_units_per_day": np.nan,
        "average_units_per_week": np.nan,
        "average_cost_per_day": total_cost_per_day,
        "average_cost_per_week": 7.0 * total_cost_per_day,
        "share_of_case_total_cost": 1.0,
    })

    breakdown_df = pd.DataFrame(rows)

    check_df = pd.DataFrame([{
        "case": case_name,
        "optimized_under_scenario": optimized_under_scenario,
        "evaluated_under_scenario": evaluated_under_scenario,
        "breakdown_total_cost_per_day": total_cost_per_day,
        "reference_average_cost_per_day": reference_average_cost_per_day,
        "difference_vs_reference_cost_per_day": (
            total_cost_per_day - reference_average_cost_per_day
            if reference_average_cost_per_day is not None else np.nan
        ),
    }])

    return breakdown_df, check_df


def build_component_difference_table(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_case: str,
    right_case: str,
    comparison_name: str,
    difference_label: str,
) -> pd.DataFrame:
    """
    Build component-wise differences between two cost-breakdown cases.

    Difference is always left_case minus right_case.
    """
    left = left_df[left_df["case"] == left_case].copy()
    right = right_df[right_df["case"] == right_case].copy()

    keep_cols = [
        "component",
        "unit_cost",
        "average_units_per_day",
        "average_units_per_week",
        "average_cost_per_day",
        "average_cost_per_week",
    ]

    merged = left[keep_cols].merge(
        right[keep_cols],
        on="component",
        how="inner",
        suffixes=("_left", "_right"),
    )

    rows = []
    for row in merged.itertuples(index=False):
        component = row.component
        left_cost_day = float(row.average_cost_per_day_left)
        right_cost_day = float(row.average_cost_per_day_right)
        delta_cost_day = left_cost_day - right_cost_day

        left_units_day = row.average_units_per_day_left
        right_units_day = row.average_units_per_day_right
        if pd.isna(left_units_day) or pd.isna(right_units_day):
            delta_units_day = np.nan
        else:
            delta_units_day = float(left_units_day) - float(right_units_day)

        unit_cost_left = row.unit_cost_left
        unit_cost_right = row.unit_cost_right
        if pd.isna(unit_cost_left):
            unit_cost = np.nan
        elif pd.isna(unit_cost_right) or abs(float(unit_cost_left) - float(unit_cost_right)) <= 1e-12:
            unit_cost = float(unit_cost_left)
        else:
            unit_cost = np.nan

        rows.append({
            "comparison": comparison_name,
            "difference_definition": difference_label,
            "left_case": left_case,
            "right_case": right_case,
            "component": component,
            "unit_cost": unit_cost,
            "delta_units_per_day": delta_units_day,
            "delta_units_per_week": 7.0 * delta_units_day if not pd.isna(delta_units_day) else np.nan,
            "delta_cost_per_day": delta_cost_day,
            "delta_cost_per_week": 7.0 * delta_cost_day,
        })

    diff_df = pd.DataFrame(rows)
    total_delta = diff_df.loc[
        diff_df["component"] == "total_immediate_cost",
        "delta_cost_per_day",
    ]
    total_delta_value = float(total_delta.iloc[0]) if len(total_delta) else np.nan

    diff_df["share_of_total_delta_cost"] = diff_df["delta_cost_per_day"].apply(
        lambda x: x / total_delta_value if abs(total_delta_value) > 1e-12 else np.nan
    )

    return diff_df

# ============================================================
# REPORT-ORIENTED TABLES AND PLOTS
# ============================================================

FREQUENCY_TABLE_SCALE = 1_000_000
FREQUENCY_PROBABILITY_CUTOFF = 1e-12

CASE_SPECS = {
    "AA": {
        "case": "assumed_policy_evaluated_under_assumed_demand",
        "short_label": "Assumed policy / assumed demand",
        "folder": "assumed_policy_evaluated_under_assumed_demand",
    },
    "AT": {
        "case": "assumed_policy_evaluated_under_true_demand",
        "short_label": "Assumed policy / true demand",
        "folder": "assumed_policy_evaluated_under_true_demand",
    },
    "TT": {
        "case": "true_policy_evaluated_under_true_demand",
        "short_label": "True policy / true demand",
        "folder": "true_policy_evaluated_under_true_demand",
    },
}


def _integerize_scaled_probabilities(probs: pd.Series, scale: int) -> pd.Series:
    """Convert probabilities to integer frequencies summing exactly to ``scale``."""
    values = probs.astype(float).to_numpy()
    raw = values * scale
    floors = np.floor(raw).astype(int)

    remainder = int(scale - floors.sum())
    fractional = raw - floors

    if remainder > 0:
        order = np.argsort(-fractional)
        for pos in order[:remainder]:
            floors[pos] += 1
    elif remainder < 0:
        order = np.argsort(fractional)
        removed = 0
        for pos in order:
            if floors[pos] > 0:
                floors[pos] -= 1
                removed += 1
                if removed == abs(remainder):
                    break

    return pd.Series(floors.astype(int), index=probs.index)


def build_order_distribution_long(policy_df: pd.DataFrame, case_code: str, case_label: str) -> pd.DataFrame:
    """Conditional distribution P(optimal order = a | weekday) for one case."""
    rows = []
    grouped = (
        policy_df.groupby(["weekday", "optimal_order"], as_index=False)["stationary_probability"]
        .sum()
    )
    weekday_mass = policy_df.groupby("weekday")["stationary_probability"].sum().to_dict()

    for row in grouped.itertuples(index=False):
        mass = float(weekday_mass.get(row.weekday, 0.0))
        conditional_probability = float(row.stationary_probability) / mass if mass > 0 else np.nan
        rows.append({
            "case_code": case_code,
            "case_label": case_label,
            "weekday": row.weekday,
            "optimal_order": int(row.optimal_order),
            "stationary_probability_mass": float(row.stationary_probability),
            "conditional_probability_given_weekday": conditional_probability,
        })

    return pd.DataFrame(rows).sort_values(["case_code", "weekday", "optimal_order"])


def build_frequency_table_for_weekday(
    policy_df: pd.DataFrame,
    weekday: str,
    case_code: str,
    case_label: str,
    scale: int = FREQUENCY_TABLE_SCALE,
    min_mass: float = FREQUENCY_PROBABILITY_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a book-style state-action frequency table for one weekday and one case.

    Columns are pre-order total stock ``x``. Rows are post-order stock
    ``S_1 = x + a``. Entries are stationary probabilities conditional on the
    weekday, scaled to ``scale`` observations.
    """
    tmp = policy_df[
        (policy_df["weekday"] == weekday)
        & (policy_df["stationary_probability"] > min_mass)
    ].copy()

    if tmp.empty:
        raise ValueError(f"No positive stationary probability for {weekday} in case {case_code}.")

    total_mass = float(tmp["stationary_probability"].sum())
    tmp["conditional_probability"] = tmp["stationary_probability"] / total_mass
    tmp["post_order_stock_S1"] = tmp["total_stock"] + tmp["optimal_order"]

    grouped = (
        tmp.groupby(["post_order_stock_S1", "total_stock"], as_index=False)["conditional_probability"]
        .sum()
    )

    keys = list(zip(grouped["post_order_stock_S1"], grouped["total_stock"]))
    scaled = _integerize_scaled_probabilities(
        pd.Series(grouped["conditional_probability"].to_numpy(), index=keys),
        scale,
    )

    long_rows = []
    for (s1, x), freq in scaled.items():
        if freq <= 0:
            continue
        long_rows.append({
            "case_code": case_code,
            "case_label": case_label,
            "weekday": weekday,
            "total_stock": int(x),
            "post_order_stock_S1": int(s1),
            "frequency": int(freq),
            "probability_given_weekday": int(freq) / scale,
        })

    long_df = pd.DataFrame(long_rows)
    x_values = sorted(long_df["total_stock"].unique())
    s_values = sorted(long_df["post_order_stock_S1"].unique(), reverse=True)

    matrix = (
        long_df.pivot(index="post_order_stock_S1", columns="total_stock", values="frequency")
        .reindex(index=s_values, columns=x_values)
        .fillna(0)
        .astype(int)
    )

    table_rows = []
    table_rows.append(["Stock x"] + x_values + ["Freq(S_1)"])
    table_rows.append(["Up-to S_1"] + [""] * len(x_values) + [""])

    for s1 in s_values:
        row_vals = []
        for x in x_values:
            val = int(matrix.loc[s1, x])
            row_vals.append("" if val == 0 else val)
        table_rows.append([s1] + row_vals + [int(matrix.loc[s1].sum())])

    col_totals = matrix.sum(axis=0).astype(int)
    table_rows.append(["Freq(x)"] + [int(col_totals.loc[x]) for x in x_values] + [int(matrix.values.sum())])

    return pd.DataFrame(table_rows), long_df


def build_frequency_tables_for_case(
    policy_df: pd.DataFrame,
    case_code: str,
    case_label: str,
    scale: int = FREQUENCY_TABLE_SCALE,
) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Build frequency tables and dominant S_1 summary for all production weekdays."""
    table_by_sheet: Dict[str, pd.DataFrame] = {}
    long_tables = []
    summary_rows = []

    for day in sorted(PRODUCTION_DAYS):
        weekday = WEEKDAYS[day]
        table_df, long_df = build_frequency_table_for_weekday(
            policy_df=policy_df,
            weekday=weekday,
            case_code=case_code,
            case_label=case_label,
            scale=scale,
        )
        long_tables.append(long_df)

        s1_summary = long_df.groupby("post_order_stock_S1")["frequency"].sum().sort_values(ascending=False)
        dominant_s1 = int(s1_summary.index[0])
        dominant_freq = int(s1_summary.iloc[0])

        tmp = policy_df[
            (policy_df["weekday"] == weekday)
            & (policy_df["stationary_probability"] > FREQUENCY_PROBABILITY_CUTOFF)
        ].copy()
        mass = float(tmp["stationary_probability"].sum())
        expected_order = float((tmp["stationary_probability"] * tmp["optimal_order"]).sum() / mass)
        expected_start_stock = float((tmp["stationary_probability"] * tmp["total_stock"]).sum() / mass)

        summary_rows.append({
            "case_code": case_code,
            "case_label": case_label,
            "weekday": weekday,
            "dominant_post_order_stock_S1": dominant_s1,
            "dominant_S1_frequency": dominant_freq,
            "dominant_S1_share": dominant_freq / scale,
            "expected_start_stock": expected_start_stock,
            "expected_order": expected_order,
            "expected_post_order_stock": expected_start_stock + expected_order,
        })

        table_by_sheet[f"Freq_{case_code}_{weekday[:3]}"] = table_df

    return (
        table_by_sheet,
        pd.concat(long_tables, ignore_index=True) if long_tables else pd.DataFrame(),
        pd.DataFrame(summary_rows),
    )


def build_all_frequency_tables_for_cases(
    case_policy_dfs: Dict[str, pd.DataFrame],
) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    all_tables: Dict[str, pd.DataFrame] = {}
    all_long = []
    all_summary = []

    for case_code, policy_df in case_policy_dfs.items():
        spec = CASE_SPECS[case_code]
        tables, long_df, summary_df = build_frequency_tables_for_case(
            policy_df=policy_df,
            case_code=case_code,
            case_label=spec["short_label"],
        )
        all_tables.update(tables)
        all_long.append(long_df)
        all_summary.append(summary_df)

    return (
        all_tables,
        pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame(),
        pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame(),
    )


def plot_frequency_table_image(table_df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Save a PNG rendering of one book-style frequency table."""
    ncols = table_df.shape[1]
    nrows = table_df.shape[0]
    fig_width = max(10, 0.55 * ncols)
    fig_height = max(2.5, 0.35 * nrows + 1.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)

    table = ax.table(cellText=table_df.astype(str).values, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_frequency_table_plots(
    frequency_tables: Dict[str, pd.DataFrame],
    plots_dir: Path = PLOTS_DIR,
) -> Dict[str, Path]:
    paths = {}
    for sheet_name, table_df in frequency_tables.items():
        _, case_code, day_abbrev = sheet_name.split("_")
        spec = CASE_SPECS[case_code]
        folder = plots_dir / spec["folder"] / "frequency_tables"
        folder.mkdir(parents=True, exist_ok=True)
        weekday = next(day for day in WEEKDAYS if day.startswith(day_abbrev))
        title = (
            f"(State, action)-frequency table for 1,000,000 {weekday}s\n"
            f"{spec['short_label']}"
        )
        path = folder / f"frequency_table_{weekday.lower()}.png"
        plot_frequency_table_image(table_df, title, path)
        paths[sheet_name] = path
    return paths


def plot_cost_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    df = comparison_df[comparison_df["case"] != "Model-risk loss"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    ax.bar(x, df["cost_per_day"].astype(float).to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(["Assumed/assumed", "Assumed/true", "True/true"], rotation=20, ha="right")
    ax.set_ylabel("Average cost per day")
    ax.set_title("Model-risk cost comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_component_delta(diff_df: pd.DataFrame, output_path: Path, title: str) -> None:
    df = diff_df[diff_df["component"] != "total_immediate_cost"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    ax.bar(x, df["delta_cost_per_day"].astype(float).to_numpy())
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["component"].astype(str).to_list(), rotation=20, ha="right")
    ax.set_ylabel("Delta cost per day")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_policy_plots(policy_df: pd.DataFrame, folder: Path) -> None:
    """Save only the report-relevant plots. No scatter plots."""
    folder.mkdir(parents=True, exist_ok=True)
    plot_stationary_probability_by_stock(policy_df, folder / "heatmap_stationary_probability.png")
    plot_policy_heatmap_weighted_order(policy_df, folder / "heatmap_weighted_order.png")
    plot_order_distribution_by_weekday(policy_df, folder / "order_distribution_by_weekday.png")
    plot_policy_heatmap_avg_order(policy_df, folder / "heatmap_avg_order_appendix.png")


# ============================================================
# MAIN MODEL-RISK ANALYSIS
# ============================================================

def main() -> None:
    if SHELF_LIFE < 2:
        raise ValueError("SHELF_LIFE must be at least 2.")

    print("=" * 80)
    print("MODEL RISK ANALYSIS")
    print("=" * 80)
    print(f"Assumed demand file: {ASSUMED_DEMAND_XLSX_PATH}")
    print(f"True demand file:    {TRUE_DEMAND_XLSX_PATH}")
    print("=" * 80)

    assumed_demand_pmf, K_assumed = load_demand_probabilities(
        ASSUMED_DEMAND_XLSX_PATH, DEMAND_SHEET_NAME
    )
    true_demand_pmf, K_true = load_demand_probabilities(
        TRUE_DEMAND_XLSX_PATH, DEMAND_SHEET_NAME
    )

    states, actions_by_state, num_state_action_pairs = build_common_state_space(
        [assumed_demand_pmf, true_demand_pmf]
    )

    print("=" * 80)
    print("COMMON MODEL SETUP")
    print("=" * 80)
    print(f"Max demand K, assumed:       {K_assumed}")
    print(f"Max demand K, true:          {K_true}")
    print(f"Shelf life:                  {SHELF_LIFE}")
    print(f"Inventory cap:               {INVENTORY_CAP}")
    print(f"Max order:                   {MAX_ORDER}")
    print(f"Production days:             {[WEEKDAYS[d] for d in sorted(PRODUCTION_DAYS)]}")
    print(f"Number of common states:     {len(states)}")
    print(f"State-action pairs:          {num_state_action_pairs}")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. Optimize under assumed demand model theta_hat
    # ------------------------------------------------------------
    print("\nSolving policy optimized under ASSUMED demand...")
    g_assumed, policy_assumed, policy_df_assumed = solve_average_cost_lp(
        states=states,
        actions_by_state=actions_by_state,
        demand_pmf=assumed_demand_pmf,
        shelf_life=SHELF_LIFE,
    )

    # Evaluate assumed policy under assumed demand, mostly as a consistency check.
    cost_assumed_policy_under_assumed, pi_assumed_policy_under_assumed, policy_df_assumed_eval_assumed, compact_assumed_eval_assumed, visited_assumed_eval_assumed, weekday_assumed_eval_assumed = evaluate_policy_under_demand(
        states=states,
        det_policy=policy_assumed,
        policy_df=policy_df_assumed,
        demand_pmf=assumed_demand_pmf,
    )

    # ------------------------------------------------------------
    # 2. Evaluate assumed policy under true demand theta
    # ------------------------------------------------------------
    print("\nEvaluating ASSUMED policy under TRUE demand...")
    cost_assumed_policy_under_true, pi_assumed_policy_under_true, policy_df_assumed_eval_true, compact_assumed_eval_true, visited_assumed_eval_true, weekday_assumed_eval_true = evaluate_policy_under_demand(
        states=states,
        det_policy=policy_assumed,
        policy_df=policy_df_assumed,
        demand_pmf=true_demand_pmf,
    )

    # ------------------------------------------------------------
    # 3. Optimize under true demand theta
    # ------------------------------------------------------------
    print("\nSolving policy optimized under TRUE demand...")
    g_true, policy_true, policy_df_true = solve_average_cost_lp(
        states=states,
        actions_by_state=actions_by_state,
        demand_pmf=true_demand_pmf,
        shelf_life=SHELF_LIFE,
    )

    cost_true_policy_under_true, pi_true_policy_under_true, policy_df_true_eval_true, compact_true_eval_true, visited_true_eval_true, weekday_true_eval_true = evaluate_policy_under_demand(
        states=states,
        det_policy=policy_true,
        policy_df=policy_df_true,
        demand_pmf=true_demand_pmf,
    )

    # ------------------------------------------------------------
    # 4. Model-risk loss
    # ------------------------------------------------------------
    model_risk_loss_per_day = cost_assumed_policy_under_true - cost_true_policy_under_true
    model_risk_loss_per_week = 7.0 * model_risk_loss_per_day
    model_risk_loss_pct = (
        100.0 * model_risk_loss_per_day / cost_true_policy_under_true
        if abs(cost_true_policy_under_true) > 1e-12 else np.nan
    )

    cost_breakdown_assumed_eval_assumed, cost_breakdown_check_assumed_eval_assumed = compute_policy_cost_breakdown_under_demand(
        states=states,
        det_policy=policy_assumed,
        demand_pmf=assumed_demand_pmf,
        stationary_probabilities=pi_assumed_policy_under_assumed,
        case_name="assumed_policy_evaluated_under_assumed_demand",
        optimized_under_scenario=ASSUMED_SCENARIO_NAME,
        evaluated_under_scenario=ASSUMED_SCENARIO_NAME,
        reference_average_cost_per_day=cost_assumed_policy_under_assumed,
    )

    cost_breakdown_assumed_eval_true, cost_breakdown_check_assumed_eval_true = compute_policy_cost_breakdown_under_demand(
        states=states,
        det_policy=policy_assumed,
        demand_pmf=true_demand_pmf,
        stationary_probabilities=pi_assumed_policy_under_true,
        case_name="assumed_policy_evaluated_under_true_demand",
        optimized_under_scenario=ASSUMED_SCENARIO_NAME,
        evaluated_under_scenario=TRUE_SCENARIO_NAME,
        reference_average_cost_per_day=cost_assumed_policy_under_true,
    )

    cost_breakdown_true_eval_true, cost_breakdown_check_true_eval_true = compute_policy_cost_breakdown_under_demand(
        states=states,
        det_policy=policy_true,
        demand_pmf=true_demand_pmf,
        stationary_probabilities=pi_true_policy_under_true,
        case_name="true_policy_evaluated_under_true_demand",
        optimized_under_scenario=TRUE_SCENARIO_NAME,
        evaluated_under_scenario=TRUE_SCENARIO_NAME,
        reference_average_cost_per_day=cost_true_policy_under_true,
    )

    cost_breakdown_df = pd.concat(
        [
            cost_breakdown_assumed_eval_assumed,
            cost_breakdown_assumed_eval_true,
            cost_breakdown_true_eval_true,
        ],
        ignore_index=True,
    )

    cost_breakdown_check_df = pd.concat(
        [
            cost_breakdown_check_assumed_eval_assumed,
            cost_breakdown_check_assumed_eval_true,
            cost_breakdown_check_true_eval_true,
        ],
        ignore_index=True,
    )

    cost_breakdown_model_risk_loss_df = build_component_difference_table(
        left_df=cost_breakdown_df,
        right_df=cost_breakdown_df,
        left_case="assumed_policy_evaluated_under_true_demand",
        right_case="true_policy_evaluated_under_true_demand",
        comparison_name="component_wise_model_risk_loss",
        difference_label="assumed_policy_under_true_demand_minus_true_policy_under_true_demand",
    )

    cost_breakdown_demand_shift_df = build_component_difference_table(
        left_df=cost_breakdown_df,
        right_df=cost_breakdown_df,
        left_case="assumed_policy_evaluated_under_true_demand",
        right_case="assumed_policy_evaluated_under_assumed_demand",
        comparison_name="component_wise_effect_of_true_demand_shift_on_assumed_policy",
        difference_label="assumed_policy_under_true_demand_minus_assumed_policy_under_assumed_demand",
    )

    summary_df = pd.DataFrame({
        "metric": [
            "assumed_scenario",
            "true_scenario",
            "assumed_demand_file",
            "true_demand_file",
            "number_of_common_states",
            "number_of_state_action_pairs",
            "optimal_cost_per_day_assumed_model",
            "optimal_cost_per_week_assumed_model",
            "assumed_policy_cost_per_day_under_assumed_demand",
            "assumed_policy_cost_per_week_under_assumed_demand",
            "assumed_policy_cost_per_day_under_true_demand",
            "assumed_policy_cost_per_week_under_true_demand",
            "optimal_cost_per_day_true_model",
            "optimal_cost_per_week_true_model",
            "model_risk_loss_per_day",
            "model_risk_loss_per_week",
            "model_risk_loss_percent_of_true_optimum",
            "shelf_life",
            "inventory_cap",
            "max_order",
            "c_outdate",
            "c_shortage",
            "c_holding",
            "c_production",
        ],
        "value": [
            ASSUMED_SCENARIO_NAME,
            TRUE_SCENARIO_NAME,
            str(ASSUMED_DEMAND_XLSX_PATH),
            str(TRUE_DEMAND_XLSX_PATH),
            len(states),
            num_state_action_pairs,
            g_assumed,
            7.0 * g_assumed,
            cost_assumed_policy_under_assumed,
            7.0 * cost_assumed_policy_under_assumed,
            cost_assumed_policy_under_true,
            7.0 * cost_assumed_policy_under_true,
            g_true,
            7.0 * g_true,
            model_risk_loss_per_day,
            model_risk_loss_per_week,
            model_risk_loss_pct,
            SHELF_LIFE,
            INVENTORY_CAP,
            MAX_ORDER,
            C_OUTDATE,
            C_SHORTAGE,
            C_HOLDING,
            C_PRODUCTION,
        ]
    })

    comparison_df = pd.DataFrame({
        "case": [
            "Policy optimized under assumed demand, evaluated under assumed demand",
            "Policy optimized under assumed demand, evaluated under true demand",
            "Policy optimized under true demand, evaluated under true demand",
            "Model-risk loss",
        ],
        "cost_per_day": [
            cost_assumed_policy_under_assumed,
            cost_assumed_policy_under_true,
            cost_true_policy_under_true,
            model_risk_loss_per_day,
        ],
        "cost_per_week": [
            7.0 * cost_assumed_policy_under_assumed,
            7.0 * cost_assumed_policy_under_true,
            7.0 * cost_true_policy_under_true,
            model_risk_loss_per_week,
        ],
    })

    # ------------------------------------------------------------
    # 4b. Report-oriented policy structure tables
    # ------------------------------------------------------------
    case_policy_dfs = {
        "AA": policy_df_assumed_eval_assumed,
        "AT": policy_df_assumed_eval_true,
        "TT": policy_df_true_eval_true,
    }

    frequency_tables, frequency_long_df, dominant_order_up_to_df = build_all_frequency_tables_for_cases(
        case_policy_dfs
    )

    order_distribution_long_df = pd.concat(
        [
            build_order_distribution_long(
                policy_df=case_policy_dfs[case_code],
                case_code=case_code,
                case_label=CASE_SPECS[case_code]["short_label"],
            )
            for case_code in ["AA", "AT", "TT"]
        ],
        ignore_index=True,
    )

    # ------------------------------------------------------------
    # 5. Save Excel outputs
    # ------------------------------------------------------------
    with pd.ExcelWriter(OUTPUT_XLSX_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="ModelRiskSummary", index=False)
        comparison_df.to_excel(writer, sheet_name="CostComparison", index=False)
        cost_breakdown_df.to_excel(writer, sheet_name="CostBreakdown", index=False)
        cost_breakdown_model_risk_loss_df.to_excel(writer, sheet_name="CostBreakdown_ModelRiskLoss", index=False)
        cost_breakdown_demand_shift_df.to_excel(writer, sheet_name="CostBreakdown_DemandShift", index=False)
        cost_breakdown_check_df.to_excel(writer, sheet_name="CostBreakdown_Checks", index=False)

        policy_df_assumed_eval_assumed.to_excel(writer, sheet_name="AssumedPolicy_EvalAssumed", index=False)
        weekday_assumed_eval_assumed.to_excel(writer, sheet_name="Weekday_Assumed_EvalAss", index=False)
        visited_assumed_eval_assumed.to_excel(writer, sheet_name="Visited_Assumed_EvalAss", index=False)
        compact_assumed_eval_assumed.to_excel(writer, sheet_name="Compact_Assumed_EvalAss", index=False)

        policy_df_assumed_eval_true.to_excel(writer, sheet_name="AssumedPolicy_EvalTrue", index=False)
        weekday_assumed_eval_true.to_excel(writer, sheet_name="Weekday_Assumed_EvalTrue", index=False)
        visited_assumed_eval_true.to_excel(writer, sheet_name="Visited_Assumed_EvalTrue", index=False)
        compact_assumed_eval_true.to_excel(writer, sheet_name="Compact_Assumed_EvalTrue", index=False)

        policy_df_true_eval_true.to_excel(writer, sheet_name="TruePolicy_EvalTrue", index=False)
        weekday_true_eval_true.to_excel(writer, sheet_name="Weekday_True_EvalTrue", index=False)
        visited_true_eval_true.to_excel(writer, sheet_name="Visited_True_EvalTrue", index=False)
        compact_true_eval_true.to_excel(writer, sheet_name="Compact_True_EvalTrue", index=False)

        dominant_order_up_to_df.to_excel(writer, sheet_name="DominantOrderUpTo", index=False)
        frequency_long_df.to_excel(writer, sheet_name="FrequencyTables_Long", index=False)
        order_distribution_long_df.to_excel(writer, sheet_name="OrderDistribution_Long", index=False)

        # Book-style frequency tables. Sheet names are kept short to satisfy Excel's 31-char limit.
        for sheet_name, table_df in frequency_tables.items():
            table_df.to_excel(writer, sheet_name=sheet_name[:31], index=False, header=False)

    # ------------------------------------------------------------
    # 6. Save plots
    # ------------------------------------------------------------
    save_policy_plots(policy_df_assumed_eval_assumed, PLOTS_DIR / "assumed_policy_evaluated_under_assumed_demand")
    save_policy_plots(policy_df_assumed_eval_true, PLOTS_DIR / "assumed_policy_evaluated_under_true_demand")
    save_policy_plots(policy_df_true_eval_true, PLOTS_DIR / "true_policy_evaluated_under_true_demand")

    comparison_plots_dir = PLOTS_DIR / "comparison"
    comparison_plots_dir.mkdir(parents=True, exist_ok=True)
    plot_cost_comparison(comparison_df, comparison_plots_dir / "cost_comparison.png")
    plot_component_delta(
        cost_breakdown_model_risk_loss_df,
        comparison_plots_dir / "model_risk_loss_by_component.png",
        "Model-risk loss by cost component",
    )
    plot_component_delta(
        cost_breakdown_demand_shift_df,
        comparison_plots_dir / "demand_shift_effect_by_component.png",
        "Effect of evaluating the assumed policy under true demand",
    )
    generate_frequency_table_plots(frequency_tables, PLOTS_DIR)

    # ------------------------------------------------------------
    # 7. Print results
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODEL RISK RESULTS")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("\nModel-risk loss:")
    print(f"  Per day:   {model_risk_loss_per_day:.6f}")
    print(f"  Per week:  {model_risk_loss_per_week:.6f}")
    print(f"  Percent:   {model_risk_loss_pct:.4f}% of true optimum")

    print("\nMost readable weekday plan for ASSUMED policy evaluated under TRUE demand:")
    print(weekday_assumed_eval_true.to_string(index=False))

    print("\nDominant order-up-to levels:")
    print(dominant_order_up_to_df.to_string(index=False))

    print("\nTop visited states for ASSUMED policy evaluated under TRUE demand:")
    print(visited_assumed_eval_true.head(PRINT_TOP_ROWS).to_string(index=False))

    print("\nSaved detailed model-risk outputs to:")
    print(f"  {OUTPUT_XLSX_PATH}")
    print(f"  {PLOTS_DIR}")


if __name__ == "__main__":
    main()
