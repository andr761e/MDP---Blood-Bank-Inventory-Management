from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

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
    plot_policy_scatter_all_states,
    plot_policy_scatter_weighted_states,
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
OUTPUT_DIR = Path("data/model_risk")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_XLSX_PATH = OUTPUT_DIR / f"model_risk_{ASSUMED_SCENARIO_NAME}_policy_under_{TRUE_SCENARIO_NAME}.xlsx"
PLOTS_DIR = OUTPUT_DIR / f"plots_{ASSUMED_SCENARIO_NAME}_policy_under_{TRUE_SCENARIO_NAME}"
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


def save_policy_plots(policy_df: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    plot_policy_heatmap_avg_order(policy_df, folder / "heatmap_avg_order.png")
    plot_policy_heatmap_weighted_order(policy_df, folder / "heatmap_weighted_order.png")
    plot_stationary_probability_by_stock(policy_df, folder / "heatmap_stationary_probability.png")
    plot_policy_scatter_all_states(policy_df, folder / "scatter_all_states.png")
    plot_policy_scatter_weighted_states(policy_df, folder / "scatter_weighted_states.png")
    plot_order_distribution_by_weekday(policy_df, folder / "order_distribution_by_weekday.png")


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
    cost_assumed_policy_under_assumed, _, policy_df_assumed_eval_assumed, compact_assumed_eval_assumed, visited_assumed_eval_assumed, weekday_assumed_eval_assumed = evaluate_policy_under_demand(
        states=states,
        det_policy=policy_assumed,
        policy_df=policy_df_assumed,
        demand_pmf=assumed_demand_pmf,
    )

    # ------------------------------------------------------------
    # 2. Evaluate assumed policy under true demand theta
    # ------------------------------------------------------------
    print("\nEvaluating ASSUMED policy under TRUE demand...")
    cost_assumed_policy_under_true, _, policy_df_assumed_eval_true, compact_assumed_eval_true, visited_assumed_eval_true, weekday_assumed_eval_true = evaluate_policy_under_demand(
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

    cost_true_policy_under_true, _, policy_df_true_eval_true, compact_true_eval_true, visited_true_eval_true, weekday_true_eval_true = evaluate_policy_under_demand(
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
    # 5. Save Excel outputs
    # ------------------------------------------------------------
    with pd.ExcelWriter(OUTPUT_XLSX_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="ModelRiskSummary", index=False)
        comparison_df.to_excel(writer, sheet_name="CostComparison", index=False)

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

    # ------------------------------------------------------------
    # 6. Save plots
    # ------------------------------------------------------------
    save_policy_plots(policy_df_assumed_eval_assumed, PLOTS_DIR / "assumed_policy_evaluated_under_assumed_demand")
    save_policy_plots(policy_df_assumed_eval_true, PLOTS_DIR / "assumed_policy_evaluated_under_true_demand")
    save_policy_plots(policy_df_true_eval_true, PLOTS_DIR / "true_policy_evaluated_under_true_demand")

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

    print("\nTop visited states for ASSUMED policy evaluated under TRUE demand:")
    print(visited_assumed_eval_true.head(PRINT_TOP_ROWS).to_string(index=False))

    print("\nSaved detailed model-risk outputs to:")
    print(f"  {OUTPUT_XLSX_PATH}")
    print(f"  {PLOTS_DIR}")


if __name__ == "__main__":
    main()
