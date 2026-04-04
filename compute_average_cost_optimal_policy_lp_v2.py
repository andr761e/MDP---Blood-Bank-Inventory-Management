from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB


# ============================================================
# USER PARAMETERS
# ============================================================

# Excel input with rows Monday,...,Sunday and columns demand_0,...,demand_K
DEMAND_XLSX_PATH = Path("weekday_demand_probabilities.xlsx")
DEMAND_SHEET_NAME = "DemandProbabilities"

# Model structure
SHELF_LIFE = 5                 # platelet shelf life from purchase
INVENTORY_CAP = 8              # max total inventory allowed immediately after ordering
MAX_ORDER = 5                  # max units that can be ordered on a production day
PRODUCTION_DAYS = {0, 1, 2, 3, 4}   # Monday-Friday; no production on weekend

# Costs
C_OUTDATE = 4.0                # c_0
C_SHORTAGE = 30.0              # c_s
C_HOLDING = 1.0                # c_H
C_PRODUCTION = 0.0             # optional production cost

# Output file
OUTPUT_XLSX_PATH = Path("data/optimal_stationary_policy_lp_v2.xlsx")

# Numerical tolerances
REDUCED_COST_TOL = 1e-8
PRINT_TOP_ROWS = 30


# ============================================================
# FIXED WEEKDAY ORDER
# ============================================================

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ============================================================
# INPUT: DEMAND PROBABILITIES
# ============================================================

def load_demand_probabilities(path: Path, sheet_name: str) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Demand file not found: {path}\n"
            "Create it first with the demand-matrix generator script."
        )

    df = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
    df.index = [str(x).strip() for x in df.index]

    demand_cols = [c for c in df.columns if str(c).startswith("demand_")]
    if not demand_cols:
        raise ValueError("No columns named demand_0, demand_1, ... found in the Excel file.")

    demand_cols = sorted(demand_cols, key=lambda x: int(str(x).split("_")[1]))
    missing_days = [d for d in WEEKDAYS if d not in df.index]
    if missing_days:
        raise ValueError(f"Missing weekday rows in Excel file: {missing_days}")

    df = df.loc[WEEKDAYS, demand_cols].astype(float)
    arr = df.to_numpy()

    if np.any(arr < -1e-12):
        raise ValueError("Demand probabilities must be nonnegative.")

    row_sums = arr.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("Each weekday row must have positive total probability.")

    arr = arr / row_sums[:, None]
    K = len(demand_cols) - 1
    return arr, K


# ============================================================
# STATE AND ACTION SPACE
# ============================================================

State = Tuple[int, ...]


def all_inventory_vectors(cap: int, dims: int):
    for vec in product(range(cap + 1), repeat=dims):
        if sum(vec) <= cap:
            yield vec


def enumerate_states(inventory_cap: int, shelf_life: int) -> List[State]:
    states = []
    for day in range(7):
        for inv in all_inventory_vectors(inventory_cap, shelf_life - 1):
            states.append((day,) + tuple(inv))
    return states


def feasible_actions(
    state: State,
    inventory_cap: int,
    max_order: int,
    production_days: set[int],
) -> List[int]:
    day = state[0]
    current_stock = sum(state[1:])

    if day not in production_days:
        return [0]

    max_feasible = min(max_order, inventory_cap - current_stock)
    return list(range(max_feasible + 1))


# ============================================================
# DYNAMICS: ACTION -> DEMAND -> FIFO -> AGEING
# ============================================================

def step_dynamics(
    state: State,
    action: int,
    demand: int,
    shelf_life: int,
) -> tuple[State, float, float, float]:
    day = state[0]
    inv = list(state[1:])

    stock = inv + [action]   # fresh stock enters with shelf_life days remaining
    remaining_demand = demand
    remaining_stock = stock[:]

    # FIFO = oldest first
    for i in range(shelf_life):
        used = min(remaining_stock[i], remaining_demand)
        remaining_stock[i] -= used
        remaining_demand -= used

    shortage = float(remaining_demand)
    outdate = float(remaining_stock[0])

    next_inv = tuple(remaining_stock[1:])
    next_day = (day + 1) % 7
    next_state = (next_day,) + next_inv

    holding = float(sum(next_inv))
    return next_state, shortage, outdate, holding


def transition_distribution_and_expected_cost(
    state: State,
    action: int,
    demand_pmf: np.ndarray,
    shelf_life: int,
) -> tuple[Dict[State, float], float]:
    day = state[0]
    probs = demand_pmf[day, :]

    dist: Dict[State, float] = {}
    expected_cost = 0.0

    for demand, p in enumerate(probs):
        next_state, shortage, outdate, holding = step_dynamics(state, action, demand, shelf_life)
        cost = (
            C_OUTDATE * outdate
            + C_SHORTAGE * shortage
            + C_HOLDING * holding
            + C_PRODUCTION * action
        )
        dist[next_state] = dist.get(next_state, 0.0) + float(p)
        expected_cost += float(p) * cost

    return dist, expected_cost


# ============================================================
# SOLVE AVERAGE-COST MDP BY DUAL LINEAR PROGRAM
# ============================================================

def solve_average_cost_lp(
    states: List[State],
    actions_by_state: Dict[State, List[int]],
    demand_pmf: np.ndarray,
    shelf_life: int,
):
    state_index = {s: i for i, s in enumerate(states)}

    transitions = {}
    expected_cost = {}

    for s in states:
        for a in actions_by_state[s]:
            dist, cost = transition_distribution_and_expected_cost(s, a, demand_pmf, shelf_life)
            transitions[(s, a)] = dist
            expected_cost[(s, a)] = cost

    model = gp.Model("average_cost_dual_lp")
    model.Params.OutputFlag = 0

    g = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="g")
    h = {
        s: model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"h_{state_index[s]}")
        for s in states
    }

    for s in states:
        for a in actions_by_state[s]:
            rhs = expected_cost[(s, a)] + gp.quicksum(
                p * h[ns] for ns, p in transitions[(s, a)].items()
            )
            model.addConstr(g + h[s] <= rhs, name=f"acoe_{state_index[s]}_{a}")

    model.setObjective(g, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find an optimal solution. Status = {model.Status}")

    g_star = float(g.X)
    h_star = {s: float(h[s].X) for s in states}

    det_policy = {}
    policy_rows = []

    for s in states:
        scores = {}
        for a in actions_by_state[s]:
            score = expected_cost[(s, a)] + sum(
                p * h_star[ns] for ns, p in transitions[(s, a)].items()
            )
            scores[a] = score

        min_score = min(scores.values())
        best_actions = [a for a, val in scores.items() if abs(val - min_score) <= REDUCED_COST_TOL]
        chosen_action = min(best_actions)
        det_policy[s] = chosen_action

        row = {
            "weekday": WEEKDAYS[s[0]],
            "total_stock": sum(s[1:]),
            "optimal_order": chosen_action,
            "bellman_min_value": min_score,
            "num_tied_best_actions": len(best_actions),
            "tied_best_actions": str(best_actions),
        }
        for i in range(len(s) - 1):
            row[f"x{i+1}"] = s[i+1]
        policy_rows.append(row)

    policy_df = pd.DataFrame(policy_rows)
    return g_star, det_policy, policy_df


# ============================================================
# EVALUATE THE EXTRACTED POLICY
# ============================================================

def build_transition_matrix_under_policy(
    states: List[State],
    det_policy: Dict[State, int],
    demand_pmf: np.ndarray,
    shelf_life: int,
):
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((n, n), dtype=float)
    c = np.zeros(n, dtype=float)
    a_vec = np.zeros(n, dtype=float)

    for s in states:
        i = idx[s]
        a = det_policy[s]
        dist, expected_cost = transition_distribution_and_expected_cost(s, a, demand_pmf, shelf_life)
        c[i] = expected_cost
        a_vec[i] = a
        for ns, p in dist.items():
            j = idx[ns]
            P[i, j] += p

    return P, c, a_vec, idx


def stationary_distribution_of_policy(P: np.ndarray) -> np.ndarray:
    """
    Solve pi = pi P, sum pi = 1
    by replacing one equation with normalization.
    """
    n = P.shape[0]
    A = P.T - np.eye(n)
    b = np.zeros(n)

    A[-1, :] = 1.0
    b[-1] = 1.0

    pi = np.linalg.solve(A, b)
    pi = np.maximum(pi, 0.0)
    pi = pi / pi.sum()
    return pi


def build_state_probability_table(states: List[State], pi: np.ndarray, det_policy: Dict[State, int]) -> pd.DataFrame:
    rows = []
    for s, prob in zip(states, pi):
        row = {
            "weekday": WEEKDAYS[s[0]],
            "stationary_probability": prob,
            "total_stock": sum(s[1:]),
            "optimal_order": det_policy[s],
        }
        for i in range(len(s) - 1):
            row[f"x{i+1}"] = s[i+1]
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["stationary_probability", "weekday", "total_stock"], ascending=[False, True, True])
    return df


def build_weekday_recommendation_table(
    states: List[State],
    pi: np.ndarray,
    det_policy: Dict[State, int],
) -> pd.DataFrame:
    """
    This gives the most human-readable summary:
    conditional on weekday, what does the optimal stationary policy typically order?
    """
    rows = []
    for day in range(7):
        state_indices = [i for i, s in enumerate(states) if s[0] == day]
        day_prob = float(np.sum(pi[state_indices]))

        action_mass = {}
        expected_order = 0.0
        for i in state_indices:
            s = states[i]
            prob = float(pi[i])
            a = det_policy[s]
            expected_order += prob * a
            action_mass[a] = action_mass.get(a, 0.0) + prob

        if day_prob > 0:
            expected_order /= day_prob
            conditional_action_mass = {a: mass / day_prob for a, mass in action_mass.items()}
        else:
            conditional_action_mass = {a: 0.0 for a in set(det_policy.values())}

        most_likely_action = min(
            [a for a, mass in conditional_action_mass.items() if mass == max(conditional_action_mass.values())]
        )

        row = {
            "weekday": WEEKDAYS[day],
            "probability_of_this_weekday": day_prob,
            "expected_order_given_weekday": expected_order,
            "most_likely_order_given_weekday": most_likely_action,
            "action_distribution_given_weekday": str(
                {a: round(mass, 4) for a, mass in sorted(conditional_action_mass.items()) if mass > 1e-8}
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_visited_state_table(state_prob_df: pd.DataFrame, cutoff: float = 1e-6) -> pd.DataFrame:
    return state_prob_df[state_prob_df["stationary_probability"] > cutoff].copy()


# ============================================================
# OUTPUT HELPERS
# ============================================================

def build_compact_summary(policy_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        policy_df.groupby(["weekday", "total_stock"], sort=False)["optimal_order"]
        .agg(["min", "max", "mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_optimal_order", "count": "num_age_profiles"})
    )
    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    if SHELF_LIFE < 2:
        raise ValueError("SHELF_LIFE must be at least 2.")

    demand_pmf, K = load_demand_probabilities(DEMAND_XLSX_PATH, DEMAND_SHEET_NAME)
    states = enumerate_states(INVENTORY_CAP, SHELF_LIFE)
    actions_by_state = {
        s: feasible_actions(s, INVENTORY_CAP, MAX_ORDER, PRODUCTION_DAYS)
        for s in states
    }

    num_state_action_pairs = sum(len(v) for v in actions_by_state.values())

    print("=" * 72)
    print("STATIONARY AVERAGE-COST PLATELET MDP (DUAL LP)")
    print("=" * 72)
    print(f"Demand file:         {DEMAND_XLSX_PATH}")
    print(f"Demand sheet:        {DEMAND_SHEET_NAME}")
    print(f"Max demand K:        {K}")
    print(f"Shelf life:          {SHELF_LIFE}")
    print(f"Inventory cap:       {INVENTORY_CAP}")
    print(f"Max order:           {MAX_ORDER}")
    print(f"Production days:     {[WEEKDAYS[d] for d in sorted(PRODUCTION_DAYS)]}")
    print(f"Costs:               c0={C_OUTDATE}, cs={C_SHORTAGE}, cH={C_HOLDING}, cp={C_PRODUCTION}")
    print(f"Number of states:    {len(states)}")
    print(f"State-action pairs:  {num_state_action_pairs}")
    print("=" * 72)

    g_star, det_policy, policy_df = solve_average_cost_lp(
        states, actions_by_state, demand_pmf, SHELF_LIFE
    )

    # Evaluate the extracted policy so the recommendations become more interpretable
    P, c_vec, a_vec, idx = build_transition_matrix_under_policy(states, det_policy, demand_pmf, SHELF_LIFE)
    pi = stationary_distribution_of_policy(P)

    policy_df["stationary_probability"] = pi
    policy_df = policy_df.sort_values(
        ["weekday", "stationary_probability", "total_stock"],
        ascending=[True, False, True]
    )

    compact_df = build_compact_summary(policy_df)
    state_prob_df = build_state_probability_table(states, pi, det_policy)
    visited_state_df = build_visited_state_table(state_prob_df, cutoff=1e-6)
    weekday_plan_df = build_weekday_recommendation_table(states, pi, det_policy)

    # Save everything
    with pd.ExcelWriter(OUTPUT_XLSX_PATH, engine="openpyxl") as writer:
        policy_df.to_excel(writer, sheet_name="OptimalPolicy_AllStates", index=False)
        compact_df.to_excel(writer, sheet_name="CompactSummary", index=False)
        weekday_plan_df.to_excel(writer, sheet_name="WeekdayPlan", index=False)
        visited_state_df.to_excel(writer, sheet_name="VisitedStates", index=False)

    print("\nOPTIMAL LONG-RUN COST")
    print(f"Per day:   {g_star:.6f}")
    print(f"Per week:  {7.0 * g_star:.6f}")

    print("\nMOST READABLE VERSION OF THE POLICY")
    print("This is NOT the exact full policy, but the best weekday-level summary")
    print("under the stationary distribution of the optimal policy:")
    print(weekday_plan_df.to_string(index=False))

    print("\nIMPORTANT WARNING")
    print("The true optimal policy still depends on the FULL state (day, x1, ..., x_{m-1}).")
    print("So there is no single fixed order for each weekday that is always optimal.")
    print("But the table above tells you what the optimal policy TYPICALLY orders on each weekday.")

    print("\nTOP VISITED STATES UNDER THE OPTIMAL POLICY")
    print(visited_state_df.head(PRINT_TOP_ROWS).to_string(index=False))

    print("\nSaved detailed outputs to:")
    print(f"  {OUTPUT_XLSX_PATH}")


if __name__ == "__main__":
    main()
