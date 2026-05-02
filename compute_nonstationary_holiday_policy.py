from __future__ import annotations

from collections import deque
from datetime import date, timedelta
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
import matplotlib.pyplot as plt



# ============================================================
# USER PARAMETERS
# ============================================================

# Same demand file as in the stationary model.
DEMAND_XLSX_PATH = Path("weekday_demand_probabilities.xlsx")
DEMAND_SHEET_NAME = "DemandProbabilities"

# Model parameters.
SHELF_LIFE = 5
INVENTORY_CAP = 50
MAX_ORDER = 35
REGULAR_PRODUCTION_DAYS = {0, 1, 2, 3, 4}  # 0=Monday, ..., 6=Sunday

# Costs.
C_OUTDATE = 2500.0
C_SHORTAGE = 20000.0
C_HOLDING = 5.0
C_PRODUCTION = 2500.0

# Output files.
OUTPUT_XLSX_PATH = Path("data/optimal_nonstationary_christmas_policy.xlsx")
PLOTS_DIR = Path("data/nonstationary_policy_plots")

# Numerical tolerances.
REDUCED_COST_TOL = 1e-8
DEMAND_SUPPORT_TOL = 1e-12
PRINT_TOP_ROWS = 30

# Calendar scenario.
# 2028 is chosen only because Christmas Day falls on a Tuesday.
# Horizon: one week before Christmas week, Christmas week, and one week after.
YEAR = 2030
SCENARIO_START_DATE = date(YEAR, 12, 16)  # Monday
HORIZON_DAYS = 21

# Production stops on ordinary weekends plus the following holidays.
HOLIDAYS = {
    date(YEAR, 12, 24): "Christmas Eve",
    date(YEAR, 12, 25): "Christmas Day",
    #date(YEAR, 12, 26): "Boxing Day",
    date(YEAR + 1, 1, 1): "New Year's Day",
}

# Demand during holidays.
# Options:
#   "usual"        -> use the weekday's ordinary demand distribution.
#   "saturday"     -> use Saturday demand distribution on holidays.
#   "sunday"       -> use Sunday demand distribution on holidays.
#   "weekend_mean" -> use the average of Saturday and Sunday distributions.
HOLIDAY_DEMAND_MODE = "weekend_mean"

# Initial states for the finite-horizon model.
# Options:
#   "empty"                    -> start from the empty state only.
#   "all_stationary_start_day" -> start from all states with the correct start weekday
#                                  that are reachable under the regular stationary model.
INITIAL_STATE_MODE = "all_stationary_start_day"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

State = Tuple[int, ...]  # (weekday, x1, ..., x_{m-1})


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


def positive_demand_values_from_probs(probs: np.ndarray, tol: float = DEMAND_SUPPORT_TOL) -> List[int]:
    return [int(k) for k, p in enumerate(probs) if p > tol]


def positive_demand_values_by_day(demand_pmf: np.ndarray, tol: float = DEMAND_SUPPORT_TOL):
    return {
        day: [int(k) for k, p in enumerate(demand_pmf[day, :]) if p > tol]
        for day in range(demand_pmf.shape[0])
    }


def extract_min_demand_by_day(demand_pmf: np.ndarray, tol: float = DEMAND_SUPPORT_TOL) -> Dict[int, int]:
    min_demand_by_day: Dict[int, int] = {}
    for day in range(demand_pmf.shape[0]):
        positive_demands = np.where(demand_pmf[day, :] > tol)[0]
        if len(positive_demands) == 0:
            raise ValueError(f"No positive-demand support found for weekday index {day}.")
        min_demand_by_day[day] = int(positive_demands[0])
    return min_demand_by_day


# ============================================================
# CALENDAR AND NON-STATIONARY DEMAND
# ============================================================

def build_calendar() -> pd.DataFrame:
    rows = []
    for t in range(HORIZON_DAYS):
        current_date = SCENARIO_START_DATE + timedelta(days=t)
        weekday = current_date.weekday()
        holiday_name = HOLIDAYS.get(current_date, "")
        is_regular_production_day = weekday in REGULAR_PRODUCTION_DAYS
        can_produce = is_regular_production_day and current_date not in HOLIDAYS

        rows.append({
            "t": t,
            "date": current_date.isoformat(),
            "weekday_index": weekday,
            "weekday": WEEKDAYS[weekday],
            "holiday_name": holiday_name,
            "is_holiday": bool(holiday_name),
            "regular_production_day": is_regular_production_day,
            "can_produce": can_produce,
        })
    return pd.DataFrame(rows)


def holiday_demand_distribution(demand_pmf: np.ndarray) -> np.ndarray:
    if HOLIDAY_DEMAND_MODE == "usual":
        raise RuntimeError("Internal error: usual mode should not call holiday_demand_distribution.")
    if HOLIDAY_DEMAND_MODE == "saturday":
        return demand_pmf[5, :].copy()
    if HOLIDAY_DEMAND_MODE == "sunday":
        return demand_pmf[6, :].copy()
    if HOLIDAY_DEMAND_MODE == "weekend_mean":
        probs = 0.5 * demand_pmf[5, :] + 0.5 * demand_pmf[6, :]
        return probs / probs.sum()
    raise ValueError(f"Unknown HOLIDAY_DEMAND_MODE: {HOLIDAY_DEMAND_MODE}")


def build_stage_demand_pmfs(calendar_df: pd.DataFrame, demand_pmf: np.ndarray) -> List[np.ndarray]:
    stage_pmfs: List[np.ndarray] = []
    for _, row in calendar_df.iterrows():
        weekday = int(row["weekday_index"])
        if bool(row["is_holiday"]) and HOLIDAY_DEMAND_MODE != "usual":
            probs = holiday_demand_distribution(demand_pmf)
        else:
            probs = demand_pmf[weekday, :].copy()
        probs = probs / probs.sum()
        stage_pmfs.append(probs)
    return stage_pmfs


# ============================================================
# STATE AND ACTION SPACE
# ============================================================

def all_inventory_vectors(cap: int, dims: int) -> Iterable[Tuple[int, ...]]:
    for vec in product(range(cap + 1), repeat=dims):
        if sum(vec) <= cap:
            yield vec


def enumerate_states(inventory_cap: int, shelf_life: int) -> List[State]:
    states = []
    for day in range(7):
        for inv in all_inventory_vectors(inventory_cap, shelf_life - 1):
            states.append((day,) + tuple(inv))
    return states


def feasible_actions_regular(
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


def feasible_actions_at_time(
    state: State,
    t: int,
    calendar_df: pd.DataFrame,
    inventory_cap: int,
    max_order: int,
) -> List[int]:
    current_stock = sum(state[1:])
    can_produce = bool(calendar_df.loc[t, "can_produce"])

    if not can_produce:
        return [0]

    max_feasible = min(max_order, inventory_cap - current_stock)
    return list(range(max_feasible + 1))


def compute_max_total_inventory_by_day_regular(min_demand_by_day: Dict[int, int]) -> Dict[int, int]:
    max_total_by_day: Dict[int, int] = {}

    for target_day in range(7):
        days_to_subtract = []
        current = (target_day - 1) % 7

        while True:
            days_to_subtract.append(current)
            if current in REGULAR_PRODUCTION_DAYS:
                break
            current = (current - 1) % 7

        bound = INVENTORY_CAP - sum(min_demand_by_day[d] for d in days_to_subtract)
        max_total_by_day[target_day] = max(0, bound)

    return max_total_by_day


def structurally_feasible_state_regular(
    state: State,
    max_total_by_day: Dict[int, int],
) -> bool:
    day = state[0]
    inv = state[1:]
    total_stock = sum(inv)

    if any(x > MAX_ORDER for x in inv):
        return False

    if total_stock > max_total_by_day[day]:
        return False

    for idx, x in enumerate(inv):
        r = idx + 1
        days_back = SHELF_LIFE - r
        origin_day = (day - days_back) % 7
        if x > 0 and origin_day not in REGULAR_PRODUCTION_DAYS:
            return False

    return True


def production_possible_before_or_inside_horizon(t: int, calendar_df: pd.DataFrame) -> bool:
    """
    Return whether production was possible at absolute stage t.

    For t >= 0, use the explicit holiday calendar.
    For t < 0, assume ordinary stationary conditions before the scenario starts.
    This is exactly what we want when the finite-horizon model is started one week
    before Christmas from normal operations.
    """
    if t >= 0:
        if t >= len(calendar_df):
            # After the horizon we assume ordinary stationary operation.
            weekday = (SCENARIO_START_DATE + timedelta(days=t)).weekday()
            return weekday in REGULAR_PRODUCTION_DAYS
        return bool(calendar_df.loc[t, "can_produce"])

    weekday = (SCENARIO_START_DATE + timedelta(days=t)).weekday()
    return weekday in REGULAR_PRODUCTION_DAYS


def demand_min_before_or_inside_horizon(
    t: int,
    calendar_df: pd.DataFrame,
    stage_demand_pmfs: List[np.ndarray],
    min_demand_by_day: Dict[int, int],
) -> int:
    if 0 <= t < len(stage_demand_pmfs):
        positive = positive_demand_values_from_probs(stage_demand_pmfs[t])
        if not positive:
            raise ValueError(f"No positive demand support at stage {t}.")
        return int(positive[0])

    weekday = (SCENARIO_START_DATE + timedelta(days=t)).weekday()
    return min_demand_by_day[weekday]


def compute_stage_max_total_inventory(
    t: int,
    calendar_df: pd.DataFrame,
    stage_demand_pmfs: List[np.ndarray],
    min_demand_by_day: Dict[int, int],
) -> int:
    days_to_subtract = []
    current = t - 1

    while True:
        days_to_subtract.append(current)
        if production_possible_before_or_inside_horizon(current, calendar_df):
            break
        current -= 1

    min_demand_sum = sum(
        demand_min_before_or_inside_horizon(d, calendar_df, stage_demand_pmfs, min_demand_by_day)
        for d in days_to_subtract
    )
    return max(0, INVENTORY_CAP - min_demand_sum)


def structurally_feasible_state_at_time(
    state: State,
    t: int,
    calendar_df: pd.DataFrame,
    stage_demand_pmfs: List[np.ndarray],
    min_demand_by_day: Dict[int, int],
) -> bool:
    weekday = int(calendar_df.loc[t, "weekday_index"]) if t < len(calendar_df) else (SCENARIO_START_DATE + timedelta(days=t)).weekday()
    if state[0] != weekday:
        return False

    inv = state[1:]
    total_stock = sum(inv)

    if any(x > MAX_ORDER for x in inv):
        return False

    if total_stock > compute_stage_max_total_inventory(t, calendar_df, stage_demand_pmfs, min_demand_by_day):
        return False

    # x_r at stage t can only be positive if production was possible exactly
    # SHELF_LIFE-r days earlier.
    for idx, x in enumerate(inv):
        if x <= 0:
            continue
        r = idx + 1
        days_back = SHELF_LIFE - r
        origin_t = t - days_back
        if not production_possible_before_or_inside_horizon(origin_t, calendar_df):
            return False

    return True


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

    stock = inv + [action]  # fresh units enter with SHELF_LIFE days remaining
    remaining_demand = demand
    remaining_stock = stock[:]

    # FIFO: oldest first.
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


def transition_distribution_and_expected_cost_from_probs(
    state: State,
    action: int,
    demand_probs: np.ndarray,
    shelf_life: int,
) -> tuple[Dict[State, float], float]:
    dist: Dict[State, float] = {}
    expected_cost = 0.0

    for demand, p in enumerate(demand_probs):
        if p <= DEMAND_SUPPORT_TOL:
            continue

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


def transition_distribution_and_expected_cost_regular(
    state: State,
    action: int,
    demand_pmf: np.ndarray,
    shelf_life: int,
) -> tuple[Dict[State, float], float]:
    day = state[0]
    return transition_distribution_and_expected_cost_from_probs(state, action, demand_pmf[day, :], shelf_life)


# ============================================================
# STATIONARY MODEL: FILTERS AND DUAL LP FOR TERMINAL VALUES
# ============================================================

def reachable_state_filter_regular(
    candidate_states: List[State],
    demand_pmf: np.ndarray,
    inventory_cap: int,
    max_order: int,
    production_days: set[int],
    shelf_life: int,
    initial_states: List[State],
) -> List[State]:
    candidate_set = set(candidate_states)
    positive_demands = positive_demand_values_by_day(demand_pmf)

    reachable = set()
    queue = deque()

    for s in initial_states:
        if s in candidate_set:
            reachable.add(s)
            queue.append(s)

    while queue:
        s = queue.popleft()
        day = s[0]
        for a in feasible_actions_regular(s, inventory_cap, max_order, production_days):
            for demand in positive_demands[day]:
                ns, _, _, _ = step_dynamics(s, a, demand, shelf_life)
                if ns in candidate_set and ns not in reachable:
                    reachable.add(ns)
                    queue.append(ns)

    return sorted(reachable)


def assert_transition_closed_regular(
    states: List[State],
    demand_pmf: np.ndarray,
    inventory_cap: int,
    max_order: int,
    production_days: set[int],
    shelf_life: int,
) -> None:
    state_set = set(states)
    positive_demands = positive_demand_values_by_day(demand_pmf)

    for s in states:
        day = s[0]
        for a in feasible_actions_regular(s, inventory_cap, max_order, production_days):
            for demand in positive_demands[day]:
                ns, _, _, _ = step_dynamics(s, a, demand, shelf_life)
                if ns not in state_set:
                    raise ValueError(
                        "Stationary pruned state space is not transition-closed.\n"
                        f"State: {s}\nAction: {a}\nDemand: {demand}\nNext state: {ns}"
                    )


def build_stationary_state_space(demand_pmf: np.ndarray) -> List[State]:
    min_demand_by_day = extract_min_demand_by_day(demand_pmf)
    max_total_by_day = compute_max_total_inventory_by_day_regular(min_demand_by_day)

    states = enumerate_states(INVENTORY_CAP, SHELF_LIFE)
    print(f"Stationary states before filtering: {len(states)}")

    states = [s for s in states if structurally_feasible_state_regular(s, max_total_by_day)]
    print(f"Stationary states after structural filtering: {len(states)}")

    empty_monday = (0,) + (0,) * (SHELF_LIFE - 1)
    states = reachable_state_filter_regular(
        candidate_states=states,
        demand_pmf=demand_pmf,
        inventory_cap=INVENTORY_CAP,
        max_order=MAX_ORDER,
        production_days=REGULAR_PRODUCTION_DAYS,
        shelf_life=SHELF_LIFE,
        initial_states=[empty_monday],
    )
    print(f"Stationary states after reachability filtering: {len(states)}")

    assert_transition_closed_regular(
        states=states,
        demand_pmf=demand_pmf,
        inventory_cap=INVENTORY_CAP,
        max_order=MAX_ORDER,
        production_days=REGULAR_PRODUCTION_DAYS,
        shelf_life=SHELF_LIFE,
    )
    return states


def solve_stationary_average_cost_lp_for_terminal_values(
    states: List[State],
    demand_pmf: np.ndarray,
) -> tuple[float, Dict[State, float], Dict[State, int]]:
    actions_by_state = {
        s: feasible_actions_regular(s, INVENTORY_CAP, MAX_ORDER, REGULAR_PRODUCTION_DAYS)
        for s in states
    }
    state_index = {s: i for i, s in enumerate(states)}

    transitions: Dict[tuple[State, int], Dict[State, float]] = {}
    expected_cost: Dict[tuple[State, int], float] = {}

    for s in states:
        for a in actions_by_state[s]:
            dist, cost = transition_distribution_and_expected_cost_regular(s, a, demand_pmf, SHELF_LIFE)
            transitions[(s, a)] = dist
            expected_cost[(s, a)] = cost

    model = gp.Model("stationary_average_cost_dual_lp_for_terminal_values")
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
        raise RuntimeError(f"Gurobi did not find an optimal stationary solution. Status = {model.Status}")

    g_star = float(g.X)
    h_star = {s: float(h[s].X) for s in states}

    det_policy: Dict[State, int] = {}
    for s in states:
        scores = {}
        for a in actions_by_state[s]:
            score = expected_cost[(s, a)] + sum(p * h_star[ns] for ns, p in transitions[(s, a)].items())
            scores[a] = score
        min_score = min(scores.values())
        best_actions = [a for a, val in scores.items() if abs(val - min_score) <= REDUCED_COST_TOL]
        det_policy[s] = min(best_actions)

    return g_star, h_star, det_policy


# ============================================================
# NON-STATIONARY STATE FILTERS
# ============================================================

def build_stage_candidate_states(
    calendar_df: pd.DataFrame,
    stage_demand_pmfs: List[np.ndarray],
    min_demand_by_day: Dict[int, int],
) -> Dict[int, List[State]]:
    all_states = enumerate_states(INVENTORY_CAP, SHELF_LIFE)
    candidate_by_t: Dict[int, List[State]] = {}

    # Include terminal stage H as well. It is the state at the start of the day after the horizon.
    for t in range(HORIZON_DAYS + 1):
        candidate_by_t[t] = [
            s for s in all_states
            if structurally_feasible_state_at_time(s, t, calendar_df, stage_demand_pmfs, min_demand_by_day)
        ]

    return candidate_by_t


def finite_horizon_reachability_filter(
    candidate_by_t: Dict[int, List[State]],
    calendar_df: pd.DataFrame,
    stage_demand_pmfs: List[np.ndarray],
    initial_states: List[State],
) -> Dict[int, List[State]]:
    reachable_by_t: Dict[int, set[State]] = {t: set() for t in range(HORIZON_DAYS + 1)}
    candidate_sets = {t: set(states) for t, states in candidate_by_t.items()}

    for s in initial_states:
        if s in candidate_sets[0]:
            reachable_by_t[0].add(s)

    if not reachable_by_t[0]:
        raise ValueError("No initial states survived the finite-horizon structural filter.")

    for t in range(HORIZON_DAYS):
        positive_demands = positive_demand_values_from_probs(stage_demand_pmfs[t])
        for s in sorted(reachable_by_t[t]):
            for a in feasible_actions_at_time(s, t, calendar_df, INVENTORY_CAP, MAX_ORDER):
                for demand in positive_demands:
                    ns, _, _, _ = step_dynamics(s, a, demand, SHELF_LIFE)
                    if ns not in candidate_sets[t + 1]:
                        raise ValueError(
                            "Finite-horizon candidate state space is not transition-closed.\n"
                            f"Stage: {t}\nState: {s}\nAction: {a}\nDemand: {demand}\nNext state: {ns}\n"
                            "This usually means that one of the structural filters is too aggressive."
                        )
                    reachable_by_t[t + 1].add(ns)

    return {t: sorted(states) for t, states in reachable_by_t.items()}


# ============================================================
# SOLVE FINITE-HORIZON NON-STATIONARY MODEL
# ============================================================

def solve_finite_horizon_backward_induction(
    reachable_by_t: Dict[int, List[State]],
    calendar_df: pd.DataFrame,
    stage_demand_pmfs: List[np.ndarray],
    terminal_h: Dict[State, float],
) -> tuple[Dict[tuple[int, State], int], Dict[tuple[int, State], float], pd.DataFrame]:
    policy: Dict[tuple[int, State], int] = {}
    value: Dict[tuple[int, State], float] = {}
    rows = []

    # Terminal continuation value from the stationary average-cost problem.
    V_next: Dict[State, float] = {}
    missing_terminal_states = []
    for s in reachable_by_t[HORIZON_DAYS]:
        if s not in terminal_h:
            missing_terminal_states.append(s)
        else:
            V_next[s] = terminal_h[s]

    if missing_terminal_states:
        raise ValueError(
            "Some terminal states are not present in the stationary state space, so terminal relative values are missing.\n"
            f"First few missing states: {missing_terminal_states[:10]}"
        )

    for t in reversed(range(HORIZON_DAYS)):
        V_current: Dict[State, float] = {}
        row = calendar_df.loc[t]
        demand_probs = stage_demand_pmfs[t]

        for s in reachable_by_t[t]:
            scores: Dict[int, float] = {}
            expected_immediate_costs: Dict[int, float] = {}

            for a in feasible_actions_at_time(s, t, calendar_df, INVENTORY_CAP, MAX_ORDER):
                dist, expected_cost = transition_distribution_and_expected_cost_from_probs(
                    s, a, demand_probs, SHELF_LIFE
                )
                continuation = 0.0
                for ns, p in dist.items():
                    if ns not in V_next:
                        raise ValueError(
                            "Backward induction encountered a next state without a value.\n"
                            f"Stage: {t}\nState: {s}\nAction: {a}\nNext state: {ns}"
                        )
                    continuation += p * V_next[ns]

                scores[a] = expected_cost + continuation
                expected_immediate_costs[a] = expected_cost

            min_score = min(scores.values())
            best_actions = [a for a, val in scores.items() if abs(val - min_score) <= REDUCED_COST_TOL]
            chosen_action = min(best_actions)

            policy[(t, s)] = chosen_action
            value[(t, s)] = min_score
            V_current[s] = min_score

            output_row = {
                "t": t,
                "date": row["date"],
                "weekday": row["weekday"],
                "is_holiday": bool(row["is_holiday"]),
                "holiday_name": row["holiday_name"],
                "can_produce": bool(row["can_produce"]),
                "total_stock": sum(s[1:]),
                "optimal_order": chosen_action,
                "value": min_score,
                "expected_immediate_cost_at_chosen_action": expected_immediate_costs[chosen_action],
                "num_tied_best_actions": len(best_actions),
                "tied_best_actions": str(best_actions),
            }
            for i in range(len(s) - 1):
                output_row[f"x{i + 1}"] = s[i + 1]
            rows.append(output_row)

        V_next = V_current

    policy_df = pd.DataFrame(rows).sort_values(["t", "weekday", "total_stock", "optimal_order"])
    return policy, value, policy_df

def compute_finite_horizon_occupancy_probabilities(
    reachable_by_t: Dict[int, List[State]],
    policy: Dict[tuple[int, State], int],
    calendar_df: pd.DataFrame,
    stage_demand_pmfs: List[np.ndarray],
    initial_states: List[State],
) -> tuple[Dict[tuple[int, State], float], pd.DataFrame]:
    """
    Computes time-dependent state probabilities under the non-stationary policy.

    This is NOT a stationary distribution.
    It is the probability of being in state s at stage t under the chosen initial distribution.
    """

    occupancy: Dict[tuple[int, State], float] = {}

    # Simple default: uniform distribution over initial states.
    # Later we can improve this by using the stationary distribution conditional on start weekday.
    initial_states = [s for s in initial_states if s in set(reachable_by_t[0])]

    if not initial_states:
        raise ValueError("No valid initial states for occupancy computation.")

    initial_prob = 1.0 / len(initial_states)

    current_dist = {s: initial_prob for s in initial_states}

    for s, p in current_dist.items():
        occupancy[(0, s)] = p

    for t in range(HORIZON_DAYS):
        next_dist: Dict[State, float] = {}

        demand_probs = stage_demand_pmfs[t]

        for s, state_prob in current_dist.items():
            if state_prob <= 0:
                continue

            a = policy[(t, s)]

            transition_dist, _ = transition_distribution_and_expected_cost_from_probs(
                state=s,
                action=a,
                demand_probs=demand_probs,
                shelf_life=SHELF_LIFE,
            )

            for ns, trans_prob in transition_dist.items():
                next_dist[ns] = next_dist.get(ns, 0.0) + state_prob * trans_prob

        for ns, p in next_dist.items():
            occupancy[(t + 1, ns)] = p

        current_dist = next_dist

    rows = []
    for t in range(HORIZON_DAYS + 1):
        for s in reachable_by_t[t]:
            row = {
                "t": t,
                "date": (SCENARIO_START_DATE + timedelta(days=t)).isoformat(),
                "weekday": WEEKDAYS[s[0]],
                "occupancy_probability": occupancy.get((t, s), 0.0),
                "total_stock": sum(s[1:]),
            }
            for i in range(SHELF_LIFE - 1):
                row[f"x{i + 1}"] = s[i + 1]
            rows.append(row)

    occupancy_df = pd.DataFrame(rows)

    return occupancy, occupancy_df

# ============================================================
# OUTPUT HELPERS
# ============================================================

def build_stage_summary(policy_df: pd.DataFrame, reachable_by_t: Dict[int, List[State]]) -> pd.DataFrame:
    summary = (
        policy_df.groupby(["t", "date", "weekday", "is_holiday", "holiday_name", "can_produce"], dropna=False)["optimal_order"]
        .agg(["min", "max", "mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_optimal_order_over_reachable_states", "count": "num_reachable_states"})
    )

    terminal_row = pd.DataFrame([{
        "t": HORIZON_DAYS,
        "date": (SCENARIO_START_DATE + timedelta(days=HORIZON_DAYS)).isoformat(),
        "weekday": WEEKDAYS[(SCENARIO_START_DATE + timedelta(days=HORIZON_DAYS)).weekday()],
        "is_holiday": False,
        "holiday_name": "Terminal stationary continuation",
        "can_produce": np.nan,
        "min": np.nan,
        "max": np.nan,
        "avg_optimal_order_over_reachable_states": np.nan,
        "num_reachable_states": len(reachable_by_t[HORIZON_DAYS]),
    }])
    return pd.concat([summary, terminal_row], ignore_index=True)


def build_reachable_state_count_table(
    candidate_by_t: Dict[int, List[State]],
    reachable_by_t: Dict[int, List[State]],
) -> pd.DataFrame:
    rows = []
    for t in range(HORIZON_DAYS + 1):
        current_date = SCENARIO_START_DATE + timedelta(days=t)
        rows.append({
            "t": t,
            "date": current_date.isoformat(),
            "weekday": WEEKDAYS[current_date.weekday()],
            "candidate_states_after_structural_filter": len(candidate_by_t[t]),
            "states_after_reachability_filter": len(reachable_by_t[t]),
        })
    return pd.DataFrame(rows)


def build_initial_state_table(initial_states: List[State]) -> pd.DataFrame:
    rows = []
    for s in initial_states:
        row = {
            "weekday": WEEKDAYS[s[0]],
            "total_stock": sum(s[1:]),
        }
        for i in range(len(s) - 1):
            row[f"x{i + 1}"] = s[i + 1]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["total_stock"] + [f"x{i + 1}" for i in range(SHELF_LIFE - 1)])


def build_terminal_state_table(terminal_states: List[State], terminal_h: Dict[State, float]) -> pd.DataFrame:
    rows = []
    for s in terminal_states:
        row = {
            "weekday": WEEKDAYS[s[0]],
            "total_stock": sum(s[1:]),
            "terminal_relative_value": terminal_h.get(s, np.nan),
        }
        for i in range(len(s) - 1):
            row[f"x{i + 1}"] = s[i + 1]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["total_stock"] + [f"x{i + 1}" for i in range(SHELF_LIFE - 1)])


# ============================================================
# PLOT HELPERS
# ============================================================

def _format_stage_axis(ax, calendar_df: pd.DataFrame) -> None:
    """Common x-axis formatting for horizon plots."""
    x = calendar_df["t"].to_numpy()
    labels = [
        f"{int(row.t)}\n{str(row.weekday)[:3]}\n{str(row.date)[5:]}"
        for row in calendar_df.itertuples(index=False)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def _shade_nonproduction_days(ax, calendar_df: pd.DataFrame) -> None:
    """Lightly mark days with no production. Holidays get a vertical line as well."""
    for row in calendar_df.itertuples(index=False):
        if not bool(row.can_produce):
            ax.axvspan(row.t - 0.5, row.t + 0.5, alpha=0.12)
        if bool(row.is_holiday):
            ax.axvline(row.t, linestyle="--", linewidth=1.0, alpha=0.8)


def add_policy_plot_columns(policy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns that make plotting and grouping easier.
    This keeps the plotting functions independent of how the policy was exported.
    """
    df = policy_df.copy()
    age_cols = [f"x{i + 1}" for i in range(SHELF_LIFE - 1)]
    df["total_stock"] = df[age_cols].sum(axis=1)
    df["old_stock"] = df["x1"]
    df["young_stock"] = df[f"x{SHELF_LIFE - 1}"]
    df["pre_action_inventory"] = df["total_stock"] + df["optimal_order"]
    return df


def plot_average_order_by_stage(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot min/mean/max optimal order over reachable states at each stage."""
    df = add_policy_plot_columns(policy_df)
    grouped = (
        df.groupby("t")["optimal_order"]
        .agg(min_order="min", mean_order="mean", max_order="max")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(grouped["t"], grouped["mean_order"], marker="o", label="Mean optimal order")
    ax.plot(grouped["t"], grouped["min_order"], marker="v", linestyle="--", label="Minimum")
    ax.plot(grouped["t"], grouped["max_order"], marker="^", linestyle="--", label="Maximum")
    _shade_nonproduction_days(ax, calendar_df)
    _format_stage_axis(ax, calendar_df)
    ax.set_title("Non-stationary policy: optimal order by stage")
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Optimal order")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_reachable_state_counts(
    state_count_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot structural vs reachability filtered state counts for each stage."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        state_count_df["t"],
        state_count_df["candidate_states_after_structural_filter"],
        marker="o",
        label="After structural filter",
    )
    ax.plot(
        state_count_df["t"],
        state_count_df["states_after_reachability_filter"],
        marker="s",
        label="After reachability filter",
    )
    _shade_nonproduction_days(ax, calendar_df)
    _format_stage_axis(ax, calendar_df)
    ax.set_title("Finite-horizon state-space reduction")
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Number of states")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_order_heatmap_by_total_stock(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Heatmap of average optimal order by stage and total stock.
    This is usually the most useful first visualization of the policy structure.
    """
    df = add_policy_plot_columns(policy_df)
    heat = (
        df.groupby(["total_stock", "t"])["optimal_order"]
        .mean()
        .unstack("t")
        .sort_index(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(13, 8))
    image = ax.imshow(heat.to_numpy(), aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Average optimal order")

    ax.set_title("Average optimal order by stage and total inventory")
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Total inventory at start of day")

    x_labels = [
        f"{int(row.t)}\n{str(row.weekday)[:3]}\n{str(row.date)[5:]}"
        for row in calendar_df.itertuples(index=False)
    ]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels([str(i) for i in heat.index], fontsize=8)

    for row in calendar_df.itertuples(index=False):
        if not bool(row.can_produce):
            ax.axvline(row.t, linewidth=2.0, alpha=0.35)
        if bool(row.is_holiday):
            ax.axvline(row.t, linestyle="--", linewidth=1.0, alpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_policy_slices_by_inventory(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
    selected_total_stocks: List[int] | None = None,
) -> None:
    """
    Plot average optimal order over time for selected total inventory levels.
    Useful for seeing whether the policy ramps up before production breaks.
    """
    df = add_policy_plot_columns(policy_df)

    if selected_total_stocks is None:
        available = sorted(df["total_stock"].unique())
        if not available:
            raise ValueError("No policy rows available for plotting.")
        candidate_levels = [0, INVENTORY_CAP // 4, INVENTORY_CAP // 2, 3 * INVENTORY_CAP // 4]
        selected_total_stocks = []
        for level in candidate_levels:
            nearest = min(available, key=lambda x: abs(x - level))
            if nearest not in selected_total_stocks:
                selected_total_stocks.append(nearest)

    fig, ax = plt.subplots(figsize=(13, 5))
    for stock in selected_total_stocks:
        tmp = (
            df[df["total_stock"] == stock]
            .groupby("t")["optimal_order"]
            .mean()
            .reindex(range(HORIZON_DAYS))
        )
        ax.plot(tmp.index, tmp.to_numpy(), marker="o", label=f"Total stock = {stock}")

    _shade_nonproduction_days(ax, calendar_df)
    _format_stage_axis(ax, calendar_df)
    ax.set_title("Policy slices: average optimal order for selected stock levels")
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Average optimal order")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_order_distribution_around_holidays(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Boxplot of optimal orders by stage.
    Unlike the mean plot, this shows how heterogeneous the state-dependent policy is.
    """
    df = add_policy_plot_columns(policy_df)
    data = [df.loc[df["t"] == t, "optimal_order"].to_numpy() for t in range(HORIZON_DAYS)]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.boxplot(data, positions=list(range(HORIZON_DAYS)), showfliers=False)
    _shade_nonproduction_days(ax, calendar_df)
    _format_stage_axis(ax, calendar_df)
    ax.set_title("Distribution of optimal orders across reachable states")
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Optimal order")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def plot_order_distribution_by_stage_stacked(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Stacked barplot of the distribution of optimal orders for each stage.

    This is the non-stationary version of the weekday order distribution plot.
    It averages over reachable states, not over simulated state probabilities.
    """
    df = add_policy_plot_columns(policy_df)

    dist = (
        df.groupby(["t", "optimal_order"])
        .size()
        .reset_index(name="count")
    )

    totals = dist.groupby("t")["count"].transform("sum")
    dist["mass"] = dist["count"] / totals

    pivot = (
        dist.pivot(index="t", columns="optimal_order", values="mass")
        .fillna(0.0)
        .reindex(range(HORIZON_DAYS), fill_value=0.0)
    )

    fig, ax = plt.subplots(figsize=(16, 6))

    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))

    for order in pivot.columns:
        values = pivot[order].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=f"Order {order}",
            width=0.85,
        )
        bottom += values

    _shade_nonproduction_days(ax, calendar_df)

    labels = [
        f"{int(row.t)}\n{str(row.weekday)[:3]}\n{str(row.date)[5:]}"
        for row in calendar_df.itertuples(index=False)
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Reachable-state mass")
    ax.set_title("Distribution of optimal orders by stage")

    ax.legend(
        title="Order size",
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        ncol=2,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def generate_policy_plots(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    state_count_df: pd.DataFrame,
    plots_dir: Path = PLOTS_DIR,
) -> Dict[str, Path]:
    """Generate all standard plots and return their paths."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "average_order_by_stage": plots_dir / "01_average_order_by_stage.png",
        "reachable_state_counts": plots_dir / "02_reachable_state_counts.png",
        "order_heatmap_by_total_stock": plots_dir / "03_order_heatmap_by_total_stock.png",
        "policy_slices_by_inventory": plots_dir / "04_policy_slices_by_inventory.png",
        "order_distribution_by_stage": plots_dir / "05_order_distribution_by_stage.png",
        "order_distribution_by_stage_stacked": plots_dir / "12_order_distribution_by_stage_stacked.png",
    }

    plot_average_order_by_stage(policy_df, calendar_df, paths["average_order_by_stage"])
    plot_reachable_state_counts(state_count_df, calendar_df, paths["reachable_state_counts"])
    plot_order_heatmap_by_total_stock(policy_df, calendar_df, paths["order_heatmap_by_total_stock"])
    plot_policy_slices_by_inventory(policy_df, calendar_df, paths["policy_slices_by_inventory"])
    plot_order_distribution_around_holidays(policy_df, calendar_df, paths["order_distribution_by_stage"])
    plot_order_distribution_by_stage_stacked(policy_df, calendar_df, paths["order_distribution_by_stage_stacked"],)

    return paths

def compute_shortage_outdate_diagnostics(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    stage_demand_pmfs: List[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes shortage/outdate risk for the chosen optimal action in each reachable state.

    Important:
    These are averages over reachable states, not simulated realized frequencies.
    """
    rows = []

    for row in policy_df.itertuples(index=False):
        t = int(row.t)
        state = (
            int(getattr(row, "t")) * 0 + calendar_df.loc[t, "weekday_index"],
            *[int(getattr(row, f"x{i + 1}")) for i in range(SHELF_LIFE - 1)]
        )
        action = int(row.optimal_order)
        demand_probs = stage_demand_pmfs[t]

        prob_shortage = 0.0
        prob_outdate = 0.0
        expected_shortage_units = 0.0
        expected_outdate_units = 0.0

        for demand, p in enumerate(demand_probs):
            if p <= DEMAND_SUPPORT_TOL:
                continue

            _, shortage, outdate, _ = step_dynamics(
                state=state,
                action=action,
                demand=demand,
                shelf_life=SHELF_LIFE,
            )

            prob_shortage += float(p) * float(shortage > 0)
            prob_outdate += float(p) * float(outdate > 0)
            expected_shortage_units += float(p) * shortage
            expected_outdate_units += float(p) * outdate

        output_row = {
            "t": t,
            "date": row.date,
            "weekday": row.weekday,
            "is_holiday": bool(row.is_holiday),
            "holiday_name": row.holiday_name,
            "can_produce": bool(row.can_produce),
            "total_stock": int(row.total_stock),
            "optimal_order": action,
            "prob_shortage": prob_shortage,
            "prob_outdate": prob_outdate,
            "expected_shortage_units": expected_shortage_units,
            "expected_outdate_units": expected_outdate_units,
        }

        for i in range(SHELF_LIFE - 1):
            output_row[f"x{i + 1}"] = int(getattr(row, f"x{i + 1}"))

        rows.append(output_row)

    risk_by_state_df = pd.DataFrame(rows)

    risk_summary_df = (
        risk_by_state_df
        .groupby(["t", "date", "weekday", "is_holiday", "holiday_name", "can_produce"], dropna=False)
        .agg(
            avg_prob_shortage=("prob_shortage", "mean"),
            max_prob_shortage=("prob_shortage", "max"),
            avg_prob_outdate=("prob_outdate", "mean"),
            max_prob_outdate=("prob_outdate", "max"),
            avg_expected_shortage_units=("expected_shortage_units", "mean"),
            max_expected_shortage_units=("expected_shortage_units", "max"),
            avg_expected_outdate_units=("expected_outdate_units", "mean"),
            max_expected_outdate_units=("expected_outdate_units", "max"),
            num_reachable_states=("prob_shortage", "count"),
        )
        .reset_index()
    )

    return risk_summary_df, risk_by_state_df


def plot_expected_shortage_outdate_by_stage(
    risk_summary_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        risk_summary_df["t"],
        risk_summary_df["avg_expected_shortage_units"],
        marker="o",
        label="Expected shortage units",
    )
    ax.plot(
        risk_summary_df["t"],
        risk_summary_df["avg_expected_outdate_units"],
        marker="s",
        label="Expected outdated units",
    )

    _shade_nonproduction_days(ax, calendar_df)
    _format_stage_axis(ax, calendar_df)

    ax.set_title("Expected shortage and outdating by stage")
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Expected units")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_probability_shortage_outdate_by_stage(
    risk_summary_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        risk_summary_df["t"],
        risk_summary_df["avg_prob_shortage"],
        marker="o",
        label="Probability of shortage",
    )
    ax.plot(
        risk_summary_df["t"],
        risk_summary_df["avg_prob_outdate"],
        marker="s",
        label="Probability of outdating",
    )

    _shade_nonproduction_days(ax, calendar_df)
    _format_stage_axis(ax, calendar_df)

    ax.set_title("Probability of shortage and outdating by stage")
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Probability")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_risk_heatmap_by_total_stock(
    risk_by_state_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    output_path: Path,
) -> None:
    heat = (
        risk_by_state_df
        .groupby(["total_stock", "t"])[value_col]
        .mean()
        .unstack("t")
        .sort_index(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(13, 8))

    image = ax.imshow(heat.to_numpy(), aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Total inventory at start of day")

    x_labels = [
        f"{int(row.t)}\n{str(row.weekday)[:3]}\n{str(row.date)[5:]}"
        for row in calendar_df.itertuples(index=False)
    ]

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8)

    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels([str(i) for i in heat.index], fontsize=8)

    for row in calendar_df.itertuples(index=False):
        if not bool(row.can_produce):
            ax.axvline(row.t, linewidth=2.0, alpha=0.35)
        if bool(row.is_holiday):
            ax.axvline(row.t, linestyle="--", linewidth=1.0, alpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_risk_plots(
    risk_summary_df: pd.DataFrame,
    risk_by_state_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    plots_dir: Path = PLOTS_DIR,
) -> Dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "expected_shortage_outdate_by_stage": plots_dir / "06_expected_shortage_outdate_by_stage.png",
        "probability_shortage_outdate_by_stage": plots_dir / "07_probability_shortage_outdate_by_stage.png",
        "shortage_probability_heatmap": plots_dir / "08_shortage_probability_heatmap.png",
        "outdate_probability_heatmap": plots_dir / "09_outdate_probability_heatmap.png",
        "expected_shortage_units_heatmap": plots_dir / "10_expected_shortage_units_heatmap.png",
        "expected_outdate_units_heatmap": plots_dir / "11_expected_outdate_units_heatmap.png",
    }

    plot_expected_shortage_outdate_by_stage(
        risk_summary_df,
        calendar_df,
        paths["expected_shortage_outdate_by_stage"],
    )

    plot_probability_shortage_outdate_by_stage(
        risk_summary_df,
        calendar_df,
        paths["probability_shortage_outdate_by_stage"],
    )

    plot_risk_heatmap_by_total_stock(
        risk_by_state_df,
        calendar_df,
        value_col="prob_shortage",
        title="Probability of shortage by stage and total inventory",
        cbar_label="Probability of shortage",
        output_path=paths["shortage_probability_heatmap"],
    )

    plot_risk_heatmap_by_total_stock(
        risk_by_state_df,
        calendar_df,
        value_col="prob_outdate",
        title="Probability of outdating by stage and total inventory",
        cbar_label="Probability of outdating",
        output_path=paths["outdate_probability_heatmap"],
    )

    plot_risk_heatmap_by_total_stock(
        risk_by_state_df,
        calendar_df,
        value_col="expected_shortage_units",
        title="Expected shortage units by stage and total inventory",
        cbar_label="Expected shortage units",
        output_path=paths["expected_shortage_units_heatmap"],
    )

    plot_risk_heatmap_by_total_stock(
        risk_by_state_df,
        calendar_df,
        value_col="expected_outdate_units",
        title="Expected outdated units by stage and total inventory",
        cbar_label="Expected outdated units",
        output_path=paths["expected_outdate_units_heatmap"],
    )

    return paths



# ============================================================
# MAIN
# ============================================================

def main():
    if SHELF_LIFE < 2:
        raise ValueError("SHELF_LIFE must be at least 2.")

    OUTPUT_XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)

    demand_pmf, K = load_demand_probabilities(DEMAND_XLSX_PATH, DEMAND_SHEET_NAME)
    calendar_df = build_calendar()
    stage_demand_pmfs = build_stage_demand_pmfs(calendar_df, demand_pmf)
    min_demand_by_day = extract_min_demand_by_day(demand_pmf)

    print("=" * 80)
    print("NON-STATIONARY FINITE-HORIZON PLATELET MDP")
    print("=" * 80)
    print(f"Demand file:          {DEMAND_XLSX_PATH}")
    print(f"Demand sheet:         {DEMAND_SHEET_NAME}")
    print(f"Max demand K:         {K}")
    print(f"Shelf life:           {SHELF_LIFE}")
    print(f"Inventory cap:        {INVENTORY_CAP}")
    print(f"Max order:            {MAX_ORDER}")
    print(f"Holiday demand mode:  {HOLIDAY_DEMAND_MODE}")
    print(f"Initial state mode:   {INITIAL_STATE_MODE}")
    print("=" * 80)

    print("\nCalendar scenario:")
    print(calendar_df[["t", "date", "weekday", "holiday_name", "can_produce"]].to_string(index=False))

    # 1) Solve the regular stationary problem first. The relative values h are used
    #    as terminal continuation values in the finite-horizon holiday problem.
    stationary_states = build_stationary_state_space(demand_pmf)
    g_star, h_star, stationary_policy = solve_stationary_average_cost_lp_for_terminal_values(
        stationary_states, demand_pmf
    )
    print(f"\nStationary terminal continuation solved. g* per day = {g_star:.6f}")

    # 2) Construct stage-wise structural candidate states.
    candidate_by_t = build_stage_candidate_states(calendar_df, stage_demand_pmfs, min_demand_by_day)
    print("\nFinite-horizon candidate states after structural filtering:")
    for t in range(HORIZON_DAYS + 1):
        print(f"  t={t:2d}: {len(candidate_by_t[t])}")

    # 3) Choose realistic initial states. This is deliberately not just the empty state
    #    unless INITIAL_STATE_MODE is set to "empty".
    start_weekday = int(calendar_df.loc[0, "weekday_index"])
    empty_start_state = (start_weekday,) + (0,) * (SHELF_LIFE - 1)

    if INITIAL_STATE_MODE == "empty":
        initial_states = [empty_start_state]
    elif INITIAL_STATE_MODE == "all_stationary_start_day":
        initial_states = [s for s in stationary_states if s[0] == start_weekday]
    else:
        raise ValueError(f"Unknown INITIAL_STATE_MODE: {INITIAL_STATE_MODE}")

    print(f"\nInitial states before finite-horizon structural filtering: {len(initial_states)}")

    # 4) Apply the finite-horizon reachability filter.
    reachable_by_t = finite_horizon_reachability_filter(
        candidate_by_t=candidate_by_t,
        calendar_df=calendar_df,
        stage_demand_pmfs=stage_demand_pmfs,
        initial_states=initial_states,
    )

    print("\nFinite-horizon states after reachability filtering:")
    for t in range(HORIZON_DAYS + 1):
        print(f"  t={t:2d}: {len(reachable_by_t[t])}")

    # 5) Solve finite-horizon problem by backward induction.
    policy, value, policy_df = solve_finite_horizon_backward_induction(
        reachable_by_t=reachable_by_t,
        calendar_df=calendar_df,
        stage_demand_pmfs=stage_demand_pmfs,
        terminal_h=h_star,
    )

    occupancy, occupancy_df = compute_finite_horizon_occupancy_probabilities(
        reachable_by_t=reachable_by_t,
        policy=policy,
        calendar_df=calendar_df,
        stage_demand_pmfs=stage_demand_pmfs,
        initial_states=initial_states,
    )
    merge_cols = ["t"] + [f"x{i + 1}" for i in range(SHELF_LIFE - 1)]

    policy_df = policy_df.merge(
        occupancy_df[merge_cols + ["occupancy_probability"]],
        on=merge_cols,
        how="left",
    )

    visited_policy_df = (
        policy_df[policy_df["occupancy_probability"] > 1e-12]
        .copy()
        .sort_values(["t", "occupancy_probability"], ascending=[True, False])
    )

    policy_df["occupancy_probability"] = policy_df["occupancy_probability"].fillna(0.0)

    stage_summary_df = build_stage_summary(policy_df, reachable_by_t)
    state_count_df = build_reachable_state_count_table(candidate_by_t, reachable_by_t)
    initial_state_df = build_initial_state_table([s for s in initial_states if s in set(candidate_by_t[0])])
    terminal_state_df = build_terminal_state_table(reachable_by_t[HORIZON_DAYS], h_star)
    risk_summary_df, risk_by_state_df = compute_shortage_outdate_diagnostics(
        policy_df=policy_df,
        calendar_df=calendar_df,
        stage_demand_pmfs=stage_demand_pmfs,
    )

    run_summary_df = pd.DataFrame({
        "metric": [
            "stationary_average_cost_per_day_terminal_model",
            "stationary_average_cost_per_week_terminal_model",
            "horizon_days",
            "scenario_start_date",
            "scenario_end_date_inclusive",
            "terminal_date",
            "holiday_demand_mode",
            "initial_state_mode",
            "stationary_states_after_filtering",
            "finite_horizon_policy_rows",
            "initial_states_used",
            "terminal_states_reached",
            "shelf_life",
            "inventory_cap",
            "max_order",
            "c_outdate",
            "c_shortage",
            "c_holding",
            "c_production",
        ],
        "value": [
            g_star,
            7.0 * g_star,
            HORIZON_DAYS,
            SCENARIO_START_DATE.isoformat(),
            (SCENARIO_START_DATE + timedelta(days=HORIZON_DAYS - 1)).isoformat(),
            (SCENARIO_START_DATE + timedelta(days=HORIZON_DAYS)).isoformat(),
            HOLIDAY_DEMAND_MODE,
            INITIAL_STATE_MODE,
            len(stationary_states),
            len(policy_df),
            len(reachable_by_t[0]),
            len(reachable_by_t[HORIZON_DAYS]),
            SHELF_LIFE,
            INVENTORY_CAP,
            MAX_ORDER,
            C_OUTDATE,
            C_SHORTAGE,
            C_HOLDING,
            C_PRODUCTION,
        ],
    })

    with pd.ExcelWriter(OUTPUT_XLSX_PATH, engine="openpyxl") as writer:
        run_summary_df.to_excel(writer, sheet_name="RunSummary", index=False)
        calendar_df.to_excel(writer, sheet_name="Calendar", index=False)
        state_count_df.to_excel(writer, sheet_name="StateCounts", index=False)
        stage_summary_df.to_excel(writer, sheet_name="StageSummary", index=False)
        policy_df.to_excel(writer, sheet_name="NSPolicy_AllStages", index=False)
        initial_state_df.to_excel(writer, sheet_name="InitialStates", index=False)
        terminal_state_df.to_excel(writer, sheet_name="TerminalStates", index=False)
        risk_summary_df.to_excel(writer, sheet_name="RiskSummary", index=False)
        risk_by_state_df.to_excel(writer, sheet_name="RiskByState", index=False)
        occupancy_df.to_excel(writer, sheet_name="OccupancyProbabilities", index=False)
        visited_policy_df.to_excel(writer, sheet_name="VisitedStates_NS", index=False)

    plot_paths = generate_policy_plots(
        policy_df=visited_policy_df,
        calendar_df=calendar_df,
        state_count_df=state_count_df,
        plots_dir=PLOTS_DIR,
    )
    risk_plot_paths = generate_risk_plots(
        risk_summary_df=risk_summary_df,
        risk_by_state_df=risk_by_state_df,
        calendar_df=calendar_df,
        plots_dir=PLOTS_DIR,
    )

    plot_paths.update(risk_plot_paths)

    print("\nMOST READABLE VERSION OF THE NON-STATIONARY POLICY")
    print("This is not the exact full policy. It averages over reachable states at each stage.")
    print(stage_summary_df.head(HORIZON_DAYS).to_string(index=False))

    print("\nTOP POLICY ROWS")
    print(policy_df.head(PRINT_TOP_ROWS).to_string(index=False))

    print("\nSaved detailed outputs to:")
    print(f"  {OUTPUT_XLSX_PATH}")

    print("\nSaved plots to:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
