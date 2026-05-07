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
C_OUTDATE = 2410.0
C_SHORTAGE = 20000.0
C_HOLDING = 5.0
C_PRODUCTION = 2410.0

# Output folders/files.
OUTPUT_DIR = Path("data/Non-stationary")
OUTPUT_XLSX_PATH = OUTPUT_DIR / "optimal_nonstationary_christmas_policy.xlsx"
PLOTS_DIR = OUTPUT_DIR / "plots_and_tables"

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
    date(YEAR, 12, 26): "Boxing Day",
    date(YEAR + 1, 1, 1): "New Year's Day",
}

# Demand during holidays.
# Options:
#   "usual"        -> use the weekday's ordinary demand distribution.
#   "saturday"     -> use Saturday demand distribution on holidays.
#   "sunday"       -> use Sunday demand distribution on holidays.
#   "weekend_mean" -> use the average of Saturday and Sunday distributions.
HOLIDAY_DEMAND_MODE = "weekend_mean"

# Initial distribution for the finite-horizon model.
# Options:
#   "stationary_start_day_distribution"
#       -> start from the stationary distribution of the regular weekly model,
#          conditional on the start weekday. This is the recommended setting.
#   "empty"
#       -> start from the empty state only. Useful for debugging, but usually not realistic.
INITIAL_STATE_MODE = "stationary_start_day_distribution"

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




def propagate_distribution_one_step_regular(
    current_dist: Dict[State, float],
    policy: Dict[State, int],
    demand_pmf: np.ndarray,
) -> Dict[State, float]:
    """Propagate a state distribution one ordinary stationary day forward."""
    next_dist: Dict[State, float] = {}

    for state, state_prob in current_dist.items():
        if state_prob <= 0.0:
            continue

        action = policy[state]
        demand_probs = demand_pmf[state[0], :]
        transition_dist, _ = transition_distribution_and_expected_cost_from_probs(
            state=state,
            action=action,
            demand_probs=demand_probs,
            shelf_life=SHELF_LIFE,
        )

        for next_state, trans_prob in transition_dist.items():
            next_dist[next_state] = next_dist.get(next_state, 0.0) + state_prob * trans_prob

    total = sum(next_dist.values())
    if total <= 0:
        raise ValueError("Distribution vanished during stationary propagation.")

    return {s: p / total for s, p in next_dist.items()}


def compute_stationary_start_day_distribution(
    states: List[State],
    demand_pmf: np.ndarray,
    policy: Dict[State, int],
    start_weekday: int,
    tol: float = 1e-13,
    max_weeks: int = 5000,
) -> Dict[State, float]:
    """
    Compute the regular stationary distribution conditional on the start weekday.

    The weekly model is periodic because the weekday advances deterministically.
    Instead of solving for a full stationary distribution over all weekdays, this
    function iterates the 7-day transition kernel on the states belonging to the
    chosen start weekday. The resulting distribution is the appropriate initial
    distribution for a finite-horizon scenario that begins during normal operation.
    """
    weekday_states = [s for s in states if s[0] == start_weekday]
    if not weekday_states:
        raise ValueError(f"No stationary states found for weekday index {start_weekday}.")

    current_dist: Dict[State, float] = {s: 1.0 / len(weekday_states) for s in weekday_states}

    for week in range(max_weeks):
        old_dist = current_dist
        new_dist = current_dist

        for _ in range(7):
            new_dist = propagate_distribution_one_step_regular(new_dist, policy, demand_pmf)

        keys = set(old_dist) | set(new_dist)
        diff = sum(abs(new_dist.get(s, 0.0) - old_dist.get(s, 0.0)) for s in keys)
        current_dist = new_dist

        if diff < tol:
            break
    else:
        print(
            f"Warning: stationary start-day distribution did not fully converge "
            f"after {max_weeks} weeks. Last L1 difference = {diff:.3e}"
        )

    # Remove numerical dust and renormalize.
    current_dist = {s: p for s, p in current_dist.items() if p > tol}
    total = sum(current_dist.values())
    if total <= 0:
        raise ValueError("Stationary start-day distribution has no positive support.")

    return {s: p / total for s, p in current_dist.items()}


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
    initial_distribution: Dict[State, float],
) -> tuple[Dict[tuple[int, State], float], pd.DataFrame]:
    """
    Compute time-dependent occupancy probabilities under the non-stationary policy.

    This is NOT a stationary distribution. It is the distribution
    P(X_t = s) at each stage t under the selected initial distribution and the
    finite-horizon policy.
    """
    occupancy: Dict[tuple[int, State], float] = {}

    reachable_start = set(reachable_by_t[0])
    current_dist = {
        s: p for s, p in initial_distribution.items()
        if s in reachable_start and p > 0.0
    }

    if not current_dist:
        raise ValueError("No positive-probability initial states survived the finite-horizon filters.")

    total_initial_mass = sum(current_dist.values())
    current_dist = {s: p / total_initial_mass for s, p in current_dist.items()}

    for s, p in current_dist.items():
        occupancy[(0, s)] = p

    for t in range(HORIZON_DAYS):
        next_dist: Dict[State, float] = {}
        demand_probs = stage_demand_pmfs[t]

        for s, state_prob in current_dist.items():
            if state_prob <= 0.0:
                continue

            action = policy[(t, s)]
            transition_dist, _ = transition_distribution_and_expected_cost_from_probs(
                state=s,
                action=action,
                demand_probs=demand_probs,
                shelf_life=SHELF_LIFE,
            )

            for ns, trans_prob in transition_dist.items():
                next_dist[ns] = next_dist.get(ns, 0.0) + state_prob * trans_prob

        total_mass = sum(next_dist.values())
        if total_mass <= 0:
            raise ValueError(f"Occupancy distribution vanished at stage {t + 1}.")

        # Renormalize to avoid tiny floating point drift.
        next_dist = {s: p / total_mass for s, p in next_dist.items()}

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
# COST BREAKDOWN UNDER NON-STATIONARY POLICY
# ============================================================

def compute_nonstationary_cost_breakdown(
    policy: Dict[tuple[int, State], int],
    occupancy: Dict[tuple[int, State], float],
    stage_demand_pmfs: List[np.ndarray],
) -> pd.DataFrame:
    """
    Decompose the expected immediate cost over the finite horizon.

    Unlike the stationary model, the non-stationary model has no stationary
    distribution. Therefore each stage is weighted by the time-dependent
    occupancy probability P(X_t = s) under the selected initial distribution
    and the computed non-stationary policy.

    The terminal continuation value is deliberately not included here, because
    it is a relative value from the stationary model and cannot be decomposed
    into holding, production, outdating and shortage costs for this finite
    horizon.
    """
    total_units = {
        "holding": 0.0,
        "production": 0.0,
        "outdate": 0.0,
        "shortage": 0.0,
    }

    for (t, state), state_prob in occupancy.items():
        if t >= HORIZON_DAYS or state_prob <= 0.0:
            continue

        action = policy[(t, state)]
        demand_probs = stage_demand_pmfs[t]

        # Production is deterministic conditional on the state and stage.
        total_units["production"] += state_prob * float(action)

        for demand, demand_prob in enumerate(demand_probs):
            if demand_prob <= DEMAND_SUPPORT_TOL:
                continue

            _, shortage, outdate, holding = step_dynamics(
                state=state,
                action=action,
                demand=demand,
                shelf_life=SHELF_LIFE,
            )

            probability_weight = state_prob * float(demand_prob)
            total_units["holding"] += probability_weight * holding
            total_units["outdate"] += probability_weight * outdate
            total_units["shortage"] += probability_weight * shortage

    unit_costs = {
        "holding": C_HOLDING,
        "production": C_PRODUCTION,
        "outdate": C_OUTDATE,
        "shortage": C_SHORTAGE,
    }

    total_costs = {
        component: total_units[component] * unit_costs[component]
        for component in total_units
    }
    grand_total_cost = sum(total_costs.values())

    rows = []
    for component in ["holding", "production", "outdate", "shortage"]:
        units = total_units[component]
        cost = total_costs[component]
        rows.append({
            "component": component,
            "unit_cost": unit_costs[component],
            "expected_total_units_over_horizon": units,
            "average_units_per_day": units / HORIZON_DAYS,
            "average_units_per_week": 7.0 * units / HORIZON_DAYS,
            "expected_total_cost_over_horizon": cost,
            "average_cost_per_day": cost / HORIZON_DAYS,
            "average_cost_per_week": 7.0 * cost / HORIZON_DAYS,
            "share_of_total_immediate_cost": (cost / grand_total_cost) if grand_total_cost > 0 else np.nan,
        })

    rows.append({
        "component": "total_immediate_cost",
        "unit_cost": np.nan,
        "expected_total_units_over_horizon": np.nan,
        "average_units_per_day": np.nan,
        "average_units_per_week": np.nan,
        "expected_total_cost_over_horizon": grand_total_cost,
        "average_cost_per_day": grand_total_cost / HORIZON_DAYS,
        "average_cost_per_week": 7.0 * grand_total_cost / HORIZON_DAYS,
        "share_of_total_immediate_cost": 1.0 if grand_total_cost > 0 else np.nan,
    })

    return pd.DataFrame(rows)


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
# PLOT AND POLICY-SUMMARY HELPERS
# ============================================================

def _format_stage_axis(ax, calendar_df: pd.DataFrame) -> None:
    x = calendar_df["t"].to_numpy()
    labels = [
        f"{int(row.t)}\n{str(row.weekday)[:3]}\n{str(row.date)[5:]}"
        for row in calendar_df.itertuples(index=False)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def _shade_nonproduction_days(ax, calendar_df: pd.DataFrame) -> None:
    for row in calendar_df.itertuples(index=False):
        if not bool(row.can_produce):
            ax.axvspan(row.t - 0.5, row.t + 0.5, alpha=0.12)
        if bool(row.is_holiday):
            ax.axvline(row.t, linestyle="--", linewidth=1.0, alpha=0.8)


def add_policy_plot_columns(policy_df: pd.DataFrame) -> pd.DataFrame:
    df = policy_df.copy()
    age_cols = [f"x{i + 1}" for i in range(SHELF_LIFE - 1)]
    df["total_stock"] = df[age_cols].sum(axis=1)
    df["post_order_stock"] = df["total_stock"] + df["optimal_order"]
    return df


def build_occupancy_weighted_stage_summary(policy_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    df = add_policy_plot_columns(policy_df)
    rows = []

    for t in range(HORIZON_DAYS):
        tmp = df[df["t"] == t].copy()
        row_cal = calendar_df.loc[t]
        mass = float(tmp["occupancy_probability"].sum())

        if mass > 0:
            weights = tmp["occupancy_probability"] / mass
            expected_order = float((weights * tmp["optimal_order"]).sum())
            expected_stock = float((weights * tmp["total_stock"]).sum())
            expected_post_order_stock = float((weights * tmp["post_order_stock"]).sum())

            order_mass = tmp.groupby("optimal_order")["occupancy_probability"].sum() / mass
            most_likely_order = int(order_mass.idxmax())
            most_likely_order_prob = float(order_mass.max())

            s1_mass = tmp.groupby("post_order_stock")["occupancy_probability"].sum() / mass
            dominant_s1 = int(s1_mass.idxmax())
            dominant_s1_prob = float(s1_mass.max())
        else:
            expected_order = np.nan
            expected_stock = np.nan
            expected_post_order_stock = np.nan
            most_likely_order = np.nan
            most_likely_order_prob = np.nan
            dominant_s1 = np.nan
            dominant_s1_prob = np.nan

        rows.append({
            "t": t,
            "date": row_cal["date"],
            "weekday": row_cal["weekday"],
            "is_holiday": bool(row_cal["is_holiday"]),
            "holiday_name": row_cal["holiday_name"],
            "can_produce": bool(row_cal["can_produce"]),
            "occupancy_mass": mass,
            "expected_start_stock": expected_stock,
            "expected_order": expected_order,
            "expected_post_order_stock": expected_post_order_stock,
            "most_likely_order": most_likely_order,
            "most_likely_order_probability": most_likely_order_prob,
            "dominant_post_order_stock_S1": dominant_s1,
            "dominant_S1_probability": dominant_s1_prob,
        })

    return pd.DataFrame(rows)


def build_initial_distribution_table(initial_distribution: Dict[State, float]) -> pd.DataFrame:
    rows = []
    for s, p in initial_distribution.items():
        row = {
            "weekday": WEEKDAYS[s[0]],
            "total_stock": sum(s[1:]),
            "initial_probability": p,
        }
        for i in range(SHELF_LIFE - 1):
            row[f"x{i + 1}"] = s[i + 1]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("initial_probability", ascending=False)


def _integerize_scaled_probabilities(probs: pd.Series, scale: int) -> pd.Series:
    """
    Convert probabilities to integer frequencies summing exactly to ``scale``.

    Important implementation detail:
    the index of ``probs`` may consist of tuple keys such as ``(S_1, x)``.
    Pandas interprets tuple keys as multi-axis indexers when using ``.loc`` on
    some Series, which can raise ``IndexingError: Too many indexers``.
    Therefore, the rounding adjustment is done positionally with NumPy arrays,
    and the original index is restored at the end.
    """
    values = probs.astype(float).to_numpy()
    raw = values * scale
    floors = np.floor(raw).astype(int)

    remainder = int(scale - floors.sum())
    fractional = raw - floors

    if remainder > 0:
        # Add one to the largest fractional remainders.
        order = np.argsort(-fractional)
        for pos in order[:remainder]:
            floors[pos] += 1
    elif remainder < 0:
        # Remove one from the smallest fractional remainders, but never below zero.
        order = np.argsort(fractional)
        removed = 0
        for pos in order:
            if floors[pos] > 0:
                floors[pos] -= 1
                removed += 1
                if removed == abs(remainder):
                    break

    return pd.Series(floors.astype(int), index=probs.index)


def build_frequency_table_for_stage(
    policy_df: pd.DataFrame,
    t: int,
    scale: int = 1_000_000,
    min_mass: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a book-style state-action frequency table for one stage.

    Columns are pre-order total stock x.
    Rows are post-order stock S_1 = x + a.
    Entries are occupancy probabilities scaled to `scale` observations.
    """
    df = add_policy_plot_columns(policy_df)
    tmp = df[(df["t"] == t) & (df["occupancy_probability"] > min_mass)].copy()

    if tmp.empty:
        raise ValueError(f"No positive occupancy probability at stage {t}.")

    total_mass = float(tmp["occupancy_probability"].sum())
    tmp["conditional_probability"] = tmp["occupancy_probability"] / total_mass

    grouped = (
        tmp.groupby(["post_order_stock", "total_stock"], as_index=False)["conditional_probability"]
        .sum()
    )

    keys = list(zip(grouped["post_order_stock"], grouped["total_stock"]))
    scaled = _integerize_scaled_probabilities(
        pd.Series(grouped["conditional_probability"].to_numpy(), index=keys),
        scale,
    )

    long_rows = []
    for (s1, x), freq in scaled.items():
        if freq <= 0:
            continue
        long_rows.append({
            "t": t,
            "total_stock": int(x),
            "post_order_stock_S1": int(s1),
            "frequency": int(freq),
            "probability": int(freq) / scale,
        })

    long_df = pd.DataFrame(long_rows)

    x_values = sorted(long_df["total_stock"].unique())
    s_values = sorted(long_df["post_order_stock_S1"].unique(), reverse=True)

    matrix = long_df.pivot(
        index="post_order_stock_S1",
        columns="total_stock",
        values="frequency",
    ).reindex(index=s_values, columns=x_values).fillna(0).astype(int)

    table_rows = []
    header = ["Stock x"] + x_values + ["Freq(S_1)"]
    table_rows.append(header)
    table_rows.append(["Up-to S_1"] + [""] * len(x_values) + [""])

    for s1 in s_values:
        row_vals = []
        for x in x_values:
            val = int(matrix.loc[s1, x])
            row_vals.append("" if val == 0 else val)
        table_rows.append([s1] + row_vals + [int(matrix.loc[s1].sum())])

    col_totals = matrix.sum(axis=0).astype(int)
    table_rows.append(["Freq(x)"] + [int(col_totals.loc[x]) for x in x_values] + [int(matrix.values.sum())])

    table_df = pd.DataFrame(table_rows)
    return table_df, long_df


def build_all_frequency_tables(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    scale: int = 1_000_000,
) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    table_by_sheet: Dict[str, pd.DataFrame] = {}
    long_tables = []
    summary_rows = []

    df = add_policy_plot_columns(policy_df)

    for row in calendar_df.itertuples(index=False):
        t = int(row.t)
        if not bool(row.can_produce):
            continue

        table_df, long_df = build_frequency_table_for_stage(df, t=t, scale=scale)
        long_df["date"] = row.date
        long_df["weekday"] = row.weekday
        long_df["holiday_name"] = row.holiday_name
        long_tables.append(long_df)

        s1_summary = (
            long_df.groupby("post_order_stock_S1")["frequency"]
            .sum()
            .sort_values(ascending=False)
        )
        dominant_s1 = int(s1_summary.index[0])
        dominant_freq = int(s1_summary.iloc[0])

        tmp = df[(df["t"] == t) & (df["occupancy_probability"] > 1e-12)].copy()
        mass = float(tmp["occupancy_probability"].sum())
        expected_order = float((tmp["occupancy_probability"] * tmp["optimal_order"]).sum() / mass)
        expected_stock = float((tmp["occupancy_probability"] * tmp["total_stock"]).sum() / mass)

        summary_rows.append({
            "t": t,
            "date": row.date,
            "weekday": row.weekday,
            "holiday_name": row.holiday_name,
            "dominant_post_order_stock_S1": dominant_s1,
            "dominant_S1_frequency": dominant_freq,
            "dominant_S1_share": dominant_freq / scale,
            "expected_start_stock": expected_stock,
            "expected_order": expected_order,
        })

        sheet_name = f"Freq_t{t:02d}_{str(row.weekday)[:3]}"
        table_by_sheet[sheet_name] = table_df

    frequency_long_df = pd.concat(long_tables, ignore_index=True) if long_tables else pd.DataFrame()
    dominant_summary_df = pd.DataFrame(summary_rows)
    return table_by_sheet, frequency_long_df, dominant_summary_df


def plot_frequency_table_image(table_df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Save a simple PNG rendering of one book-style frequency table."""
    ncols = table_df.shape[1]
    nrows = table_df.shape[0]
    fig_width = max(10, 0.55 * ncols)
    fig_height = max(2.5, 0.35 * nrows + 1.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)

    table = ax.table(
        cellText=table_df.astype(str).values,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_frequency_table_plots(
    frequency_tables: Dict[str, pd.DataFrame],
    calendar_df: pd.DataFrame,
    plots_dir: Path = PLOTS_DIR,
) -> Dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for sheet_name, table_df in frequency_tables.items():
        # Sheet name is Freq_tXX_Day.
        t = int(sheet_name.split("_")[1][1:])
        row = calendar_df.loc[t]
        title = f"(State, action)-frequency table for 1,000,000 stage-{t} {row['weekday']}s"
        path = plots_dir / f"frequency_table_t{t:02d}_{str(row['weekday']).lower()}.png"
        plot_frequency_table_image(table_df, title, path)
        paths[f"frequency_table_t{t:02d}"] = path

    return paths


def plot_occupancy_probability_heatmap(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
) -> None:
    df = add_policy_plot_columns(policy_df)
    heat = (
        df.groupby(["total_stock", "t"])["occupancy_probability"]
        .sum()
        .unstack("t")
        .reindex(columns=range(HORIZON_DAYS), fill_value=0.0)
        .sort_index(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(13, 8))
    image = ax.imshow(heat.to_numpy(), aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Occupancy probability")

    ax.set_title("Occupancy probability mass by stage and total stock")
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Total inventory at start of day")

    labels = [
        f"{int(row.t)}\n{str(row.weekday)[:3]}\n{str(row.date)[5:]}"
        for row in calendar_df.itertuples(index=False)
    ]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
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


def plot_weighted_order_heatmap_by_total_stock(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
) -> None:
    df = add_policy_plot_columns(policy_df)
    rows = []

    for (stock, t), tmp in df.groupby(["total_stock", "t"]):
        mass = float(tmp["occupancy_probability"].sum())
        if mass <= 0.0:
            continue
        avg_order = float((tmp["occupancy_probability"] * tmp["optimal_order"]).sum() / mass)
        rows.append({"total_stock": stock, "t": t, "weighted_average_order": avg_order})

    heat_df = pd.DataFrame(rows)
    heat = (
        heat_df.pivot(index="total_stock", columns="t", values="weighted_average_order")
        .reindex(columns=range(HORIZON_DAYS))
        .sort_index(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(13, 8))
    image = ax.imshow(heat.to_numpy(), aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Weighted average optimal order")

    ax.set_title("Occupancy-weighted average optimal order by stage and total stock")
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Total inventory at start of day")

    labels = [
        f"{int(row.t)}\n{str(row.weekday)[:3]}\n{str(row.date)[5:]}"
        for row in calendar_df.itertuples(index=False)
    ]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
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


def plot_unweighted_order_heatmap_by_total_stock(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
) -> None:
    df = add_policy_plot_columns(policy_df)
    heat = (
        df.groupby(["total_stock", "t"])["optimal_order"]
        .mean()
        .unstack("t")
        .reindex(columns=range(HORIZON_DAYS))
        .sort_index(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(13, 8))
    image = ax.imshow(heat.to_numpy(), aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Average optimal order")

    ax.set_title("Average optimal order by stage and total stock")
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Total inventory at start of day")

    labels = [
        f"{int(row.t)}\n{str(row.weekday)[:3]}\n{str(row.date)[5:]}"
        for row in calendar_df.itertuples(index=False)
    ]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
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


def plot_order_distribution_by_stage_weighted(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
) -> None:
    df = add_policy_plot_columns(policy_df)
    dist = (
        df.groupby(["t", "optimal_order"])["occupancy_probability"]
        .sum()
        .reset_index(name="mass")
    )

    totals = dist.groupby("t")["mass"].transform("sum")
    dist["conditional_mass"] = dist["mass"] / totals

    pivot = (
        dist.pivot(index="t", columns="optimal_order", values="conditional_mass")
        .fillna(0.0)
        .reindex(range(HORIZON_DAYS), fill_value=0.0)
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))

    for order in pivot.columns:
        values = pivot[order].to_numpy()
        ax.bar(x, values, bottom=bottom, label=f"Order {order}", width=0.85)
        bottom += values

    _shade_nonproduction_days(ax, calendar_df)

    labels = [
        f"{int(row.t)}\n{str(row.weekday)[:3]}\n{str(row.date)[5:]}"
        for row in calendar_df.itertuples(index=False)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Stage / date")
    ax.set_ylabel("Conditional probability")
    ax.set_ylim(0, 1)
    ax.set_title("Distribution of optimal orders conditional on stage")
    ax.legend(title="Order size", bbox_to_anchor=(1.01, 1.0), loc="upper left", ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_expected_stock_and_order_by_stage(
    stage_summary_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(13, 5))

    ax1.plot(
        stage_summary_df["t"],
        stage_summary_df["expected_start_stock"],
        marker="o",
        label="Expected start stock",
    )
    ax1.plot(
        stage_summary_df["t"],
        stage_summary_df["expected_post_order_stock"],
        marker="s",
        linestyle="--",
        label="Expected post-order stock",
    )
    ax1.set_ylabel("Expected stock")

    ax2 = ax1.twinx()
    ax2.plot(
        stage_summary_df["t"],
        stage_summary_df["expected_order"],
        marker="^",
        linestyle=":",
        label="Expected order",
    )
    ax2.set_ylabel("Expected order")

    _shade_nonproduction_days(ax1, calendar_df)
    _format_stage_axis(ax1, calendar_df)
    ax1.set_title("Expected stock and production over the finite horizon")
    ax1.set_xlabel("Stage / date")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_policy_plots(
    policy_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    stage_occupancy_summary_df: pd.DataFrame,
    plots_dir: Path = PLOTS_DIR,
) -> Dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "occupancy_probability_heatmap": plots_dir / "01_occupancy_probability_heatmap.png",
        "weighted_order_heatmap": plots_dir / "02_weighted_order_heatmap.png",
        "order_distribution_by_stage": plots_dir / "03_order_distribution_by_stage.png",
        "expected_stock_and_order_by_stage": plots_dir / "04_expected_stock_and_order_by_stage.png",
        "unweighted_order_heatmap": plots_dir / "05_unweighted_order_heatmap.png",
    }

    plot_occupancy_probability_heatmap(policy_df, calendar_df, paths["occupancy_probability_heatmap"])
    plot_weighted_order_heatmap_by_total_stock(policy_df, calendar_df, paths["weighted_order_heatmap"])
    plot_order_distribution_by_stage_weighted(policy_df, calendar_df, paths["order_distribution_by_stage"])
    plot_expected_stock_and_order_by_stage(stage_occupancy_summary_df, calendar_df, paths["expected_stock_and_order_by_stage"])
    plot_unweighted_order_heatmap_by_total_stock(policy_df, calendar_df, paths["unweighted_order_heatmap"])

    return paths


# ============================================================
# MAIN
# ============================================================

def main():
    if SHELF_LIFE < 2:
        raise ValueError("SHELF_LIFE must be at least 2.")

    OUTPUT_XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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

    # 3) Choose the initial distribution.
    start_weekday = int(calendar_df.loc[0, "weekday_index"])
    empty_start_state = (start_weekday,) + (0,) * (SHELF_LIFE - 1)

    if INITIAL_STATE_MODE == "empty":
        initial_distribution = {empty_start_state: 1.0}
    elif INITIAL_STATE_MODE == "stationary_start_day_distribution":
        initial_distribution = compute_stationary_start_day_distribution(
            states=stationary_states,
            demand_pmf=demand_pmf,
            policy=stationary_policy,
            start_weekday=start_weekday,
        )
    else:
        raise ValueError(f"Unknown INITIAL_STATE_MODE: {INITIAL_STATE_MODE}")

    initial_states = list(initial_distribution.keys())
    print(f"\nInitial states with positive initial probability: {len(initial_states)}")

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

    # 6) Compute occupancy probabilities under the non-stationary policy.
    occupancy, occupancy_df = compute_finite_horizon_occupancy_probabilities(
        reachable_by_t=reachable_by_t,
        policy=policy,
        calendar_df=calendar_df,
        stage_demand_pmfs=stage_demand_pmfs,
        initial_distribution=initial_distribution,
    )

    merge_cols = ["t"] + [f"x{i + 1}" for i in range(SHELF_LIFE - 1)]
    policy_df = policy_df.merge(
        occupancy_df[merge_cols + ["occupancy_probability"]],
        on=merge_cols,
        how="left",
    )
    policy_df["occupancy_probability"] = policy_df["occupancy_probability"].fillna(0.0)

    visited_policy_df = (
        policy_df[policy_df["occupancy_probability"] > 1e-12]
        .copy()
        .sort_values(["t", "occupancy_probability"], ascending=[True, False])
    )

    # 7) Build summary sheets.
    stage_summary_df = build_stage_summary(policy_df, reachable_by_t)
    stage_occupancy_summary_df = build_occupancy_weighted_stage_summary(policy_df, calendar_df)
    state_count_df = build_reachable_state_count_table(candidate_by_t, reachable_by_t)
    initial_distribution_df = build_initial_distribution_table(initial_distribution)
    terminal_state_df = build_terminal_state_table(reachable_by_t[HORIZON_DAYS], h_star)

    cost_breakdown_df = compute_nonstationary_cost_breakdown(
        policy=policy,
        occupancy=occupancy,
        stage_demand_pmfs=stage_demand_pmfs,
    )

    frequency_tables, frequency_long_df, dominant_order_up_to_df = build_all_frequency_tables(
        policy_df=policy_df,
        calendar_df=calendar_df,
        scale=1_000_000,
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
            "visited_policy_rows_positive_occupancy",
            "initial_states_with_positive_probability",
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
            len(visited_policy_df),
            len(initial_distribution),
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

    # 8) Write Excel output.
    with pd.ExcelWriter(OUTPUT_XLSX_PATH, engine="openpyxl") as writer:
        run_summary_df.to_excel(writer, sheet_name="RunSummary", index=False)
        cost_breakdown_df.to_excel(writer, sheet_name="CostBreakdown", index=False)
        calendar_df.to_excel(writer, sheet_name="Calendar", index=False)
        state_count_df.to_excel(writer, sheet_name="StateCounts", index=False)
        stage_summary_df.to_excel(writer, sheet_name="StageSummary_Unweighted", index=False)
        stage_occupancy_summary_df.to_excel(writer, sheet_name="StageSummary_Weighted", index=False)
        dominant_order_up_to_df.to_excel(writer, sheet_name="DominantOrderUpTo", index=False)
        frequency_long_df.to_excel(writer, sheet_name="FrequencyTables_Long", index=False)
        policy_df.to_excel(writer, sheet_name="NSPolicy_AllStages", index=False)
        visited_policy_df.to_excel(writer, sheet_name="VisitedStates_NS", index=False)
        occupancy_df.to_excel(writer, sheet_name="OccupancyProbabilities", index=False)
        initial_distribution_df.to_excel(writer, sheet_name="InitialDistribution", index=False)
        terminal_state_df.to_excel(writer, sheet_name="TerminalStates", index=False)

        for sheet_name, table_df in frequency_tables.items():
            table_df.to_excel(writer, sheet_name=sheet_name[:31], index=False, header=False)

    # 9) Generate only the plots that mirror the stationary analysis.
    plot_paths = generate_policy_plots(
        policy_df=policy_df,
        calendar_df=calendar_df,
        stage_occupancy_summary_df=stage_occupancy_summary_df,
        plots_dir=PLOTS_DIR,
    )
    plot_paths.update(generate_frequency_table_plots(frequency_tables, calendar_df, PLOTS_DIR))

    print("\nOCCUPANCY-WEIGHTED STAGE SUMMARY")
    print(stage_occupancy_summary_df.to_string(index=False))

    print("\nDOMINANT ORDER-UP-TO LEVELS")
    print(dominant_order_up_to_df.to_string(index=False))

    print("\nCOST BREAKDOWN")
    print(cost_breakdown_df.to_string(index=False))

    print("\nSaved detailed outputs to:")
    print(f"  {OUTPUT_XLSX_PATH}")

    print("\nSaved plots to:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
