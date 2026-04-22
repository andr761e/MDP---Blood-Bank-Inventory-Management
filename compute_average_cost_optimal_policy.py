from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
import matplotlib.pyplot as plt


# ============================================================
# USER PARAMETERS
# ============================================================

# The path to the Excel file containing demand probabilities. Create this file first using the demand-matrix generator script.   
DEMAND_XLSX_PATH = Path("weekday_demand_probabilities.xlsx")
DEMAND_SHEET_NAME = "DemandProbabilities"

# Model parameters
SHELF_LIFE = 5                # platelet shelf life from purchase
INVENTORY_CAP = 15              # Max total inventory allowed
MAX_ORDER = 9                  # Max order quantity allowed
PRODUCTION_DAYS = {0, 1, 2, 3, 4}   # Production days (0=Monday, 1=Tuesday, ..., 6=Sunday)

# Costs
C_OUTDATE = 2500.0                # c_0
C_SHORTAGE = 20000.0              # c_s
C_HOLDING = 5.0                   # c_H
C_PRODUCTION = 2500.0             # Optional cost for production (keep at 0 for now)

# Output file name
OUTPUT_XLSX_PATH = Path("data/optimal_stationary_policy_lp_v2.xlsx")

PLOTS_DIR = Path("data/policy_plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Numerical tolerances
REDUCED_COST_TOL = 1e-8
PRINT_TOP_ROWS = 30


# WEEKSDAYS
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ============================================================
# INPUT: DEMAND PROBABILITIES
# ============================================================

# Opens excel file, finds demand columns, checks for missing weekdays, normalizes probabilities, and returns them as a numpy array. 
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

State = Tuple[int, ...] # Inventory vector (x_0, x_1, ..., x_{M-1}) 

#Enumarate all inventory vectors with total inventory <= cap (INVENTORY_CAP). This is used to build the state space of the MDP, which consists of (day, inventory vector) pairs. The inventory vector tracks how many units have 1 day left, 2 days left, ..., up to shelf_life-1 days left. The total inventory is the sum of these components and must be <= INVENTORY_CAP.
def all_inventory_vectors(cap: int, dims: int):
    for vec in product(range(cap + 1), repeat=dims):
        if sum(vec) <= cap:
            yield vec


# Enumerate all states (day, inventory vector) with total inventory <= cap (INVENTORY_CAP). The day component cycles through 0 to 6 (Monday to Sunday), and the inventory vector tracks how many units have 1 day left, 2 days left, ..., up to shelf_life-1 days left. The total inventory is the sum of these components and must be <= INVENTORY_CAP.
def enumerate_states(inventory_cap: int, shelf_life: int) -> List[State]:
    states = []
    for day in range(7):
        for inv in all_inventory_vectors(inventory_cap, shelf_life - 1):
            states.append((day,) + tuple(inv))
    return states

# Given a state, return the list of feasible actions (order quantities) that respect the inventory cap and production constraints. The state includes the current day and inventory vector, which allows us to determine if production is possible on that day and how much we can order without exceeding the inventory cap. If it's not a production day, the only feasible action is to order 0.
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


def extract_min_demand_by_day(demand_pmf: np.ndarray, tol: float = 1e-12) -> Dict[int, int]:
    """
    For each weekday d, return the smallest demand k with positive probability.
    """
    min_demand_by_day: Dict[int, int] = {}

    for day in range(demand_pmf.shape[0]):
        positive_demands = np.where(demand_pmf[day, :] > tol)[0]
        if len(positive_demands) == 0:
            raise ValueError(f"No positive-demand support found for weekday index {day}.")
        min_demand_by_day[day] = int(positive_demands[0])

    return min_demand_by_day

def compute_max_total_inventory_by_day(min_demand_by_day: Dict[int, int]) -> Dict[int, int]:
    """
    Compute a safe upper bound on total inventory at the start of each weekday.

    Logic:
    Work backwards from the target day until the most recent production day.
    Starting from INVENTORY_CAP on that production day (after ordering),
    subtract the minimum demand for each intervening day up to the day before
    the target day.
    """
    max_total_by_day: Dict[int, int] = {}

    for target_day in range(7):
        days_to_subtract = []

        current = (target_day - 1) % 7
        while True:
            days_to_subtract.append(current)
            if current in PRODUCTION_DAYS:
                break
            current = (current - 1) % 7

        bound = INVENTORY_CAP - sum(min_demand_by_day[d] for d in days_to_subtract)
        max_total_by_day[target_day] = max(0, bound)

    return max_total_by_day



# Check if a state is structurally feasible given the production constraints. For example, on Monday (day=0), we cannot have any inventory with 3 or 4 days of shelf life remaining, because that would imply production on Saturday or Sunday, which is not allowed. Similarly, on Tuesday (day=1), we cannot have inventory with 3 days of shelf life remaining, and on Sunday (day=6), we cannot have inventory with 4 days of shelf life remaining. This function can be used to filter out states that are impossible to reach under the given production schedule.
def structurally_feasible_state(
    state: State,
    max_total_by_day: Dict[int, int],
) -> bool:
    day = state[0]
    x1, x2, x3, x4 = state[1:]
    total_stock = x1 + x2 + x3 + x4

    # No single age bucket can exceed one day's production
    if any(x > MAX_ORDER for x in (x1, x2, x3, x4)):
        return False

    # Dynamic upper bound on total stock from minimum-demand support
    if total_stock > max_total_by_day[day]:
        return False

    # Weekday-specific impossibilities caused by no production on weekends
    if day == 0:  # Monday
        if x3 > 0 or x4 > 0:
            return False

    elif day == 1:  # Tuesday
        if x2 > 0 or x3 > 0:
            return False

    elif day == 2:  # Wednesday
        if x1 > 0 or x2 > 0:
            return False

    elif day == 3:  # Thursday
        if x1 > 0:
            return False

    elif day == 4:  # Friday
        pass

    elif day == 5:  # Saturday
        pass

    elif day == 6:  # Sunday
        if x4 > 0:
            return False

    else:
        return False

    return True

# ============================================================
# DYNAMICS: ACTION -> DEMAND -> FIFO -> AGEING
# ============================================================

# State observed at the start of the day, action taken, new units in, demand realized, then FIFO and ageing happen to get next state and costs. The state is a tuple where the first element is the day of the week (0=Monday, ..., 6=Sunday) and the remaining elements represent the inventory vector (x_1, x_2, ..., x_{m-1}), where x_i is the number of units with i days of shelf life remaining. The action is the order quantity placed at the start of the day. The demand is a random variable that will be realized after the action. The function computes the next state after applying FIFO (first-in-first-out) to meet demand and then ageing (decreasing shelf life by 1 day). It also calculates the shortage, outdate, and holding costs incurred during this transition.
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

# Given a state and action, compute the distribution over next states and the expected cost by integrating over the demand probabilities. The demand probabilities depend on the current day of the week, which is part of the state. For each possible demand realization, we use the step_dynamics function to find the next state and compute the cost incurred (outdate cost, shortage cost, holding cost, and production cost). We then aggregate these to get the overall distribution of next states and the expected cost for taking that action in the given state.
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
        if p <= 1e-12:
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


# ============================================================
# SOLVE AVERAGE-COST MDP BY DUAL LINEAR PROGRAM
# ============================================================

# Formulate and solve the dual linear program for the average-cost MDP. Then extract the optimal average cost and a deterministic stationary policy. Also create a DataFrame summarizing the policy for each state. The dual LP has a variable g representing the average cost and variables h[s] representing the relative value function for each state. The constraints ensure that g + h[s] <= expected_cost(s,a) + sum_{s'} P(s'|s,a) h[s'] for all states s and actions a. After solving the LP, we extract the optimal policy by choosing the action that minimizes the right-hand side of the constraint for each state, using the optimal h values. We also create a DataFrame that includes the weekday, total stock, optimal order, Bellman minimum value, number of tied best actions, and the list of tied best actions for each state.
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

# Given a deterministic policy, build the transition probability matrix P and cost vector c for the induced Markov chain. Also return the action taken in each state and the mapping from states to indices. The transition matrix P is constructed by iterating over each state, applying the deterministic policy to get the action, and then using the transition_distribution_and_expected_cost function to find the distribution over next states and the expected cost for that action. The cost vector c contains the expected cost for taking the action prescribed by the policy in each state. The action vector a_vec contains the action taken in each state according to the policy. The idx mapping allows us to convert between states and their corresponding indices in the matrix and vectors.
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

# Given the transition matrix P and cost vector c under a policy, compute the average cost by finding the stationary distribution of P and taking the weighted average of costs. The stationary distribution pi is found by solving the linear system pi = pi P with the normalization constraint sum(pi) = 1. Once we have pi, we can compute the average cost as the dot product of pi and c, which gives us the long-run average cost per time step under the given policy.
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

# Build a DataFrame that includes the stationary probability of each state under the given policy, along with the optimal order and total stock. The DataFrame is sorted by stationary probability (descending), weekday (ascending), and total stock (ascending) to highlight the most likely states under the optimal policy. This table provides insights into which states are most frequently visited under the optimal policy and what actions are taken in those states.
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

# This gives the most human-readable summary:
# conditional on weekday, what does the optimal stationary policy typically order? The function iterates over each weekday, identifies the states corresponding to that weekday, and calculates the probability of being in those states under the stationary distribution. It then computes the expected order quantity given that weekday by taking a weighted average of the actions prescribed by the deterministic policy, using the stationary probabilities as weights. It also identifies the most likely action given the weekday and the distribution of actions. The resulting DataFrame summarizes the typical ordering behavior of the optimal policy for each weekday, which can be more interpretable than the full state-dependent policy. It includes the probability of each weekday, the expected order given the weekday, the most likely order, and the distribution of actions given the weekday.
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

# Filter the state probability table to include only states with stationary probability above a certain cutoff. This helps to focus on the most relevant states under the optimal policy, as many states may have negligible probability and can be ignored for practical decision-making. The resulting DataFrame will contain only the states that are significantly visited under the optimal policy, making it easier to analyze and interpret the policy's behavior in those states.
def build_visited_state_table(state_prob_df: pd.DataFrame, cutoff: float = 1e-6) -> pd.DataFrame:
    return state_prob_df[state_prob_df["stationary_probability"] > cutoff].copy()


# ============================================================
# OUTPUT HELPERS
# ============================================================

# Build a compact summary table that aggregates the optimal order by weekday and total stock level, showing the average optimal order and how many state profiles correspond to each (weekday, total_stock) pair. This table provides a more concise summary of the optimal policy by showing the typical order quantity for each combination of weekday and total stock level, along with the number of different inventory age profiles that lead to that combination. It can help identify general patterns in the optimal policy without having to look at every individual state.
def build_compact_summary(policy_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        policy_df.groupby(["weekday", "total_stock"], sort=False)["optimal_order"]
        .agg(["min", "max", "mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_optimal_order", "count": "num_age_profiles"})
    )
    return summary

# ============================================================
# OUTPUT HELPERS (continued)    
# ============================================================
def plot_policy_heatmap_avg_order(policy_df: pd.DataFrame, output_path: Path) -> None:

    

    heatmap_df = (
        policy_df.pivot_table(
            index="weekday",
            columns="total_stock",
            values="optimal_order",
            aggfunc="mean"
        )
        .reindex(WEEKDAYS)
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap_df.to_numpy(), aspect="auto")
    plt.colorbar(label="Average optimal order")
    plt.xticks(
        ticks=np.arange(len(heatmap_df.columns)),
        labels=heatmap_df.columns
    )
    plt.yticks(
        ticks=np.arange(len(heatmap_df.index)),
        labels=heatmap_df.index
    )
    plt.xlabel("Total stock")
    plt.ylabel("Weekday")
    plt.title("Average optimal order by weekday and total stock")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_policy_heatmap_weighted_order(policy_df: pd.DataFrame, output_path: Path) -> None:

    grouped_rows = []
    for (weekday, total_stock), group in policy_df.groupby(["weekday", "total_stock"]):
        w = group["stationary_probability"].to_numpy()
        x = group["optimal_order"].to_numpy()

        if np.sum(w) <= 0:
            weighted_avg_order = np.nan
        else:
            weighted_avg_order = np.sum(w * x) / np.sum(w)

        grouped_rows.append({
            "weekday": weekday,
            "total_stock": total_stock,
            "weighted_avg_order": weighted_avg_order,
        })

    grouped = pd.DataFrame(grouped_rows)

    heatmap_df = (
        grouped.pivot(index="weekday", columns="total_stock", values="weighted_avg_order")
        .reindex(WEEKDAYS)
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap_df.to_numpy(), aspect="auto")
    plt.colorbar(label="Weighted average optimal order")
    plt.xticks(
        ticks=np.arange(len(heatmap_df.columns)),
        labels=heatmap_df.columns
    )
    plt.yticks(
        ticks=np.arange(len(heatmap_df.index)),
        labels=heatmap_df.index
    )
    plt.xlabel("Total stock")
    plt.ylabel("Weekday")
    plt.title("Weighted average optimal order by weekday and total stock")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_stationary_probability_by_stock(policy_df: pd.DataFrame, output_path: Path) -> None:

    prob_df = (
        policy_df.groupby(["weekday", "total_stock"], as_index=False)["stationary_probability"]
        .sum()
    )

    heatmap_df = (
        prob_df.pivot(index="weekday", columns="total_stock", values="stationary_probability")
        .reindex(WEEKDAYS)
        .fillna(0.0)
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(heatmap_df.to_numpy(), aspect="auto")
    plt.colorbar(label="Stationary probability")
    plt.xticks(
        ticks=np.arange(len(heatmap_df.columns)),
        labels=heatmap_df.columns
    )
    plt.yticks(
        ticks=np.arange(len(heatmap_df.index)),
        labels=heatmap_df.index
    )
    plt.xlabel("Total stock")
    plt.ylabel("Weekday")
    plt.title("Stationary probability mass by weekday and total stock")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_policy_scatter_all_states(policy_df: pd.DataFrame, output_path: Path) -> None:

    weekday_to_num = {day: i for i, day in enumerate(WEEKDAYS)}

    x = policy_df["total_stock"]
    y = policy_df["optimal_order"]
    c = policy_df["weekday"].map(weekday_to_num)

    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, c=c)
    plt.xlabel("Total stock")
    plt.ylabel("Optimal order")
    plt.title("Optimal order for all states")
    cbar = plt.colorbar()
    cbar.set_ticks(range(len(WEEKDAYS)))
    cbar.set_ticklabels(WEEKDAYS)
    cbar.set_label("Weekday")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_policy_scatter_weighted_states(policy_df: pd.DataFrame, output_path: Path) -> None:

    weekday_to_num = {day: i for i, day in enumerate(WEEKDAYS)}

    x = policy_df["total_stock"]
    y = policy_df["optimal_order"]
    c = policy_df["weekday"].map(weekday_to_num)
    sizes = 50 + 5000 * policy_df["stationary_probability"].fillna(0.0)

    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, c=c, s=sizes, alpha=0.6)
    plt.xlabel("Total stock")
    plt.ylabel("Optimal order")
    plt.title("Optimal order for visited states, weighted by stationary probability")
    cbar = plt.colorbar()
    cbar.set_ticks(range(len(WEEKDAYS)))
    cbar.set_ticklabels(WEEKDAYS)
    cbar.set_label("Weekday")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_order_distribution_by_weekday(policy_df: pd.DataFrame, output_path: Path) -> None:

    dist_df = (
        policy_df.groupby(["weekday", "optimal_order"], as_index=False)["stationary_probability"]
        .sum()
    )

    pivot_df = (
        dist_df.pivot(index="weekday", columns="optimal_order", values="stationary_probability")
        .reindex(WEEKDAYS)
        .fillna(0.0)
    )

    x = np.arange(len(pivot_df.index))
    bottom = np.zeros(len(pivot_df.index))

    plt.figure(figsize=(12, 6))
    for order_val in pivot_df.columns:
        vals = pivot_df[order_val].to_numpy()
        plt.bar(x, vals, bottom=bottom, label=f"Order {order_val}")
        bottom += vals

    plt.xticks(x, pivot_df.index, rotation=45)
    plt.xlabel("Weekday")
    plt.ylabel("Stationary probability mass")
    plt.title("Distribution of optimal orders by weekday")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()



# ============================================================
# MAIN
# ============================================================

# Main function to load demand probabilities, build the state and action space, solve the average-cost MDP using the dual linear program, extract the optimal policy, evaluate it to find the stationary distribution and average cost, and save the results to an Excel file. The function also prints out key information about the problem setup, the optimal long-run cost, a readable summary of the policy by weekday, and the most visited states under the optimal policy. It ensures that the shelf life is at least 2, as the model assumes that there are at least two age buckets (x_1, x_2, ..., x_{m-1}) in the inventory vector. The outputs include the full optimal policy for all states, a compact summary by weekday and total stock, a table of visited states with their stationary probabilities, and a weekday-level recommendation table that summarizes the typical order quantity for each weekday under the optimal policy.
def main():
    if SHELF_LIFE < 2:
        raise ValueError("SHELF_LIFE must be at least 2.")

    demand_pmf, K = load_demand_probabilities(DEMAND_XLSX_PATH, DEMAND_SHEET_NAME)


    min_demand_by_day = extract_min_demand_by_day(demand_pmf)
    max_total_by_day = compute_max_total_inventory_by_day(min_demand_by_day)
    print("Minimum demand with positive probability by weekday:", {WEEKDAYS[d]: k for d, k in min_demand_by_day.items()})
    print("Computed upper bound on total inventory by weekday:", {WEEKDAYS[d]: max_total_by_day[d] for d in range(7)})

    states = enumerate_states(INVENTORY_CAP, SHELF_LIFE)
    states = [s for s in states if structurally_feasible_state(s, max_total_by_day)]      # Filter out states that are impossible to reach under the production constraints, to reduce the state space and speed up computation.
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
    

    plot_policy_heatmap_avg_order(policy_df, PLOTS_DIR / "heatmap_avg_order.png")
    plot_policy_heatmap_weighted_order(policy_df,PLOTS_DIR / "heatmap_weighted_order.png")
    plot_stationary_probability_by_stock(policy_df,PLOTS_DIR / "heatmap_stationary_probability.png")
    plot_policy_scatter_all_states(policy_df,PLOTS_DIR / "scatter_all_states.png")
    plot_policy_scatter_weighted_states(policy_df,PLOTS_DIR / "scatter_weighted_states.png")
    plot_order_distribution_by_weekday(policy_df,PLOTS_DIR / "order_distribution_by_weekday.png")

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
    print(f"  Plots directory:           {PLOTS_DIR}")

    


if __name__ == "__main__":
    main()
