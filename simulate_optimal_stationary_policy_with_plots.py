from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import GRB


# ============================================================
# USER PARAMETERS
# ============================================================

DEMAND_XLSX_PATH = Path("weekday_demand_probabilities.xlsx")
DEMAND_SHEET_NAME = "DemandProbabilities"

SHELF_LIFE = 5
INVENTORY_CAP = 15
MAX_ORDER = 6
PRODUCTION_DAYS = {0, 1, 2, 3, 4}

C_OUTDATE = 4.0
C_SHORTAGE = 30.0
C_HOLDING = 1.0
C_PRODUCTION = 0.0

START_STATE = (4, 3, 2, 1, 0)  # (weekday, x1, x2, ..., x_{SHELF_LIFE-1})
SIMULATION_DAYS = 730
RANDOM_SEED = 42
N_REPLICATIONS = 1

OUTPUT_DIR = Path("data")
POLICY_OUTPUT_XLSX_PATH = OUTPUT_DIR / "optimal_stationary_policy_for_simulation.xlsx"
SIM_OUTPUT_XLSX_PATH = OUTPUT_DIR / "stationary_policy_simulation.xlsx"
PLOTS_DIR = OUTPUT_DIR / "simulation_plots"

REDUCED_COST_TOL = 1e-8


# ============================================================
# FIXED WEEKDAY ORDER
# ============================================================

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
State = Tuple[int, ...]


# ============================================================
# INPUT
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
# DYNAMICS
# ============================================================

def step_dynamics_detailed(
    state: State,
    action: int,
    demand: int,
    shelf_life: int,
):
    day = state[0]
    inv = list(state[1:])

    stock_before_issue = inv + [action]
    remaining_demand = demand
    remaining_stock = stock_before_issue[:]

    fifo_used = [0] * shelf_life
    for i in range(shelf_life):
        used = min(remaining_stock[i], remaining_demand)
        fifo_used[i] = used
        remaining_stock[i] -= used
        remaining_demand -= used

    shortage = float(remaining_demand)
    outdate = float(remaining_stock[0])

    next_inv = tuple(remaining_stock[1:])
    next_day = (day + 1) % 7
    next_state = (next_day,) + next_inv
    holding = float(sum(next_inv))

    period_cost = (
        C_OUTDATE * outdate
        + C_SHORTAGE * shortage
        + C_HOLDING * holding
        + C_PRODUCTION * action
    )

    return {
        "next_state": next_state,
        "stock_before_issue": tuple(stock_before_issue),
        "fifo_used": tuple(fifo_used),
        "shortage": shortage,
        "outdate": outdate,
        "holding_next_morning": holding,
        "period_cost": period_cost,
    }


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
        details = step_dynamics_detailed(state, action, demand, shelf_life)
        cost = details["period_cost"]
        next_state = details["next_state"]
        dist[next_state] = dist.get(next_state, 0.0) + float(p)
        expected_cost += float(p) * cost

    return dist, expected_cost


# ============================================================
# OPTIMAL STATIONARY POLICY
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
        }
        for i in range(len(s) - 1):
            row[f"x{i+1}"] = s[i+1]
        policy_rows.append(row)

    policy_df = pd.DataFrame(policy_rows)
    return g_star, det_policy, policy_df


# ============================================================
# SIMULATION
# ============================================================

def simulate_one_path(
    start_state: State,
    det_policy: Dict[State, int],
    demand_pmf: np.ndarray,
    shelf_life: int,
    n_days: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    state = start_state
    rows = []

    for t in range(n_days):
        day = state[0]
        action = det_policy[state]
        demand_support = np.arange(demand_pmf.shape[1])
        demand = int(rng.choice(demand_support, p=demand_pmf[day]))

        details = step_dynamics_detailed(state, action, demand, shelf_life)
        next_state = details["next_state"]

        row = {
            "t": t,
            "weekday": WEEKDAYS[day],
            "state_before": str(state),
            "total_stock_before": sum(state[1:]),
            "order": action,
            "demand": demand,
            "stock_before_issue": str(details["stock_before_issue"]),
            "shortage": details["shortage"],
            "outdate": details["outdate"],
            "holding_next_morning": details["holding_next_morning"],
            "state_after": str(next_state),
            "period_cost": details["period_cost"],
        }

        for i in range(shelf_life - 1):
            row[f"x{i+1}_before"] = state[i + 1]
            row[f"x{i+1}_after"] = next_state[i + 1]

        rows.append(row)
        state = next_state

    sim_df = pd.DataFrame(rows)
    sim_df["cumulative_cost"] = sim_df["period_cost"].cumsum()
    sim_df["week"] = sim_df["t"] // 7 + 1
    return sim_df


def build_daily_and_weekly_summaries(sim_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_summary = (
        sim_df.groupby("weekday", sort=False)
        .agg(
            avg_order=("order", "mean"),
            avg_demand=("demand", "mean"),
            avg_inventory=("total_stock_before", "mean"),
            avg_shortage=("shortage", "mean"),
            avg_outdate=("outdate", "mean"),
            avg_period_cost=("period_cost", "mean"),
            n_obs=("t", "count"),
        )
        .reset_index()
    )

    weekly_summary = (
        sim_df.groupby("week")
        .agg(
            weekly_cost=("period_cost", "sum"),
            weekly_order=("order", "sum"),
            weekly_demand=("demand", "sum"),
            weekly_shortage=("shortage", "sum"),
            weekly_outdate=("outdate", "sum"),
        )
        .reset_index()
    )

    return daily_summary, weekly_summary


def build_replication_summary(replication_results: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for i, sim_df in enumerate(replication_results, start=1):
        rows.append({
            "replication": i,
            "avg_cost_per_day": sim_df["period_cost"].mean(),
            "avg_cost_per_week": sim_df.groupby("week")["period_cost"].sum().mean(),
            "avg_order_per_day": sim_df["order"].mean(),
            "avg_shortage_per_day": sim_df["shortage"].mean(),
            "avg_outdate_per_day": sim_df["outdate"].mean(),
        })
    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def plot_single_path_order_vs_demand(sim_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(sim_df["t"], sim_df["order"], label="Optimal order")
    plt.plot(sim_df["t"], sim_df["demand"], label="Realized demand")
    plt.xlabel("Day")
    plt.ylabel("Units")
    plt.title("Simulation path: optimal order and realized demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_single_path_inventory_components(sim_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    for i in range(1, SHELF_LIFE):
        plt.plot(sim_df["t"], sim_df[f"x{i}_before"], label=f"x{i} before")
    plt.xlabel("Day")
    plt.ylabel("Units")
    plt.title("Simulation path: inventory by remaining shelf life")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_single_path_costs(sim_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(sim_df["t"], sim_df["period_cost"], label="Period cost")
    plt.plot(sim_df["t"], sim_df["cumulative_cost"], label="Cumulative cost")
    plt.xlabel("Day")
    plt.ylabel("Cost")
    plt.title("Simulation path: daily and cumulative cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_single_path_shortage_outdate(sim_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(sim_df["t"], sim_df["shortage"], label="Shortage")
    plt.plot(sim_df["t"], sim_df["outdate"], label="Outdate")
    plt.xlabel("Day")
    plt.ylabel("Units")
    plt.title("Simulation path: shortage and outdating")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_weekly_summary(weekly_summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_summary["week"], weekly_summary["weekly_cost"], label="Weekly cost")
    plt.plot(weekly_summary["week"], weekly_summary["weekly_order"], label="Weekly order")
    plt.plot(weekly_summary["week"], weekly_summary["weekly_demand"], label="Weekly demand")
    plt.xlabel("Week")
    plt.ylabel("Value")
    plt.title("Weekly summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_replication_average_costs(replication_summary_df: pd.DataFrame, output_path: Path) -> None:
    if len(replication_summary_df) <= 1:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(replication_summary_df["replication"], replication_summary_df["avg_cost_per_week"], marker="o")
    plt.xlabel("Replication")
    plt.ylabel("Average cost per week")
    plt.title("Average weekly cost across replications")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_total_inventory(sim_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(sim_df["t"], sim_df["total_stock_before"], label="Total inventory before demand")
    plt.xlabel("Day")
    plt.ylabel("Units")
    plt.title("Simulation path: total inventory over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_order_demand_inventory(sim_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(sim_df["t"], sim_df["order"], label="Optimal order")
    plt.plot(sim_df["t"], sim_df["demand"], label="Realized demand")
    plt.plot(sim_df["t"], sim_df["total_stock_before"], label="Total inventory before demand")
    plt.xlabel("Day")
    plt.ylabel("Units")
    plt.title("Simulation path: order, demand, and inventory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_demand_histogram(sim_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(sim_df["demand"], bins=range(int(sim_df["demand"].max()) + 2), align="left", rwidth=0.8)
    plt.xlabel("Demand")
    plt.ylabel("Frequency")
    plt.title("Histogram of realized demand")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_order_histogram(sim_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(sim_df["order"], bins=range(int(sim_df["order"].max()) + 2), align="left", rwidth=0.8)
    plt.xlabel("Order")
    plt.ylabel("Frequency")
    plt.title("Histogram of optimal orders used in simulation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_rolling_average_cost(sim_df: pd.DataFrame, output_path: Path, window: int = 7) -> None:
    rolling_cost = sim_df["period_cost"].rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(sim_df["t"], rolling_cost, label=f"{window}-day rolling average cost")
    plt.xlabel("Day")
    plt.ylabel("Cost")
    plt.title("Simulation path: rolling average of period cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_rolling_shortage_outdate(sim_df: pd.DataFrame, output_path: Path, window: int = 7) -> None:
    rolling_shortage = sim_df["shortage"].rolling(window=window, min_periods=1).mean()
    rolling_outdate = sim_df["outdate"].rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(sim_df["t"], rolling_shortage, label=f"{window}-day rolling shortage")
    plt.plot(sim_df["t"], rolling_outdate, label=f"{window}-day rolling outdate")
    plt.xlabel("Day")
    plt.ylabel("Units")
    plt.title("Simulation path: rolling shortage and outdating")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_weekly_shortage_outdate(weekly_summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_summary["week"], weekly_summary["weekly_shortage"], label="Weekly shortage")
    plt.plot(weekly_summary["week"], weekly_summary["weekly_outdate"], label="Weekly outdate")
    plt.xlabel("Week")
    plt.ylabel("Units")
    plt.title("Weekly shortage and outdating")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_inventory_age_mix(sim_df: pd.DataFrame, output_path: Path) -> None:
    x_cols = [f"x{i}_before" for i in range(1, SHELF_LIFE)]
    y_values = [sim_df[col].to_numpy() for col in x_cols]

    plt.figure(figsize=(12, 6))
    plt.stackplot(sim_df["t"], *y_values, labels=x_cols)
    plt.xlabel("Day")
    plt.ylabel("Units")
    plt.title("Simulation path: inventory age composition")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_order_vs_demand_scatter(sim_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(sim_df["demand"], sim_df["order"])
    plt.xlabel("Realized demand")
    plt.ylabel("Optimal order")
    plt.title("Order versus realized demand")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_weekday_average_orders_and_demand(daily_summary: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(daily_summary))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, daily_summary["avg_order"], width=width, label="Average order")
    plt.bar(x + width / 2, daily_summary["avg_demand"], width=width, label="Average demand")
    plt.xticks(x, daily_summary["weekday"], rotation=45)
    plt.xlabel("Weekday")
    plt.ylabel("Units")
    plt.title("Average order and demand by weekday")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_weekday_order_demand_inventory(daily_summary: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(daily_summary))
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(x - width, daily_summary["avg_order"], width=width, label="Average order")
    plt.bar(x, daily_summary["avg_demand"], width=width, label="Average demand")
    plt.bar(x + width, daily_summary["avg_inventory"], width=width, label="Average inventory")

    plt.xticks(x, daily_summary["weekday"], rotation=45)
    plt.xlabel("Weekday")
    plt.ylabel("Units")
    plt.title("Average order, demand, and inventory by weekday")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_weekday_shortage_outdate_counts(sim_df: pd.DataFrame, output_path: Path) -> None:
    summary = (
        sim_df.groupby("weekday", sort=False)
        .agg(
            total_shortage=("shortage", "sum"),
            total_outdate=("outdate", "sum"),
        )
        .reset_index()
    )

    x = np.arange(len(summary))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(x - width / 2, summary["total_shortage"], width=width, label="Total shortage")
    plt.bar(x + width / 2, summary["total_outdate"], width=width, label="Total outdate")

    plt.xticks(x, summary["weekday"], rotation=45)
    plt.xlabel("Weekday")
    plt.ylabel("Units")
    plt.title("Total shortage and outdating by weekday")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if SHELF_LIFE < 2:
        raise ValueError("SHELF_LIFE must be at least 2.")
    if len(START_STATE) != SHELF_LIFE:
        raise ValueError(
            f"START_STATE must have length {SHELF_LIFE}: (day, x1, ..., x_{SHELF_LIFE-1})"
        )
    if START_STATE[0] not in range(7):
        raise ValueError("The first entry of START_STATE must be a weekday index in {0,...,6}.")
    if any(x < 0 for x in START_STATE[1:]):
        raise ValueError("Inventory entries in START_STATE must be nonnegative.")
    if sum(START_STATE[1:]) > INVENTORY_CAP:
        raise ValueError("START_STATE exceeds INVENTORY_CAP.")

    demand_pmf, K = load_demand_probabilities(DEMAND_XLSX_PATH, DEMAND_SHEET_NAME)
    states = enumerate_states(INVENTORY_CAP, SHELF_LIFE)
    actions_by_state = {
        s: feasible_actions(s, INVENTORY_CAP, MAX_ORDER, PRODUCTION_DAYS)
        for s in states
    }

    print("=" * 74)
    print("SIMULATION OF THE OPTIMAL STATIONARY POLICY")
    print("=" * 74)
    print(f"Demand file:         {DEMAND_XLSX_PATH}")
    print(f"Demand sheet:        {DEMAND_SHEET_NAME}")
    print(f"Max demand K:        {K}")
    print(f"Shelf life:          {SHELF_LIFE}")
    print(f"Inventory cap:       {INVENTORY_CAP}")
    print(f"Max order:           {MAX_ORDER}")
    print(f"Production days:     {[WEEKDAYS[d] for d in sorted(PRODUCTION_DAYS)]}")
    print(f"Costs:               c0={C_OUTDATE}, cs={C_SHORTAGE}, cH={C_HOLDING}, cp={C_PRODUCTION}")
    print(f"Start state:         {START_STATE}")
    print(f"Simulation days:     {SIMULATION_DAYS}")
    print(f"Replications:        {N_REPLICATIONS}")
    print("=" * 74)

    g_star, det_policy, policy_df = solve_average_cost_lp(
        states, actions_by_state, demand_pmf, SHELF_LIFE
    )

    with pd.ExcelWriter(POLICY_OUTPUT_XLSX_PATH, engine="openpyxl") as writer:
        policy_df.to_excel(writer, sheet_name="OptimalPolicy_AllStates", index=False)

    print("\nOPTIMAL LONG-RUN COST FROM LP")
    print(f"Per day:   {g_star:.6f}")
    print(f"Per week:  {7.0 * g_star:.6f}")

    replication_results = []
    first_daily_summary = None
    first_weekly_summary = None

    with pd.ExcelWriter(SIM_OUTPUT_XLSX_PATH, engine="openpyxl") as writer:
        for rep in range(N_REPLICATIONS):
            sim_df = simulate_one_path(
                start_state=START_STATE,
                det_policy=det_policy,
                demand_pmf=demand_pmf,
                shelf_life=SHELF_LIFE,
                n_days=SIMULATION_DAYS,
                seed=RANDOM_SEED + rep,
            )
            daily_summary, weekly_summary = build_daily_and_weekly_summaries(sim_df)

            sim_df.to_excel(writer, sheet_name=f"Path_{rep+1}", index=False)
            daily_summary.to_excel(writer, sheet_name=f"DaySummary_{rep+1}", index=False)
            weekly_summary.to_excel(writer, sheet_name=f"WeekSummary_{rep+1}", index=False)

            replication_results.append(sim_df)

            if rep == 0:
                first_daily_summary = daily_summary
                first_weekly_summary = weekly_summary

                plot_single_path_order_vs_demand(sim_df, PLOTS_DIR / "path1_order_vs_demand.png")
                plot_single_path_inventory_components(sim_df, PLOTS_DIR / "path1_inventory_components.png")
                plot_single_path_costs(sim_df, PLOTS_DIR / "path1_costs.png")
                plot_single_path_shortage_outdate(sim_df, PLOTS_DIR / "path1_shortage_outdate.png")
                plot_weekly_summary(weekly_summary, PLOTS_DIR / "path1_weekly_summary.png")
                plot_total_inventory(sim_df, PLOTS_DIR / "path1_total_inventory.png")
                plot_order_demand_inventory(sim_df, PLOTS_DIR / "path1_order_demand_inventory.png")
                plot_demand_histogram(sim_df, PLOTS_DIR / "path1_demand_histogram.png")
                plot_order_histogram(sim_df, PLOTS_DIR / "path1_order_histogram.png")
                plot_rolling_average_cost(sim_df, PLOTS_DIR / "path1_rolling_average_cost.png")
                plot_rolling_shortage_outdate(sim_df, PLOTS_DIR / "path1_rolling_shortage_outdate.png")
                plot_weekly_shortage_outdate(weekly_summary, PLOTS_DIR / "path1_weekly_shortage_outdate.png")   
                plot_inventory_age_mix(sim_df, PLOTS_DIR / "path1_inventory_age_mix.png")
                plot_order_vs_demand_scatter(sim_df, PLOTS_DIR / "path1_order_vs_demand_scatter.png")
                plot_weekday_average_orders_and_demand(daily_summary, PLOTS_DIR / "path1_weekday_avg_orders_demand.png")
                plot_weekday_order_demand_inventory(daily_summary, PLOTS_DIR / "path1_weekday_order_demand_inventory.png")
                plot_weekday_shortage_outdate_counts(sim_df, PLOTS_DIR / "path1_weekday_shortage_outdate_counts.png")

        replication_summary_df = build_replication_summary(replication_results)
        replication_summary_df.to_excel(writer, sheet_name="ReplicationSummary", index=False)

    plot_replication_average_costs(replication_summary_df, PLOTS_DIR / "replication_avg_costs.png")

    print("\nFIRST SIMULATED PATH: WEEKDAY SUMMARY")
    print(first_daily_summary.to_string(index=False))

    print("\nFIRST SIMULATED PATH: FIRST WEEKS")
    print(first_weekly_summary.head(10).to_string(index=False))

    print("\nINTERPRETATION")
    print("The LP gives the optimal long-run average cost in the stationary model.")
    print("The simulation then applies the resulting optimal policy day by day from START_STATE,")
    print("with random demand draws from the weekday demand distributions.")
    print("So the simulated weekly costs will fluctuate around the long-run benchmark;")
    print("they are not identical to the LP value in each single week.")

    print("\nSaved files:")
    print(f"  Policy used in simulation: {POLICY_OUTPUT_XLSX_PATH}")
    print(f"  Simulation output:         {SIM_OUTPUT_XLSX_PATH}")
    print(f"  Plots directory:           {PLOTS_DIR}")


if __name__ == "__main__":
    main()
