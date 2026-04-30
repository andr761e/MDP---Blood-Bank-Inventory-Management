import math
from datetime import date, timedelta

import numpy as np
import pandas as pd


# ============================================================
# PARAMETERS
# ============================================================

# Demand is measured in platelet pools before scaling.
# If units_per_model_unit = 2, then 1 model unit corresponds to 2 platelet pools.
units_per_model_unit = 1

# AUH 2025 demand data: Monday, Tuesday, ..., Sunday.
calendar_year = 2025
annual_weekday_demands = [963, 883, 837, 939, 905, 607, 555]

# IMPORTANT INTERPRETATION:
# max_noise is the full width of the stochastic demand band.
# Example: if the weekday base demand is 15 and max_noise = 10,
# then demand is supported on {10, 11, ..., 20}.
# Equivalently, the signed deviation is {-5, -4, ..., 5} when max_noise is even.
max_noise = 10

# If True, lambda is fitted so the shifted truncated Poisson has exactly the
# empirical weekday mean. This avoids systematically adding demand.
fit_lambda_to_centered_mean = True

# Output used by the policy scripts.
output_path = "weekday_demand_probabilities.xlsx"


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ============================================================
# BASIC HELPERS
# ============================================================

def count_weekdays_in_year(year: int) -> list[int]:
    """Counts how many Mondays, Tuesdays, ..., Sundays occur in a calendar year."""
    counts = [0] * 7
    current = date(year, 1, 1)
    while current.year == year:
        counts[current.weekday()] += 1
        current += timedelta(days=1)
    return counts


def truncated_poisson_probs(max_value: int, lam: float) -> list[float]:
    """
    Returns probabilities for Z in {0, 1, ..., max_value}, where Z follows
    a Poisson(lam) distribution truncated above max_value and then renormalized.

    Probabilities are computed recursively to avoid numerical issues from
    factorials and exp(-lambda) cancelling during normalization.
    """
    if max_value < 0:
        raise ValueError("max_value must be nonnegative.")

    if lam < 0:
        raise ValueError("lam must be nonnegative.")

    if lam == 0:
        return [1.0] + [0.0] * max_value

    weights = [1.0]
    for k in range(1, max_value + 1):
        weights.append(weights[-1] * lam / k)

    total = sum(weights)
    return [w / total for w in weights]


def expected_value(probs: list[float]) -> float:
    return sum(k * p for k, p in enumerate(probs))


def solve_lambda_for_truncated_poisson_mean(target_mean: float, max_value: int) -> float:
    """
    Finds lambda such that a truncated Poisson on {0,...,max_value} has
    expected value approximately equal to target_mean.
    """
    if target_mean < 0:
        raise ValueError("target_mean must be nonnegative.")

    if target_mean > max_value:
        raise ValueError(
            f"target_mean={target_mean:.4f} is larger than max_value={max_value}. "
            "Increase max_noise or change the scaling."
        )

    if math.isclose(target_mean, 0.0, abs_tol=1e-12):
        return 0.0

    if math.isclose(target_mean, float(max_value), abs_tol=1e-12):
        return 1e6

    lo = 0.0
    hi = max(1.0, target_mean)

    while expected_value(truncated_poisson_probs(max_value, hi)) < target_mean:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Could not bracket lambda for truncated Poisson mean.")

    for _ in range(100):
        mid = (lo + hi) / 2.0
        mean_mid = expected_value(truncated_poisson_probs(max_value, mid))
        if mean_mid < target_mean:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


# ============================================================
# DEMAND CONSTRUCTION
# ============================================================

def compute_weekday_means_from_annual_counts(
    year: int,
    annual_weekday_demands: list[int],
    units_per_model_unit: int,
) -> pd.DataFrame:
    """
    Converts annual weekday totals into expected demand per occurrence of each weekday.
    Example: Monday demand is divided by the number of Mondays in the year.
    """
    if len(annual_weekday_demands) != 7:
        raise ValueError("annual_weekday_demands must contain exactly 7 numbers.")

    if any(x < 0 for x in annual_weekday_demands):
        raise ValueError("annual_weekday_demands must be nonnegative.")

    if units_per_model_unit <= 0:
        raise ValueError("units_per_model_unit must be positive.")

    weekday_counts = count_weekdays_in_year(year)

    rows = []
    for weekday, annual_demand, n_days in zip(WEEKDAYS, annual_weekday_demands, weekday_counts):
        mean_real = annual_demand / n_days
        mean_model = mean_real / units_per_model_unit
        rows.append({
            "weekday": weekday,
            "annual_demand_real_units": annual_demand,
            "number_of_weekdays_in_year": n_days,
            "mean_real_units_per_day": mean_real,
            "mean_model_units_per_day": mean_model,
            "base_demand_nearest_integer": int(round(mean_model)),
        })

    return pd.DataFrame(rows)


def build_centered_shifted_poisson_demand_matrix(
    mean_df: pd.DataFrame,
    max_noise: int,
    fit_lambda_to_centered_mean: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds a weekday-by-demand probability matrix where the demand distribution
    is centered around the empirical weekday mean.

    Construction:
        Z_d in {0, 1, ..., max_noise}
        D_d = lower_demand_d + Z_d

    The lower demand is chosen so the support is approximately symmetric around
    the weekday mean. If the weekday base demand is 15 and max_noise = 10, then
    lower = 10 and D is supported on {10, 11, ..., 20}.

    Lambda is fitted so E[D_d] matches the empirical weekday mean. Therefore the
    stochastic term can move realized demand both below and above the expected level
    without increasing expected demand mechanically.
    """
    if max_noise < 0:
        raise ValueError("max_noise must be nonnegative.")

    rows = []
    max_total_demand = 0

    # First pass: determine supports and lambdas.
    for row in mean_df.to_dict("records"):
        mean_model = float(row["mean_model_units_per_day"])

        lower_demand = max(0, int(round(mean_model - max_noise / 2)))
        upper_demand = lower_demand + max_noise
        target_offset_mean = mean_model - lower_demand

        if target_offset_mean < 0:
            raise ValueError(
                f"For {row['weekday']}, target_offset_mean={target_offset_mean:.4f} is negative. "
                "This should not happen unless the support construction is changed."
            )

        if target_offset_mean > max_noise:
            raise ValueError(
                f"For {row['weekday']}, target_offset_mean={target_offset_mean:.4f} is larger "
                f"than max_noise={max_noise}. Increase max_noise or change scaling."
            )

        if fit_lambda_to_centered_mean:
            lam = solve_lambda_for_truncated_poisson_mean(
                target_mean=target_offset_mean,
                max_value=max_noise,
            )
        else:
            lam = target_offset_mean

        probs = truncated_poisson_probs(max_noise, lam)
        expected_offset = expected_value(probs)
        expected_total = lower_demand + expected_offset
        max_total_demand = max(max_total_demand, upper_demand)

        rows.append({
            **row,
            "max_noise_full_width_model_units": max_noise,
            "lower_demand_model_units": lower_demand,
            "upper_demand_model_units": upper_demand,
            "target_offset_mean_model_units": target_offset_mean,
            "fitted_truncated_poisson_lambda": lam,
            "expected_offset_model_units": expected_offset,
            "expected_total_model_units": expected_total,
            "expected_total_real_units": expected_total * units_per_model_unit,
            "signed_deviation_min_from_nearest_base": lower_demand - int(round(mean_model)),
            "signed_deviation_max_from_nearest_base": upper_demand - int(round(mean_model)),
        })

    split_df = pd.DataFrame(rows)
    matrix = np.zeros((7, max_total_demand + 1))
    noise_rows = []

    for i, row in enumerate(split_df.to_dict("records")):
        lower_demand = int(row["lower_demand_model_units"])
        lam = float(row["fitted_truncated_poisson_lambda"])
        probs = truncated_poisson_probs(max_noise, lam)

        for offset, p in enumerate(probs):
            total_demand = lower_demand + offset
            matrix[i, total_demand] += p

        noise_rows.append({
            "weekday": row["weekday"],
            "lower_demand_model_units": lower_demand,
            "upper_demand_model_units": int(row["upper_demand_model_units"]),
            "target_offset_mean_model_units": row["target_offset_mean_model_units"],
            "fitted_truncated_poisson_lambda": lam,
            "expected_total_model_units": row["expected_total_model_units"],
            **{f"P_offset_{k}_demand_{lower_demand + k}": probs[k] for k in range(max_noise + 1)},
        })

    demand_df = pd.DataFrame(
        matrix,
        index=split_df["weekday"].tolist(),
        columns=[f"demand_{k}" for k in range(max_total_demand + 1)],
    )

    noise_df = pd.DataFrame(noise_rows)
    return demand_df, split_df, noise_df


def build_assumptions_df(demand_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "parameter": [
            "calendar_year",
            "annual_total_real_units",
            "annual_weekday_demands_real_units",
            "units_per_model_unit",
            "max_noise_full_width_model_units",
            "fit_lambda_to_centered_mean",
            "max_total_demand_column_K",
            "interpretation",
            "example",
        ],
        "value": [
            calendar_year,
            int(sum(annual_weekday_demands)),
            str(annual_weekday_demands),
            units_per_model_unit,
            max_noise,
            fit_lambda_to_centered_mean,
            demand_df.shape[1] - 1,
            "D = lower_demand + Z, where Z is truncated Poisson on {0,...,max_noise}; E[D] matches the 2025 weekday mean.",
            "If base demand is 15 and max_noise is 10, support is demand 10 through 20.",
        ],
    })


if __name__ == "__main__":
    mean_df = compute_weekday_means_from_annual_counts(
        year=calendar_year,
        annual_weekday_demands=annual_weekday_demands,
        units_per_model_unit=units_per_model_unit,
    )

    demand_df, split_df, noise_df = build_centered_shifted_poisson_demand_matrix(
        mean_df=mean_df,
        max_noise=max_noise,
        fit_lambda_to_centered_mean=fit_lambda_to_centered_mean,
    )

    assumptions_df = build_assumptions_df(demand_df=demand_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        demand_df.to_excel(writer, sheet_name="DemandProbabilities")
        split_df.to_excel(writer, sheet_name="WeekdayCenteredDemand", index=False)
        noise_df.to_excel(writer, sheet_name="CenteredPoissonOffset", index=False)
        assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)

    print("AUH 2025 centered weekday demand distributions:")
    print(split_df)
    print("\nCentered shifted Poisson offset distributions:")
    print(noise_df)
    print(f"\nSaved to: {output_path}")
