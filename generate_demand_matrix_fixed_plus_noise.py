import math
import numpy as np
import pandas as pd


# Parameters for demand generation
units_per_model_unit = 6  # 1 model unit = 7 real products; this is to keep the LP manageable
max_noise = 3             # noise in {0,1,...,max_noise}
noise_lambdas = [1.2, 1.3, 1.4, 1.3, 1.2, 0.5, 0.6] # Poisson lambda for each weekday's noise; can be adjusted as needed 
weekday_weights = [24, 19, 30, 18, 23, 5, 7] # relative weekday weights; based on historical data but can be adjusted as needed


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def truncated_poisson_noise_probs(max_noise: int, lam: float) -> list[float]:
    """
    Returns probabilities for noise = 0,1,...,max_noise
    using a Poisson(lam), truncated at max_noise and renormalized.
    """
    probs = [poisson_pmf(k, lam) for k in range(max_noise + 1)]
    total = sum(probs)
    return [p / total for p in probs]


def allocate_weekday_base_demands(
    annual_total: float,
    units_per_model_unit: int,
    weekday_weights: list[float]
) -> tuple[list[int], float, list[float]]:
    """
    Converts annual total demand into base weekday demands.

    annual_total:
        total annual platelet products (e.g. pools + aphereses)

    units_per_model_unit:
        aggregation factor; if = 5, then 1 model unit = 5 real products

    weekday_weights:
        relative weekday weights, not necessarily normalized
    """
    if len(weekday_weights) != 7:
        raise ValueError("weekday_weights must contain exactly 7 numbers.")

    if any(w <= 0 for w in weekday_weights):
        raise ValueError("weekday_weights must be strictly positive.")

    avg_daily_demand = annual_total / 365.0 / units_per_model_unit

    weights = np.array(weekday_weights, dtype=float)
    weights = weights / weights.mean()   # normalize so average weight = 1

    weekday_means = avg_daily_demand * weights
    base_demands = [int(round(x)) for x in weekday_means]

    return base_demands, avg_daily_demand, weekday_means.tolist()


def build_fixed_plus_weekday_noise_demand_matrix(
    base_demands: list[int],
    noise_lambdas: list[float],
    max_noise: int
) -> pd.DataFrame:
    """
    Builds a 7 x (K+1) demand matrix where weekday d has demand:
        base_demands[d] + noise_d
    with noise_d in {0,1,...,max_noise}, where the noise distribution
    is weekday-specific and follows a truncated Poisson(noise_lambdas[d]).
    """
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if len(base_demands) != 7:
        raise ValueError("base_demands must contain exactly 7 integers.")

    if len(noise_lambdas) != 7:
        raise ValueError("noise_lambdas must contain exactly 7 numbers.")

    if any(d < 0 for d in base_demands):
        raise ValueError("All base demands must be nonnegative.")

    if any(lam <= 0 for lam in noise_lambdas):
        raise ValueError("All noise lambdas must be strictly positive.")

    K = max(base_demands) + max_noise
    matrix = np.zeros((7, K + 1))

    noise_prob_rows = []

    for i, (base, lam) in enumerate(zip(base_demands, noise_lambdas)):
        probs = truncated_poisson_noise_probs(max_noise=max_noise, lam=lam)

        for noise, p in enumerate(probs):
            matrix[i, base + noise] += p

        noise_prob_rows.append({
            "weekday": weekdays[i],
            "noise_lambda": lam,
            **{f"P_noise_{k}": probs[k] for k in range(max_noise + 1)}
        })

    df = pd.DataFrame(
        matrix,
        index=weekdays,
        columns=[f"demand_{k}" for k in range(K + 1)]
    )

    noise_df = pd.DataFrame(noise_prob_rows)
    return df, noise_df


if __name__ == "__main__":
    # ============================================================
    # USER INPUT
    # ============================================================

    # AUH totals
    use_multi_year_average = True
    multi_year_totals = [
        7459 + 358,   # 2023
        7548 + 360,   # 2022
        6915 + 162,   # 2021
        7917 + 327,   # 2020
        7469 + 256,   # 2019
    ]

    annual_total_2023 = 7459 + 358

    if use_multi_year_average:
        annual_total = sum(multi_year_totals) / len(multi_year_totals)
    else:
        annual_total = annual_total_2023

    output_path = "weekday_demand_probabilities.xlsx"

    # ============================================================
    # BUILD BASE + NOISE
    # ============================================================

    base_demands, avg_daily_demand, weekday_means = allocate_weekday_base_demands(
        annual_total=annual_total,
        units_per_model_unit=units_per_model_unit,
        weekday_weights=weekday_weights,
    )

    df, noise_df = build_fixed_plus_weekday_noise_demand_matrix(
        base_demands=base_demands,
        noise_lambdas=noise_lambdas,
        max_noise=max_noise,
    )

    summary_df = pd.DataFrame({
        "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "weekday_weight": weekday_weights,
        "weekday_mean_before_rounding": weekday_means,
        "base_demand": base_demands,
    })

    assumptions_df = pd.DataFrame({
        "parameter": [
            "annual_total_products",
            "units_per_model_unit",
            "average_daily_demand_in_model_units",
            "max_noise",
            "K"
        ],
        "value": [
            annual_total,
            units_per_model_unit,
            avg_daily_demand,
            max_noise,
            df.shape[1] - 1
        ]
    })

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="DemandProbabilities")
        summary_df.to_excel(writer, sheet_name="WeekdayBaseDemand", index=False)
        noise_df.to_excel(writer, sheet_name="NoiseDistribution", index=False)
        assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)

    print("Base weekday demands:")
    print(summary_df)
    print("\nWeekday-specific noise distributions:")
    print(noise_df)
    print(f"\nSaved to: {output_path}")