import numpy as np
import pandas as pd


def build_deterministic_demand_matrix(daily_demands: list[int], K: int) -> pd.DataFrame:
    """
    Builds a 7 x (K+1) demand matrix where each row has probability 1
    on exactly one demand value.
    """
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if len(daily_demands) != 7:
        raise ValueError("daily_demands must contain exactly 7 integers.")

    if any(d < 0 for d in daily_demands):
        raise ValueError("All daily demands must be nonnegative.")

    if max(daily_demands) > K:
        raise ValueError("K must be at least as large as the largest daily demand.")

    matrix = np.zeros((7, K + 1))
    for i, d in enumerate(daily_demands):
        matrix[i, d] = 1.0

    df = pd.DataFrame(
        matrix,
        index=weekdays,
        columns=[f"demand_{k}" for k in range(K + 1)]
    )
    return df


def allocate_weekday_demands_from_annual_total(
    annual_total: float,
    units_per_model_unit: int,
    weekday_weights: list[float]
) -> tuple[list[int], float, list[float]]:
    """
    Converts annual total demand into deterministic weekday demands.

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
    deterministic_demands = [int(round(x)) for x in weekday_means]

    return deterministic_demands, avg_daily_demand, weekday_means.tolist()


if __name__ == "__main__":
    # ============================================================
    # USER INPUT
    # ============================================================

    # AUH production figures (choose either one year or multi-year average)
    annual_pool = 7459
    annual_apheresis = 358

    # Alternative: multi-year average
    use_multi_year_average = True
    multi_year_totals = [
        7459 + 358,   # 2023
        7548 + 360,   # 2022
        6915 + 162,   # 2021
        7917 + 327,   # 2020
        7469 + 256,   # 2019
    ]

    if use_multi_year_average:
        annual_total = sum(multi_year_totals) / len(multi_year_totals)
    else:
        annual_total = annual_pool + annual_apheresis

    # IMPORTANT:
    # If 1 model unit = 1 real product, the demand gets quite large.
    # To keep the exact LP tractable, you may want aggregation.
    units_per_model_unit = 5

    # Weekday structure:
    # more booked activity on weekdays, lower on weekend
    weekday_weights = [1.10, 1.10, 1.05, 1.05, 1.00, 0.85, 0.85]

    # Output file
    output_path = "weekday_demand_probabilities.xlsx"

    # ============================================================
    # BUILD DEMAND
    # ============================================================

    daily_demands, avg_daily_demand, weekday_means = allocate_weekday_demands_from_annual_total(
        annual_total=annual_total,
        units_per_model_unit=units_per_model_unit,
        weekday_weights=weekday_weights,
    )

    K = max(daily_demands)

    df = build_deterministic_demand_matrix(daily_demands, K)

    summary_df = pd.DataFrame({
        "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "weekday_weight": weekday_weights,
        "weekday_mean_before_rounding": weekday_means,
        "deterministic_demand": daily_demands,
    })

    assumptions_df = pd.DataFrame({
        "parameter": [
            "annual_total_products",
            "units_per_model_unit",
            "average_daily_demand_in_model_units",
            "K"
        ],
        "value": [
            annual_total,
            units_per_model_unit,
            avg_daily_demand,
            K
        ]
    })

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="DemandProbabilities")
        summary_df.to_excel(writer, sheet_name="WeekdayDemandSummary", index=False)
        assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)

    print("Deterministic weekday demands:")
    print(summary_df)
    print(f"\nSaved to: {output_path}")