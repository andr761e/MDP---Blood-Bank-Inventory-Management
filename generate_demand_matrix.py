import math
import pandas as pd


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF: P(D = k) for mean lam."""
    return math.exp(-lam) * lam**k / math.factorial(k)


def truncated_poisson_row(K: int, lam: float) -> list[float]:
    """
    Generate probabilities for demand 0,1,...,K using a Poisson(lam),
    truncated at K and renormalized so the row sums to 1.
    """
    probs = [poisson_pmf(k, lam) for k in range(K + 1)]
    total = sum(probs)
    return [p / total for p in probs]


def build_demand_matrix(K: int, weekday_means: list[float]) -> pd.DataFrame:
    """
    Returns a 7 x (K+1) DataFrame where each row is a weekday demand distribution.
    """
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if len(weekday_means) != 7:
        raise ValueError("weekday_means must contain exactly 7 numbers.")

    matrix = [truncated_poisson_row(K, lam) for lam in weekday_means]
    df = pd.DataFrame(matrix, index=weekdays, columns=[f"demand_{k}" for k in range(K + 1)])

    # Helpful control column
    #df["row_sum"] = df.sum(axis=1)
    return df


if __name__ == "__main__":
    # -----------------------------
    # USER INPUT
    # -----------------------------
    K = 5

    # Realistic toy means:
    # - slightly lower early week
    # - higher Thursday/Friday
    # - still positive demand on weekend
    weekday_means = [2.5, 3.5, 4.0, 3.5, 3.0, 2.0, 1.5]

    output_path = "weekday_demand_probabilities.xlsx"

    # -----------------------------
    # BUILD AND SAVE
    # -----------------------------
    df = build_demand_matrix(K, weekday_means)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="DemandProbabilities")

    print("Demand probability matrix:")
    print(df.round(4))
    print(f"\nSaved to: {output_path}")
