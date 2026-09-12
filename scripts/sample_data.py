"""Generate a realistic sample dataset for local development and the demo.

Deliberately contains the problems the platform is supposed to catch: an identifier, a
constant column, a high-missingness column, a leaked copy of the target, a post-outcome
column and a free-text column.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def build_churn_dataset(rows: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, rows)
    monthly_charges = np.round(rng.normal(70, 25, rows).clip(15, 180), 2)
    support_tickets = rng.poisson(1.2, rows)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], rows, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["card", "transfer", "invoice", "direct-debit"], rows)
    region = rng.choice(["north", "south", "east", "west", "central"], rows)

    risk = (
        0.04 * support_tickets
        + 0.012 * (monthly_charges - 70) / 10
        - 0.02 * tenure / 12
        + np.where(contract == "month-to-month", 0.9, -0.4)
        + rng.normal(0, 0.6, rows)
    )
    churned = (risk > 0.7).astype(int)

    frame = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(rows)],
            "signup_date": pd.to_datetime("2019-01-01")
            + pd.to_timedelta(rng.integers(0, 1500, rows), unit="D"),
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": np.round(monthly_charges * tenure * rng.uniform(0.9, 1.1, rows), 2),
            "support_tickets": support_tickets,
            "contract_type": contract,
            "payment_method": payment,
            "region": region,
            "has_fibre": rng.choice([True, False], rows, p=[0.6, 0.4]),
            "satisfaction_score": rng.integers(1, 6, rows).astype(float),
            "data_region_code": "EU",
            "last_survey_comment": rng.choice(
                [
                    "the service has been reliable but the price increase was not communicated",
                    "support took several days to respond to my connectivity problem",
                    "happy overall, the technician visit was handled professionally",
                ],
                rows,
            ),
            # Deliberate problems for the platform to find:
            "account_closed_reason": np.where(
                churned == 1, rng.choice(["price", "service"], rows), None
            ),
            "churn_label_copy": churned,
            "churned": churned,
        }
    )
    missing = rng.random(rows) < 0.45
    frame.loc[missing, "satisfaction_score"] = np.nan
    frame.loc[rng.random(rows) < 0.02, "monthly_charges"] = np.nan
    return frame


def build_price_dataset(rows: int = 1500, seed: int = 11) -> pd.DataFrame:
    """Regression counterpart, used to exercise the regression path."""
    rng = np.random.default_rng(seed)
    area = rng.normal(95, 30, rows).clip(25, 300)
    rooms = np.maximum(1, (area / 30 + rng.normal(0, 0.6, rows)).round())
    age = rng.integers(0, 90, rows)
    district = rng.choice(["centre", "north", "south", "harbour"], rows)
    premium = (
        pd.Series(district)
        .map({"centre": 1800, "north": 900, "south": 950, "harbour": 1500})
        .to_numpy()
    )
    price = 1200 * area + 8000 * rooms - 900 * age + premium * 10 + rng.normal(0, 25_000, rows)
    return pd.DataFrame(
        {
            "listing_id": range(rows),
            "area_sqm": np.round(area, 1),
            "rooms": rooms,
            "building_age_years": age,
            "district": district,
            "has_balcony": rng.choice([True, False], rows),
            "price_eur": np.round(price, 2),
        }
    )


def write_sample_datasets(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    churn = directory / "customer_churn.csv"
    price = directory / "property_prices.parquet"
    build_churn_dataset().to_csv(churn, index=False)
    build_price_dataset().to_parquet(price, index=False)
    return {"churn": churn, "price": price}


if __name__ == "__main__":  # pragma: no cover - developer utility
    paths = write_sample_datasets(Path("var/sample-data"))
    for name, path in paths.items():
        print(f"{name}: {path}")
