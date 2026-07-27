import numpy as np
import pandas as pd
import os

rng = np.random.RandomState(42)
n_train, n_test = 1000, 100

def generate_data(n, rng):
    amount = rng.uniform(10, 25000, n)
    hour = rng.uniform(0, 24, n)
    distance = rng.uniform(0, 500, n)
    prev_attempts = rng.poisson(0.5, n).astype(float)
    country_change = rng.binomial(1, 0.15, n).astype(float)
    device_new = rng.binomial(1, 0.1, n).astype(float)
    amount_ratio = rng.uniform(0.1, 20, n)
    velocity = rng.uniform(0, 10, n)

    amount_n = (amount - 10) / (25000 - 10)
    hour_n = hour / 24
    distance_n = distance / 500
    amount_ratio_n = (amount_ratio - 0.1) / (20 - 0.1)
    velocity_n = velocity / 10

    logit = (
        3.0 * amount_n
        - 2.0 * (hour_n - 0.5) ** 2
        + 1.5 * distance_n
        + 3.0 * prev_attempts * 0.2
        + 2.0 * country_change
        + 1.5 * device_new
        + 2.5 * amount_ratio_n
        + 2.0 * velocity_n
        - 3.0
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    is_fraud = (p > 0.5).astype(float)
    flip = rng.binomial(1, 0.02, n)
    is_fraud = np.where(flip == 1, 1.0 - is_fraud, is_fraud)

    return pd.DataFrame({
        "amount": amount,
        "hour": hour,
        "distance": distance,
        "prev_attempts": prev_attempts,
        "country_change": country_change,
        "device_new": device_new,
        "amount_ratio": amount_ratio,
        "velocity": velocity,
        "is_fraud": is_fraud,
    })

train_df = generate_data(n_train, rng)
test_df = generate_data(n_test, rng)

out_dir = os.path.dirname(os.path.abspath(__file__))
train_df.to_csv(os.path.join(out_dir, "train_fraud.csv"), index=False)
test_df.to_csv(os.path.join(out_dir, "test_fraud.csv"), index=False)
print(f"Train: {len(train_df)} rows, fraud rate={train_df['is_fraud'].mean():.3f}")
print(f"Test:  {len(test_df)} rows, fraud rate={test_df['is_fraud'].mean():.3f}")
