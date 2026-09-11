"""
DermaSensor - Rule-Based Synthetic Dataset Generator
======================================================
Generates a physiologically-grounded synthetic dataset that maps three
wound-bed sensor readings -> infection risk class:

    - temperature (°C)  : local wound-site skin temperature
    - ph_level          : wound exudate pH
    - moisture (%)      : wound bed moisture / exudate saturation

Clinical rationale for the thresholds used (used as the "ground truth"
rule engine that labels every synthetic sample):

    Temperature:
        < 37.5 C            -> normal peri-wound temperature
        37.5 - 38.5 C       -> mild local inflammation
        > 38.5 C            -> significant local heat, early infection sign

    pH:
        4.0 - 6.0           -> normal healing wound (mildly acidic)
        6.0 - 7.5           -> transitional / stalled healing
        > 7.5               -> alkaline shift, associated with bacterial
                                bioburden / infection
        < 4.0               -> abnormally acidic (rare, also flagged)

    Moisture:
        20 - 65 %           -> normal moist wound-healing range
        65 - 85 %           -> elevated exudate
        > 85 %               -> excessive exudate / maceration risk,
                                common with infected wounds

Each factor contributes an integer "risk point" score. The summed score is
thresholded into three classes: Low, Medium, High. Gaussian sensor noise
and a small label-flip probability are added so the dataset behaves like
real (noisy) sensor data rather than a perfectly separable rule table,
which gives the downstream ML model something non-trivial to learn while
keeping it fully explainable and auditable against the same rules.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_SAMPLES = 6000
LABEL_NOISE_PROB = 0.04  # 4% of samples get their label randomly perturbed

RISK_CLASSES = ["Low", "Medium", "High"]


def temperature_points(temp: float) -> int:
    if temp > 38.5:
        return 2
    if temp >= 37.5:
        return 1
    return 0


def ph_points(ph: float) -> int:
    if ph > 8.0:
        return 2
    if ph > 7.5:
        return 1
    if ph < 4.0:
        return 1  # abnormally acidic also counts as a risk signal
    return 0


def moisture_points(moisture: float) -> int:
    if moisture > 85:
        return 2
    if moisture > 65:
        return 1
    return 0


def score_to_class(score: int) -> str:
    if score <= 1:
        return "Low"
    if score <= 3:
        return "Medium"
    return "High"


def rule_engine(temp: float, ph: float, moisture: float):
    """Returns (risk_class, score, per_factor_points) using the ground-truth
    clinical rule set. This exact function is reused by the backend for
    explainability, so the model's explanation always traces back to the
    same rules the dataset was generated from."""
    tp = temperature_points(temp)
    pp = ph_points(ph)
    mp = moisture_points(moisture)
    score = tp + pp + mp
    return score_to_class(score), score, {"temperature": tp, "ph_level": pp, "moisture": mp}


def generate_dataset(n=N_SAMPLES, seed=RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []
    # Sample roughly evenly across three physiological "regimes" so every
    # risk class is well represented, then add continuous noise on top so
    # class boundaries aren't perfectly crisp.
    regime_choices = rng.integers(0, 3, size=n)

    for regime in regime_choices:
        if regime == 0:  # mostly-normal wound
            temp = rng.normal(36.6, 0.5)
            ph = rng.normal(5.0, 0.7)
            moisture = rng.normal(45, 12)
        elif regime == 1:  # borderline / early-warning wound
            temp = rng.normal(37.9, 0.5)
            ph = rng.normal(6.8, 0.8)
            moisture = rng.normal(70, 10)
        else:  # infected-leaning wound
            temp = rng.normal(38.9, 0.6)
            ph = rng.normal(8.2, 0.6)
            moisture = rng.normal(90, 8)

        temp = float(np.clip(temp, 34.0, 41.5))
        ph = float(np.clip(ph, 3.0, 9.5))
        moisture = float(np.clip(moisture, 5.0, 100.0))

        risk_class, score, _ = rule_engine(temp, ph, moisture)

        # small label noise to simulate real-world ambiguity/sensor error
        if rng.random() < LABEL_NOISE_PROB:
            risk_class = rng.choice(RISK_CLASSES)

        rows.append(
            {
                "temperature": round(temp, 2),
                "ph_level": round(ph, 2),
                "moisture": round(moisture, 2),
                "risk_score": score,
                "risk_class": risk_class,
            }
        )

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "dermasensor_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["risk_class"].value_counts())
    print(df.describe())
