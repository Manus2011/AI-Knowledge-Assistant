"""
Week 2 - Day 2: Customer prediction model

Predicts whether a customer will make a purchase based on a few basic
features (time on site, pages viewed, past purchases). Uses Logistic
Regression since this is a binary classification problem (will buy / won't buy).

Data is synthetic, generated with some intentional pattern (more time on
site + more past purchases = more likely to buy) so the model actually
has something real to learn from.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def generate_data(n=200, seed=42):
    rng = np.random.default_rng(seed)

    time_on_site = rng.uniform(0, 20, n)          # minutes
    pages_viewed = rng.integers(1, 15, n)
    past_purchases = rng.integers(0, 10, n)

    # rough underlying pattern with some randomness mixed in
    score = (time_on_site * 0.3) + (pages_viewed * 0.4) + (past_purchases * 1.2)
    noise = rng.normal(0, 3, n)
    will_purchase = (score + noise > 12).astype(int)

    return pd.DataFrame({
        "time_on_site": time_on_site,
        "pages_viewed": pages_viewed,
        "past_purchases": past_purchases,
        "will_purchase": will_purchase,
    })


def run():
    df = generate_data()
    print("Sample of the data:")
    print(df.head())
    print(f"\nPurchase rate in dataset: {df['will_purchase'].mean():.2%}")

    X = df[["time_on_site", "pages_viewed", "past_purchases"]]
    y = df["will_purchase"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print(f"\nAccuracy: {acc:.2%}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)

    # which feature matters most, based on the model's learned coefficients
    print("\nFeature weights:")
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"  {feature}: {coef:.3f}")


if __name__ == "__main__":
    run()
