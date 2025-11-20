import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

penguin_df = pd.read_csv("./data/penguins.csv")
# print(penguin_df.head())

penguin_df = penguin_df.dropna()
output = penguin_df.species
features = penguin_df[
    [
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
    ]
]
features = pd.get_dummies(features)

print("OUTPUT")
print(output.head())
print("\nFEATURES")
print(features.head())


output, uniques = pd.factorize(output)

X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.2)

rfc = RandomForestClassifier(random_state=15)
rfc.fit(X_train.values, y_train)
y_pred = rfc.predict(X_test.values)
score = accuracy_score(y_pred, y_test)

print(f"Our accuracy score for this model is {score}")

output_dir = Path(__file__).parent / "outputs"
model_dir = output_dir / "models"

with open(model_dir / "random_forest_penguin.pickle", "wb") as f:
    pickle.dump(rfc, f)

with open(model_dir / "output_penguin.pickle", "wb") as f:
    pickle.dump(uniques, f)

fig, ax = plt.subplots(figsize=(12, 10))
ax = sns.barplot(
    x=rfc.feature_importances_, y=features.columns, hue=features.columns, legend=False
)
ax.set(
    title="Which features are the most important for species prediction?",
    xlabel="Importance",
    ylabel="Feature",
)
fig.tight_layout()

figure_dir = output_dir / "figures"
fig.savefig(figure_dir / "feature_importance.png")
