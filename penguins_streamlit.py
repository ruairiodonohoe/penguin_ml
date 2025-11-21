import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# st.write(rfc)
# st.write(mapping)
st.title("Penguin Classifier: A Machine Learning App")
st.write(
    "This app uses 6 inputs to predict the species of penguin using"
    "a model built on the Palmer Penguins dataset. Use the form below"
    " to get started!"
)

# password_guess = st.text_input("What is the Password?")
# if password_guess != st.secrets["password"]:
#     st.stop()


penguin_file = st.file_uploader("Upload your own penguin data.")
outputs_dir = Path(__file__).parent / "outputs"

if penguin_file is None:
    model_output_dir = outputs_dir / "models"
    model_path = model_output_dir / "random_forest_penguin.pickle"
    mapping_map = model_output_dir / "output_penguin.pickle"
    with open(model_path, "rb") as f:
        rfc = pickle.load(f)
    with open(mapping_map, "rb") as f:
        label_mapping = pickle.load(f)
else:
    penguin_df = pd.read_csv(penguin_file)
    penguin_df = penguin_df.dropna()
    output = penguin_df["species"]
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
    output, label_mapping = pd.factorize(output)
    X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.2)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(X_train.values, y_train)
    y_pred = rfc.predict(X_test.values)
    score = round(accuracy_score(y_pred, y_test), 2)
    st.write(
        "We trained a Random Forest model on these  data, "
        f"it has a score of {score}! "
        "Use the  inputs below to try out the model"
    )

with st.form("user_inputs"):
    bill_length = st.number_input("Bill Length (mm)", min_value=0)
    bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
    flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
    body_mass = st.number_input("Body Mass (g)", min_value=0)
    island = st.selectbox("Penguin Island", options=["Biscoe", "Dream", "Torgerson"])
    sex = st.selectbox("Sex", options=["Female", "Male"])
    user_inputs = [bill_length, bill_depth, flipper_length, body_mass, island, sex]
    st.write(f"The User Inputs are {user_inputs}.")
    st.form_submit_button()

island_biscoe, island_dream, island_torgerson = (
    int(island == "Biscoe"),
    int(island == "Dream"),
    int(island == "Torgerson"),
)
sex_female, sex_male = int(sex == "Female"), int(sex == "Male")

input_features = [
    [
        bill_length,
        bill_depth,
        flipper_length,
        body_mass,
        island_biscoe,
        island_dream,
        island_torgerson,
        sex_female,
        sex_male,
    ]
]
st.write(f"The input features are {input_features}.")


new_prediction = rfc.predict(input_features)
prediction_species = label_mapping[new_prediction][0]

st.subheader("Predicting Your Penguin's Species:")
st.write(f"We predict your new penguin is of the {prediction_species} species.")
st.write(
    """We used a machine learning (Random Forest) 
    model to predict the species, the features 
    used in this prediction are ranked by 
    relative importance below."""
)
figures_dir = outputs_dir / "figures"
st.image(figures_dir / "feature_importance.png")
st.write(
    """Below are the histograms for each
 continuous variable separated by penguin
 species. The vertical line represents
 your the inputted value."""
)

data_dir = Path(__file__).parent / "data"
penguin_df = pd.read_csv(data_dir / "penguins.csv")
fig, ax = plt.subplots()
ax = sns.histplot(x=penguin_df["bill_length_mm"], hue=penguin_df.species)
ax.axvline(bill_length)
ax.set_title("Bill Length by Species")
st.pyplot(fig)

fig, ax = plt.subplots()
ax = sns.histplot(x=penguin_df["bill_depth_mm"], hue=penguin_df.species)
ax.axvline(bill_depth)
ax.set_title("Bill Depth by Species")
st.pyplot(fig)

fig, ax = plt.subplots()
ax = sns.histplot(x=penguin_df["flipper_length_mm"], hue=penguin_df.species)
ax.axvline(flipper_length)
ax.set_title("Flipper Length by Species")
st.pyplot(fig)
