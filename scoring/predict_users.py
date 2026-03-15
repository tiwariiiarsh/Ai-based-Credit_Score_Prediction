import pandas as pd
import pickle

# load model
bundle = pickle.load(open("models/xgb_model.pkl","rb"))
model = bundle["model"]
feature_cols = bundle["feature_cols"]
imputer = bundle["imputer"]

# load users
df_new = pd.read_csv("data/new_users.csv")

# take first 100 users
users = df_new.iloc[:100]

# prepare features
X = imputer.transform(users[feature_cols])

# predict scores
scores = model.predict(X)

# band function
def band(score):
    if score < 580:
        return "Poor"
    elif score < 670:
        return "Fair"
    elif score < 740:
        return "Good"
    elif score < 800:
        return "Very Good"
    else:
        return "Excellent"

# print results
for i, s in enumerate(scores):
    print("User", i, "| Score:", round(s), "| Band:", band(s))

# optional: save to csv
users["predicted_score"] = scores
users["band"] = users["predicted_score"].apply(band)

users.to_csv("predicted_scores.csv", index=False)

print("\nSaved → predicted_scores.csv")