from src.fetch_mysql_data import fetch_data
from src.feature_engineering_mysql import build_features
from src.train_gmm_mysql import train_gmm
from src.train_xgboost_mysql import train_xgboost


def run_pipeline():

    print("STEP 1: Fetching raw data from MySQL...")

    users, transactions, loans, utilities, monthly = fetch_data()

    print("Users:", users.shape)
    print("Transactions:", transactions.shape)
    print("Loans:", loans.shape)
    print("Utilities:", utilities.shape)
    print("Monthly:", monthly.shape)


    print("\nSTEP 2: Building features...")

    features = build_features(
        users,
        transactions,
        loans,
        utilities,
        monthly
    )

    print("Feature dataset shape:", features.shape)


    print("\nSTEP 3: Training GMM clustering...")

    clustered_df = train_gmm(features)

    print("Cluster distribution:")
    print(clustered_df["cluster"].value_counts())


    print("\nSTEP 4: Training XGBoost model...")

    train_xgboost(clustered_df)


    print("\nTraining complete")


if __name__ == "__main__":

    run_pipeline()







# from src.fetch_mysql_data import fetch_data
# from src.feature_engineering_mysql import build_features
# from src.train_gmm_mysql import train_gmm
# from src.train_xgboost_mysql import train_xgboost

# users, transactions, loans, utilities, monthly = fetch_data()

# features = build_features(users, transactions, loans, utilities, monthly)

# clustered = train_gmm(features)

# train_xgboost(clustered)

# print("Training complete")