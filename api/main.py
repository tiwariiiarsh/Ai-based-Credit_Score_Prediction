from fastapi import FastAPI
import mysql.connector
from scoring.score import score_user

app = FastAPI(title="Darcrays Credit Scoring API")

def get_all_users():
    conn = mysql.connector.connect(
        host="localhost", user="root",
        password="Arsh@1106", database="Barclays_Bank"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM users")
    ids = [r[0] for r in cursor.fetchall()]
    conn.close()
    return ids

@app.get("/")
def root():
    return {"message": "Darcrays Credit Scoring API", "status": "running"}

@app.get("/score/{user_id}")
def score(user_id: str):
    return score_user(user_id)

@app.get("/score_all")
def score_all():
    ids = get_all_users()
    results = []
    for uid in ids[:50]:
        try:
            results.append(score_user(uid))
        except Exception as e:
            results.append({"user_id": uid, "error": str(e)})
    return results

@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost + GMM", "features": 52}
