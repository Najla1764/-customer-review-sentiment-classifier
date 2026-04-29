import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── STEP 1: Load your data ──────────────────────────
df = pd.read_excel("raw_data.xlsx")
print(f"Total reviews loaded: {len(df)}")

# ── STEP 2: Remove UNCLEAR rows ─────────────────────
df = df[df["label"] != "UNCLEAR"]
df = df.dropna(subset=["label"])
print(f"Reviews after cleaning: {len(df)}")

# ── STEP 3: Prepare data ────────────────────────────
X = df["customer_review"]
y = df["label"]

# ── STEP 4: Convert text to numbers ─────────────────
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# ── STEP 5: Split into train and test ───────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)
print(f"Training reviews : {X_train.shape[0]}")
print(f"Testing reviews  : {X_test.shape[0]}")

# ── STEP 6: Train the model ─────────────────────────
model = MultinomialNB()
model.fit(X_train, y_train)
print("\n✅ Model trained successfully!")

# ── STEP 7: Test the model ──────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {round(accuracy * 100, 2)}%")

# ── STEP 8: Try your own reviews ────────────────────
def predict_sentiment(review):
    vec = vectorizer.transform([review])
    result = model.predict(vec)[0]
    print(f"\nReview     : {review}")
    print(f"Prediction : {result} ✅")

print("\n--- Testing with new reviews ---")
predict_sentiment("This product is amazing, very happy!")
predict_sentiment("Worst purchase ever, complete waste of money")
predict_sentiment("Product is okay, nothing special")

# ── STEP 9: Save final report ────────────────────────
report = pd.DataFrame({
    "Metric": [
        "Total Reviews",
        "Training Reviews",
        "Testing Reviews",
        "Model Accuracy"
    ],
    "Value": [
        len(df),
        X_train.shape[0],
        X_test.shape[0],
        f"{round(accuracy * 100, 2)}%"
    ]
})
report.to_excel("final_report.xlsx", index=False)
print("\n📊 Final report saved to final_report.xlsx ✅")