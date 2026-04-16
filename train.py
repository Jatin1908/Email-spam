import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

print("🚀 Training Start")

# Load dataset
df = pd.read_csv("spam.csv", encoding='ISO-8859-1')

# Keep only needed columns
df = df[['v1','v2']]
df.columns = ['label','message']

# Convert labels
df['label'] = df['label'].map({'ham':0,'spam':1})

# Check dataset
print("Label count:")
print(df['label'].value_counts())

# Vectorization (IMPORTANT 🔥)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train-test split (better model)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model with class balance fix 🔥
model = MultinomialNB(class_prior=[0.4, 0.6])
model.fit(X_train, y_train)

# Accuracy check
accuracy = model.score(X_test, y_test)
print(f"✅ Accuracy: {accuracy:.2f}")

# Save model
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("✅ TRAINING DONE & PKL FILE CREATED")