import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

print("Data padikuthu...")
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data["content"] = data["title"] + " " + data["text"]
data = data[["content", "label"]].dropna()

x_train, x_test, y_train, y_test = train_test_split(data["content"], data["label"], test_size=0.2, random_state=42)

vector = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_vec = vector.fit_transform(x_train)
x_test_vec = vector.transform(x_test)

model = PassiveAggressiveClassifier()
model.fit(x_train_vec, y_train)

pred = model.predict(x_test_vec)
print(f"Accuracy: {accuracy_score(y_test, pred)*100:.2f}%")

def check_news(news):
    news_vec = vector.transform([news])
    result = model.predict(news_vec)
    return "REAL News" if result[0]==1 else "FAKE News"
