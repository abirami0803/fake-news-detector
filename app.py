import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

print("Data padikuthu...")
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake["label"] = 0
true["label"] = 1
data = pd.concat([fake, true])
data["content"] = data["title"] + " " + data["text"]
data = data[["content", "label"]].dropna()
x_train, x_test, y_train, y_test = train_test_split(data["content"], data["label"], test_size=0.2)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(xv_train, y_train)
print("Model ready!")

while True:
    news = input("\nNews type pannu (exit):\n> ")
    if news.lower() == "exit":
        break
    pred = model.predict(vectorizer.transform([news]))[0]
    if pred == 0:
        print("Result: FAKE NEWS")
    else:
        print("Result: TRUE NEWS")