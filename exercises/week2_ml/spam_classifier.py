# day 1 - spam classifier using naive bayes
# dataset is small/made up just to keep this self contained

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

messages = [
    "Win a free iPhone now, click this link",
    "Congratulations you have won $1000 cash prize",
    "URGENT: your account has been suspended, verify now",
    "Claim your free vacation package today",
    "You have been selected for a special offer, act fast",
    "Limited time deal, buy one get one free",
    "Hey, are we still on for lunch tomorrow?",
    "Can you send me the report before end of day",
    "Meeting moved to 3pm, let me know if that works",
    "Thanks for your help earlier, really appreciate it",
    "Reminder: dentist appointment tomorrow morning",
    "Let's catch up this weekend if you're free",
]

labels = [
    "spam", "spam", "spam", "spam", "spam", "spam",
    "not_spam", "not_spam", "not_spam", "not_spam", "not_spam", "not_spam",
]


def run():
    X_train, X_test, y_train, y_test = train_test_split(
        messages, labels, test_size=0.25, random_state=42
    )

    # turn text into word count vectors
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    print("Predictions vs actual:")
    for msg, pred, actual in zip(X_test, predictions, y_test):
        print(f"  '{msg[:40]}...' -> predicted: {pred}, actual: {actual}")

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.2f}")
    print(classification_report(y_test, predictions, zero_division=0))

    # quick sanity check on a new message
    new_msg = ["You have won a free prize, click here now"]
    new_vec = vectorizer.transform(new_msg)
    print(f"\nNew message: '{new_msg[0]}'")
    print(f"Predicted: {model.predict(new_vec)[0]}")


if __name__ == "__main__":
    run()
