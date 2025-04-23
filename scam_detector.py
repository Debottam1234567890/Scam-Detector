import csv
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import joblib

def load_csv_data(filename):
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return [row for row in reader]

def preprocess_data(data):
    messages = [row[0] for row in data]
    labels = [1 if row[1].lower() == "scam" else 0 for row in data]
    features = [assign_values_to_factors(message) for message in messages]
    return features, labels

def assign_values_to_factors(message):
    message_lower = message.lower()

    def score_from_keywords(keywords):
        return sum(1 for kw in keywords if kw in message_lower) / len(keywords)

    urgency_keywords = [
        "urgent", "immediately", "asap", "now", "instantly", "right away", "critical",
        "emergency", "act fast", "without delay", "rush", "time sensitive", "immediate attention",
        "important", "priority", "respond quickly", "final notice", "quickly", "within hours", "last chance"
    ]

    money_keywords = [
        "send money", "payment", "bank account", "transfer", "fund", "financial assistance",
        "deposit", "remit", "wire", "moneygram", "western union", "btc", "crypto", "currency",
        "dollars", "cash", "fee", "transaction", "cheque", "inheritance", "unclaimed funds", "reward",
        "$", "million", "billion", "usd", "50m", "50 million", "50$", "50000", "lot of money", "wealth"
    ]

    official_keywords = [
        "official", "government", "irs", "fbi", "customs", "account update", "verification",
        "authority", "compliance", "legal", "investigation", "officer", "department", "administrator",
        "state", "national", "hq", "regulation", "policy", "internal audit"
    ]

    reward_keywords = [
        "win", "congratulations", "lucky", "jackpot", "lottery", "cash prize", "gift card",
        "you've won", "sweepstakes", "bingo", "claim prize", "million", "billion", "bonanza",
        "reward", "exclusive prize", "you qualify", "redeem", "receive funds", "special winner"
    ]

    celebrity_keywords = [
        "elon musk", "taylor swift", "jeff bezos", "bill gates", "oprah", "lebron", "cristiano",
        "selena", "beyonce", "trump", "biden", "modi", "virat", "shahrukh", "kardashian", 
        "celebrity", "hollywood", "influencer", "verified", "blue tick"
    ]

    grammar_issues_keywords = [
        "recieve", "seperated", "definately", "adress", "freind", "untill", "wich", "immediatly",
        "inconvienent", "completly", "alot", "happend", "beleive", "enviroment", "goverment",
        "neccessary", "occurence", "seperate", "succesful", "truely"
    ]

    contact_keywords = [
        "telegram", "whatsapp", "sms", "text", "chat", "dm", "message me", "contact via app",
        "reach me", "wechat", "imo", "viber", "signal", "messenger", "snapchat", "facebook",
        "call this number", "ping me", "alternative number", "line"
    ]

    pressure_keywords = [
        "act now", "limited time", "only today", "last chance", "hurry", "urgent deadline",
        "before it’s too late", "don't miss out", "one-time offer", "expires soon",
        "final offer", "time running out", "claim fast", "do not delay", "fast response",
        "limited stock", "urgent response needed", "need quick answer", "instantly confirm", "must act quickly"
    ]

    link_keywords = [
        "http", "https", "bit.ly", "tinyurl", "shorturl", "redirect", ".xyz", ".top", ".win",
        "click here", "open link", "see details", "login page", "promo code", "verify link",
        "security page", "unusual login", "confirm access", "track order", "claim voucher"
    ]

    upfront_keywords = [
        "pay upfront", "advance payment", "initial deposit", "send fee", "registration fee", 
        "processing charge", "application cost", "service fee", "transfer cost", "one-time charge",
        "security fee", "membership fee", "setup cost", "handling fee", "deposit first",
        "pay before", "cash advance", "shipping fee", "booking charge", "consultation fee"
    ]

    return [
        round(score_from_keywords(urgency_keywords), 2),
        round(score_from_keywords(money_keywords), 2),
        round(score_from_keywords(official_keywords), 2),
        round(score_from_keywords(reward_keywords), 2),
        round(score_from_keywords(celebrity_keywords), 2),
        round(score_from_keywords(grammar_issues_keywords), 2),
        round(score_from_keywords(contact_keywords), 2),
        round(score_from_keywords(pressure_keywords), 2),
        round(score_from_keywords(link_keywords), 2),
        round(score_from_keywords(upfront_keywords), 2)
    ]

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, predictions))
    return model

def classify_message(model, message):
    features = [assign_values_to_factors(message)]
    prediction = model.predict(features)[0]
    print("\n🤖 Prediction:", "SCAM" if prediction == 1 else "NOT SCAM")

if __name__ == "__main__":
    print("📥 Loading data...")
    data = load_csv_data("labeled_dataset.csv")
    X, y = preprocess_data(data)
    model = train_model(X, y)
    joblib.dump(model, "scam_detector_model.pkl")

    # Test input
    while True:
        msg = input("\n💬 Enter a message to check (or 'quit' to exit):\n")
        if msg.lower() == 'quit':
            break
        classify_message(model, msg)

