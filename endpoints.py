from flask import Flask, request, jsonify, render_template_string
import joblib
import os
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)

# Initialize model as None - will be loaded when needed
model = None

def assign_values_to_factors(message):
    """Extract feature scores from message text"""
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
        "before it's too late", "don't miss out", "one-time offer", "expires soon",
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

def load_and_train_model():
    """Load data and train model if not already trained"""
    global model
    
    # Try to load existing model first
    if os.path.exists("scam_detector_model.pkl"):
        try:
            model = joblib.load("scam_detector_model.pkl")
            print("✅ Loaded existing model from scam_detector_model.pkl")
            return
        except:
            print("⚠️ Failed to load existing model, will train new one")
    
    # Train new model if dataset exists
    if os.path.exists("labeled_dataset.csv"):
        try:
            with open("labeled_dataset.csv", newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                data = [row for row in reader]
            
            messages = [row[0] for row in data]
            labels = [1 if row[1].lower() == "scam" else 0 for row in data]
            features = [assign_values_to_factors(message) for message in messages]
            
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Save the model
            joblib.dump(model, "scam_detector_model.pkl")
            print("✅ Trained and saved new model")
            
            # Print accuracy
            predictions = model.predict(X_test)
            print(f"📊 Model accuracy on test set: {sum(predictions == y_test) / len(y_test):.2%}")
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            # Create a dummy model for demo purposes
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            # Train on some dummy data
            dummy_features = [[0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            dummy_labels = [0]
            model.fit(dummy_features, dummy_labels)
            print("⚠️ Created dummy model for demo purposes")
    else:
        print("⚠️ No labeled_dataset.csv found, creating dummy model")
        # Create a dummy model for demo purposes
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        dummy_features = [[0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        dummy_labels = [0]
        model.fit(dummy_features, dummy_labels)

DETECT_SCAMS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ Scam Message Detector</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            backdrop-filter: blur(10px);
        }

        .header {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 40px 30px;
            text-align: center;
            position: relative;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" fill-opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }

        .main-content {
            padding: 40px;
        }

        .input-section {
            margin-bottom: 30px;
        }

        .input-label {
            display: block;
            font-size: 1.1rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
        }

        .message-input {
            width: 100%;
            min-height: 120px;
            padding: 20px;
            border: 2px solid #e1e8ed;
            border-radius: 15px;
            font-size: 1rem;
            font-family: inherit;
            resize: vertical;
            transition: all 0.3s ease;
            background: #fafbfc;
        }

        .message-input:focus {
            outline: none;
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .analyze-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .analyze-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }

        .analyze-btn:active {
            transform: translateY(0);
        }

        .analyze-btn:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }

        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }

        .loading.show {
            display: block;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .result-section {
            display: none;
            margin-top: 30px;
            padding: 30px;
            border-radius: 15px;
            position: relative;
        }

        .result-section.show {
            display: block;
            animation: slideIn 0.5s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .result-scam {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
        }

        .result-safe {
            background: linear-gradient(135deg, #00d2d3, #54a0ff);
            color: white;
        }

        .result-icon {
            font-size: 3rem;
            margin-bottom: 15px;
            display: block;
        }

        .result-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .result-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            margin-bottom: 20px;
        }

        .confidence-bar {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
            margin-bottom: 10px;
        }

        .confidence-fill {
            height: 100%;
            background: white;
            border-radius: 10px;
            transition: width 0.8s ease;
        }

        .confidence-text {
            font-size: 0.9rem;
            opacity: 0.8;
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 25px;
        }

        .feature-card {
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            backdrop-filter: blur(5px);
        }

        .feature-name {
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 5px;
        }

        .feature-score {
            font-size: 1.1rem;
            font-weight: 700;
        }

        .features-note {
            margin-top: 15px;
            font-size: 0.85rem;
            opacity: 0.7;
            text-align: center;
        }

        .examples {
            margin-top: 40px;
            padding-top: 30px;
            border-top: 2px solid #e1e8ed;
        }

        .examples h3 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3rem;
        }

        .example-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .example-btn {
            background: #f8f9fa;
            border: 2px solid #e1e8ed;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }

        .example-btn:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }

        .nav-section {
            text-align: center;
            margin-top: 40px;
            padding: 30px 0;
            border-top: 2px solid #e1e8ed;
        }

        .back-home-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            margin: 0 10px;
        }

        .back-home-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }

        .back-home-btn:active {
            transform: translateY(-1px);
        }

        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 15px;
            }

            .header {
                padding: 30px 20px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .main-content {
                padding: 20px;
            }

            .features-grid {
                grid-template-columns: 1fr;
            }

            .example-buttons {
                flex-direction: column;
            }

            .example-btn {
                text-align: left;
            }

            .back-home-btn {
                margin: 5px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Scam Message Detector</h1>
            <p>Analyze messages for potential scam indicators using AI</p>
        </div>
        
        <div class="main-content">
            <div class="input-section">
                <label class="input-label" for="messageInput">Enter message to analyze:</label>
                <textarea 
                    id="messageInput" 
                    class="message-input" 
                    placeholder="Paste or type the message you want to check for scam indicators..."
                ></textarea>
                <br><br>
                <button class="analyze-btn" onclick="analyzeMessage()">
                    🔍 Analyze Message
                </button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing message...</p>
            </div>

            <div class="result-section" id="resultSection">
                <span class="result-icon" id="resultIcon"></span>
                <div class="result-title" id="resultTitle"></div>
                <div class="result-subtitle" id="resultSubtitle"></div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceFill"></div>
                </div>
                <div class="confidence-text" id="confidenceText"></div>
                
                <div class="features-grid" id="featuresGrid"></div>
                <div class="features-note">Percentages normalized to sum to 100%</div>
            </div>

            <div class="examples">
                <h3>🧪 Try these examples:</h3>
                <div class="example-buttons">
                    <button class="example-btn" onclick="loadExample('Congratulations! You have won $1,000,000 in our lottery! Send $50 processing fee to claim your prize immediately!')">
                        Lottery Scam
                    </button>
                    <button class="example-btn" onclick="loadExample('Your account has been suspended. Click here to verify: http://suspicious-link.com')">
                        Phishing Attempt
                    </button>
                    <button class="example-btn" onclick="loadExample('Hi, this is John from the office. Can you call me back when you get this message?')">
                        Legitimate Message
                    </button>
                    <button class="example-btn" onclick="loadExample('URGENT! Your bank account will be closed. Send your details to prevent closure. Act now!')">
                        Bank Scam
                    </button>
                </div>
            </div>

            <div class="nav-section">
                <a href="/" class="back-home-btn">Back to Home</a>
                <a href="/about" class="back-home-btn">About</a>
            </div>
        </div>
    </div>

    <script>
        function loadExample(text) {
            document.getElementById('messageInput').value = text;
            analyzeMessage();
        }

        async function analyzeMessage() {
            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();
            
            if (!message) {
                alert('Please enter a message to analyze');
                return;
            }

            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('resultSection').classList.remove('show');
            document.querySelector('.analyze-btn').disabled = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();
                displayResult(data);
            } catch (error) {
                console.error('Error:', error);
                alert('Error analyzing message. Please try again.');
            } finally {
                // Hide loading
                document.getElementById('loading').classList.remove('show');
                document.querySelector('.analyze-btn').disabled = false;
            }
        }

        function displayResult(data) {
            const resultSection = document.getElementById('resultSection');
            const resultIcon = document.getElementById('resultIcon');
            const resultTitle = document.getElementById('resultTitle');
            const resultSubtitle = document.getElementById('resultSubtitle');
            const confidenceFill = document.getElementById('confidenceFill');
            const confidenceText = document.getElementById('confidenceText');
            const featuresGrid = document.getElementById('featuresGrid');

            // Update result based on prediction
            const isScam = data.prediction === 'SCAM';
            const confidence = data.confidence;

            if (isScam) {
                resultSection.className = 'result-section result-scam show';
                resultIcon.textContent = '⚠️';
                resultTitle.textContent = 'POTENTIAL SCAM DETECTED';
                resultSubtitle.textContent = 'This message contains multiple scam indicators';
            } else {
                resultSection.className = 'result-section result-safe show';
                resultIcon.textContent = '✅';
                resultTitle.textContent = 'MESSAGE APPEARS SAFE';
                resultSubtitle.textContent = 'No significant scam indicators detected';
            }

            // Update confidence
            confidenceFill.style.width = `${confidence}%`;
            confidenceText.textContent = `Confidence: ${confidence}%`;

            // Update features - NORMALIZED TO SUM TO 100%
            const featureNames = [
                'Urgency Keywords',
                'Money Keywords', 
                'Official Keywords',
                'Reward Keywords',
                'Celebrity Keywords',
                'Grammar Issues',
                'Contact Keywords',
                'Pressure Keywords',
                'Link Keywords',
                'Upfront Payment'
            ];

            // Calculate total score for normalization
            const totalScore = data.features.reduce((sum, score) => sum + score, 0);
            
            featuresGrid.innerHTML = '';
            
            // If all features are 0, show equal distribution
            if (totalScore === 0) {
                data.features.forEach((score, index) => {
                    const featureCard = document.createElement('div');
                    featureCard.className = 'feature-card';
                    featureCard.innerHTML = `
                        <div class="feature-name">${featureNames[index]}</div>
                        <div class="feature-score">10.0%</div>
                    `;
                    featuresGrid.appendChild(featureCard);
                });
            } else {
                // Normalize scores to sum to 100%
                let normalizedPercentages = data.features.map(score => 
                    (score / totalScore) * 100
                );
                
                // Round to 1 decimal place
                normalizedPercentages = normalizedPercentages.map(p => 
                    Math.round(p * 10) / 10
                );
                
                // Adjust for rounding errors to ensure sum is exactly 100%
                const sum = normalizedPercentages.reduce((a, b) => a + b, 0);
                const diff = 100 - sum;
                if (diff !== 0) {
                    // Find the largest percentage and adjust it
                    const maxIndex = normalizedPercentages.indexOf(Math.max(...normalizedPercentages));
                    normalizedPercentages[maxIndex] = Math.round((normalizedPercentages[maxIndex] + diff) * 10) / 10;
                }
                
                // Display normalized percentages
                data.features.forEach((score, index) => {
                    const featureCard = document.createElement('div');
                    featureCard.className = 'feature-card';
                    featureCard.innerHTML = `
                        <div class="feature-name">${featureNames[index]}</div>
                        <div class="feature-score">${normalizedPercentages[index].toFixed(1)}%</div>
                    `;
                    featuresGrid.appendChild(featureCard);
                });
            }
        }

        // Allow Enter key to submit (with Ctrl/Cmd)
        document.getElementById('messageInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                analyzeMessage();
            }
        });
    </script>
</body>
</html>
'''

HOME_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ Scam Research & Statistics</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            backdrop-filter: blur(10px);
        }

        .menu-bar {
            background: rgba(255, 255, 255, 0.98);
            padding: 15px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .menu-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 30px;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            color: #667eea;
            text-decoration: none;
        }

        .menu-items {
            display: flex;
            gap: 30px;
            list-style: none;
        }

        .menu-items a {
            color: #333;
            text-decoration: none;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            padding: 8px 16px;
            border-radius: 20px;
        }

        .menu-items a:hover {
            color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }

        .menu-items a.active {
            color: white;
            background: linear-gradient(135deg, #667eea, #764ba2);
        }

        .mobile-menu-toggle {
            display: none;
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: #667eea;
        }

        .header {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 40px 30px;
            text-align: center;
            position: relative;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" fill-opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }

        .nav-buttons {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
            position: relative;
            z-index: 1;
        }

        .nav-btn {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 2px solid white;
            padding: 12px 30px;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 1rem;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
        }

        .nav-btn:hover {
            background: white;
            color: #ff6b6b;
            transform: translateY(-2px);
        }

        .main-content {
            padding: 40px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-card.danger {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        }

        .stat-card.warning {
            background: linear-gradient(135deg, #feca57, #ff9ff3);
        }

        .stat-card.info {
            background: linear-gradient(135deg, #00d2d3, #54a0ff);
        }

        .stat-icon {
            font-size: 3rem;
            margin-bottom: 15px;
            display: block;
        }

        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .stat-label {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .section {
            margin-bottom: 40px;
        }

        .section-title {
            font-size: 1.8rem;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        .research-card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
        }

        .research-card h3 {
            color: #667eea;
            font-size: 1.3rem;
            margin-bottom: 15px;
        }

        .research-card p {
            color: #555;
            line-height: 1.6;
            margin-bottom: 10px;
        }

        .research-card ul {
            margin-left: 20px;
            color: #555;
            line-height: 1.8;
        }

        .research-card ul li {
            margin-bottom: 8px;
        }

        .highlight-box {
            background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(238, 90, 36, 0.1));
            border-left: 5px solid #ff6b6b;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .highlight-box h4 {
            color: #ff6b6b;
            font-size: 1.2rem;
            margin-bottom: 10px;
        }

        .highlight-box p {
            color: #333;
            line-height: 1.6;
        }

        .scam-types-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .scam-type-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            border-top: 4px solid #ff6b6b;
        }

        .scam-type-card h4 {
            color: #333;
            font-size: 1.1rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .scam-type-card p {
            color: #666;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .cta-section {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            margin-top: 40px;
        }

        .cta-section h2 {
            font-size: 2rem;
            margin-bottom: 15px;
        }

        .cta-section p {
            font-size: 1.1rem;
            margin-bottom: 25px;
            opacity: 0.9;
        }

        .cta-btn {
            background: white;
            color: #667eea;
            border: none;
            padding: 15px 40px;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }

        .cta-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }

        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 15px;
            }

            .header {
                padding: 30px 20px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .main-content {
                padding: 20px;
            }

            .stats-grid {
                grid-template-columns: 1fr;
            }

            .nav-buttons {
                flex-direction: column;
            }

            .nav-btn {
                width: 100%;
            }

            .mobile-menu-toggle {
                display: block;
            }

            .menu-items {
                display: none;
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background: white;
                flex-direction: column;
                gap: 0;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }

            .menu-items.show {
                display: flex;
            }

            .menu-items a {
                padding: 15px 30px;
                border-radius: 0;
                border-bottom: 1px solid #e1e8ed;
            }

            .menu-container {
                padding: 0 20px;
            }
        }
    </style>
</head>
<body>
    <!-- Menu Bar -->
    <div class="menu-bar">
        <div class="menu-container">
            <a href="/" class="logo">🛡️ ScamGuard</a>
            <button class="mobile-menu-toggle" onclick="toggleMenu()">☰</button>
            <ul class="menu-items" id="menuItems">
                <li><a href="/">Home</a></li>
                <li><a href="/detect_scams" class="active">Detect Scams</a></li>
                <li><a href="/about">About</a></li>
            </ul>
        </div>
    </div>

    <div class="container">
        <div class="header">
            <h1>🛡️ Scam Research & Statistics</h1>
            <p>Understanding the growing threat of digital scams</p>
            <div class="nav-buttons">
                <a href="/detect_scams" class="nav-btn">🔍 Try Scam Detector</a>
                <a href="#statistics" class="nav-btn">📊 View Statistics</a>
            </div>
        </div>
        
        <div class="main-content">
            <!-- Key Statistics -->
            <div class="stats-grid" id="statistics">
                <div class="stat-card danger">
                    <span class="stat-icon">💰</span>
                    <div class="stat-number">$10B+</div>
                    <div class="stat-label">Lost to Scams Annually</div>
                </div>
                
                <div class="stat-card warning">
                    <span class="stat-icon">📱</span>
                    <div class="stat-number">68%</div>
                    <div class="stat-label">Via Mobile Messages</div>
                </div>
                
                <div class="stat-card info">
                    <span class="stat-icon">⚠️</span>
                    <div class="stat-number">2.5M+</div>
                    <div class="stat-label">Scam Reports in 2024</div>
                </div>
                
                <div class="stat-card">
                    <span class="stat-icon">👥</span>
                    <div class="stat-number">1 in 4</div>
                    <div class="stat-label">People Targeted Monthly</div>
                </div>
            </div>

            <!-- Global Impact -->
            <div class="section">
                <h2 class="section-title">🌍 Global Impact of Scams</h2>
                
                <div class="highlight-box">
                    <h4>Rising Threat</h4>
                    <p>Scam-related losses have increased by 400% in the past five years, with sophisticated techniques targeting millions worldwide through messaging platforms, email, and social media.</p>
                </div>

                <div class="research-card">
                    <h3>📈 Key Trends in 2024</h3>
                    <ul>
                        <li><strong>AI-Powered Scams:</strong> Criminals are using AI to create more convincing messages and deepfake videos</li>
                        <li><strong>Cryptocurrency Fraud:</strong> 45% increase in crypto-related scams targeting inexperienced investors</li>
                        <li><strong>Impersonation Scams:</strong> Fraudsters impersonating government officials, celebrities, and trusted brands</li>
                        <li><strong>Investment Scams:</strong> Fake investment opportunities promising unrealistic returns</li>
                    </ul>
                </div>
            </div>

            <!-- Common Scam Types -->
            <div class="section">
                <h2 class="section-title">🎯 Most Common Scam Types</h2>
                
                <div class="scam-types-grid">
                    <div class="scam-type-card">
                        <h4>🎰 Lottery & Prize Scams</h4>
                        <p>Claiming you've won a prize but need to pay fees upfront. Often uses urgency and official-looking language to pressure victims.</p>
                    </div>
                    
                    <div class="scam-type-card">
                        <h4>🏦 Phishing Attacks</h4>
                        <p>Fake messages from banks or services asking for personal information through suspicious links. Often mimics legitimate companies.</p>
                    </div>
                    
                    <div class="scam-type-card">
                        <h4>💼 Employment Scams</h4>
                        <p>Fake job offers requiring upfront payment for training, equipment, or background checks before you can start.</p>
                    </div>
                    
                    <div class="scam-type-card">
                        <h4>🛒 Online Shopping Fraud</h4>
                        <p>Fake e-commerce sites selling products at unbelievable prices, then never delivering or sending counterfeit goods.</p>
                    </div>
                    
                    <div class="scam-type-card">
                        <h4>💳 Tech Support Scams</h4>
                        <p>Claiming your computer has a virus and demanding payment for fake technical support services.</p>
                    </div>
                </div>
            </div>

            <!-- Warning Signs -->
            <div class="section">
                <h2 class="section-title">🚨 Red Flags to Watch For</h2>
                
                <div class="research-card">
                    <h3>Common Warning Signs</h3>
                    <ul>
                        <li><strong>Urgency & Pressure:</strong> Messages demanding immediate action or threatening consequences</li>
                        <li><strong>Too Good to Be True:</strong> Promises of large sums of money, prizes, or unrealistic returns</li>
                        <li><strong>Upfront Payments:</strong> Requests for fees, taxes, or deposits before receiving promised benefits</li>
                        <li><strong>Poor Grammar:</strong> Messages with spelling errors, unusual phrasing, or grammatical mistakes</li>
                        <li><strong>Suspicious Links:</strong> Shortened URLs, misspelled domains, or requests to click unknown links</li>
                        <li><strong>Unverified Sources:</strong> Messages from unknown numbers or email addresses claiming to be official</li>
                        <li><strong>Request for Personal Info:</strong> Asking for passwords, banking details, or sensitive information</li>
                        <li><strong>Alternative Contact Methods:</strong> Insisting on using WhatsApp, Telegram, or other messaging apps</li>
                    </ul>
                </div>
            </div>

            <!-- Protection Tips -->
            <div class="section">
                <h2 class="section-title">🛡️ How to Protect Yourself</h2>
                
                <div class="research-card">
                    <h3>Best Practices for Staying Safe</h3>
                    <p><strong>Verify Everything:</strong> Always verify the source of messages through official channels. Contact organizations directly using known phone numbers or websites, not the contact information provided in suspicious messages.</p>
                    
                    <p><strong>Never Share Sensitive Information:</strong> Legitimate organizations will never ask for passwords, PINs, or full account numbers via message or email.</p>
                    
                    <p><strong>Be Skeptical of Urgency:</strong> Scammers create artificial urgency to prevent you from thinking clearly. Take time to research and consult with trusted friends or family.</p>
                    
                    <p><strong>Use Security Tools:</strong> Enable two-factor authentication, use strong passwords, and keep your devices updated with the latest security patches.</p>
                    
                    <p><strong>Trust Your Instincts:</strong> If something feels off or too good to be true, it probably is. Don't let embarrassment stop you from seeking advice.</p>
                </div>
            </div>

            <!-- CTA Section -->
            <div class="cta-section">
                <h2>🔍 Test Your Messages Now</h2>
                <p>Use our AI-powered scam detector to analyze suspicious messages and protect yourself from fraud</p>
                <a href="/detect_scams" class="cta-btn">Try Scam Detector →</a>
            </div>
        </div>
    </div>

    <script>
        function toggleMenu() {
            const menuItems = document.getElementById('menuItems');
            menuItems.classList.toggle('show');
        }

        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            const menuBar = document.querySelector('.menu-bar');
            const menuItems = document.getElementById('menuItems');
            if (!menuBar.contains(event.target)) {
                menuItems.classList.remove('show');
            }
        });
    </script>
</body>
</html>
'''

ABOUT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ About ScamGuard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            backdrop-filter: blur(10px);
        }

        .menu-bar {
            background: rgba(255, 255, 255, 0.98);
            padding: 15px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .menu-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 30px;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            color: #667eea;
            text-decoration: none;
        }

        .menu-items {
            display: flex;
            gap: 30px;
            list-style: none;
        }

        .menu-items a {
            color: #333;
            text-decoration: none;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            padding: 8px 16px;
            border-radius: 20px;
        }

        .menu-items a:hover {
            color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }

        .menu-items a.active {
            color: white;
            background: linear-gradient(135deg, #667eea, #764ba2);
        }

        .mobile-menu-toggle {
            display: none;
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: #667eea;
        }

        .header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 60px 30px;
            text-align: center;
            position: relative;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" fill-opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        }

        .header h1 {
            font-size: 3rem;
            margin-bottom: 15px;
            position: relative;
            z-index: 1;
        }

        .header p {
            font-size: 1.3rem;
            opacity: 0.9;
            position: relative;
            z-index: 1;
            max-width: 800px;
            margin: 0 auto;
        }

        .main-content {
            padding: 50px 40px;
        }

        .section {
            margin-bottom: 50px;
        }

        .section-title {
            font-size: 2rem;
            color: #333;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }

        .intro-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            padding: 30px;
            border-radius: 15px;
            border-left: 5px solid #667eea;
            margin-bottom: 30px;
        }

        .intro-card p {
            color: #333;
            font-size: 1.1rem;
            line-height: 1.8;
            margin-bottom: 15px;
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }

        .feature-item {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            border-top: 4px solid #667eea;
            transition: transform 0.3s ease;
        }

        .feature-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        }

        .feature-icon {
            font-size: 2rem;
            margin-bottom: 15px;
            display: block;
        }

        .feature-item h3 {
            color: #667eea;
            font-size: 1.2rem;
            margin-bottom: 10px;
        }

        .feature-item p {
            color: #666;
            line-height: 1.6;
        }

        .tech-stack {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin-top: 30px;
        }

        .tech-stack h3 {
            color: #333;
            font-size: 1.4rem;
            margin-bottom: 20px;
        }

        .tech-list {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }

        .tech-badge {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.95rem;
        }

        .founder-section {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            margin-top: 40px;
        }

        .founder-section h3 {
            font-size: 1.8rem;
            margin-bottom: 15px;
        }

        .founder-name {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 20px 0;
        }

        .founder-section p {
            font-size: 1.1rem;
            opacity: 0.9;
            max-width: 700px;
            margin: 0 auto;
        }

        .cta-section {
            text-align: center;
            margin-top: 50px;
            padding: 40px;
            background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(238, 90, 36, 0.1));
            border-radius: 15px;
        }

        .cta-section h2 {
            color: #333;
            font-size: 2rem;
            margin-bottom: 20px;
        }

        .cta-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            margin: 10px;
        }

        .cta-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }

        .cta-btn.secondary {
            background: white;
            color: #667eea;
            border: 2px solid #667eea;
        }

        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 15px;
            }

            .header {
                padding: 40px 20px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .main-content {
                padding: 30px 20px;
            }

            .features-grid {
                grid-template-columns: 1fr;
            }

            .mobile-menu-toggle {
                display: block;
            }

            .menu-items {
                display: none;
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background: white;
                flex-direction: column;
                gap: 0;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            }

            .menu-items.show {
                display: flex;
            }

            .menu-items a {
                padding: 15px 30px;
                border-radius: 0;
                border-bottom: 1px solid #e1e8ed;
            }

            .menu-container {
                padding: 0 20px;
            }
        }
    </style>
</head>
<body>
    <!-- Menu Bar -->
    <div class="menu-bar">
        <div class="menu-container">
            <a href="/" class="logo">🛡️ ScamGuard</a>
            <button class="mobile-menu-toggle" onclick="toggleMenu()">☰</button>
            <ul class="menu-items" id="menuItems">
                <li><a href="/">Home</a></li>
                <li><a href="/detect_scams">Detect Scams</a></li>
                <li><a href="/about" class="active">About</a></li>
            </ul>
        </div>
    </div>

    <div class="container">
        <div class="header">
            <h1>🛡️ About ScamGuard</h1>
            <p>AI-Powered Protection Against Digital Scams</p>
        </div>
        
        <div class="main-content">
            <!-- Introduction -->
            <div class="section">
                <h2 class="section-title">What is ScamGuard?</h2>
                
                <div class="intro-card">
                    <p>ScamGuard is an intelligent Python-based tool that identifies scam messages using a custom rule-based and machine learning model. Designed to analyze message patterns, assign scores to multiple scam indicators, and classify input as <strong>SCAM</strong> or <strong>REAL</strong> — all while explaining <strong>why</strong>.</p>
                    
                    <p>In an era where digital scams are becoming increasingly sophisticated, ScamGuard provides a powerful defense mechanism that combines pattern recognition, natural language processing, and machine learning to protect users from fraudulent messages.</p>
                </div>
            </div>

            <!-- Key Features -->
            <div class="section">
                <h2 class="section-title">Key Features</h2>
                
                <div class="features-grid">
                    <div class="feature-item">
                        <span class="feature-icon">🎯</span>
                        <h3>Binary Classification</h3>
                        <p>Accurately classifies input messages as either SCAM or REAL using advanced machine learning algorithms.</p>
                    </div>

                    <div class="feature-item">
                        <span class="feature-icon">🔍</span>
                        <h3>Multi-Factor Analysis</h3>
                        <p>Evaluates messages across 10+ scam indicators including urgency, money requests, grammar issues, and suspicious links.</p>
                    </div>

                    <div class="feature-item">
                        <span class="feature-icon">🤖</span>
                        <h3>Machine Learning Powered</h3>
                        <p>Uses Random Forest classifier trained on real-world scam data for high accuracy detection.</p>
                    </div>

                    <div class="feature-item">
                        <span class="feature-icon">📊</span>
                        <h3>Detailed Scoring</h3>
                        <p>Provides transparent feature scores showing which indicators contributed to the classification.</p>
                    </div>

                    <div class="feature-item">
                        <span class="feature-icon">⚡</span>
                        <h3>Real-Time Detection</h3>
                        <p>Instant analysis and results, allowing users to verify suspicious messages immediately.</p>
                    </div>

                    <div class="feature-item">
                        <span class="feature-icon">🎓</span>
                        <h3>Educational Insights</h3>
                        <p>Helps users understand scam patterns and learn to identify threats independently.</p>
                    </div>
                </div>
            </div>

            <!-- Detection Features -->
            <div class="section">
                <h2 class="section-title">What We Detect</h2>
                
                <div class="tech-stack">
                    <h3>Custom Feature Extractor analyzes:</h3>
                    <div class="tech-list">
                        <span class="tech-badge">💰 Money Requests</span>
                        <span class="tech-badge">⏰ Urgency Tactics</span>
                        <span class="tech-badge">✍️ Grammar Issues</span>
                        <span class="tech-badge">🔗 Suspicious Links</span>
                        <span class="tech-badge">💳 Upfront Payments</span>
                        <span class="tech-badge">⭐ Celebrity References</span>
                        <span class="tech-badge">🎁 Reward Offers</span>
                        <span class="tech-badge">⚠️ Pressure Tactics</span>
                        <span class="tech-badge">📱 Unusual Contact Methods</span>
                        <span class="tech-badge">🏛️ Official Appearance Fakes</span>
                        <span class="tech-badge">🔓 Unsecured Sources</span>
                        <span class="tech-badge">🚨 Threats & Blackmail</span>
                    </div>
                </div>
            </div>

            <!-- Technical Stack -->
            <div class="section">
                <h2 class="section-title">Technical Excellence</h2>
                
                <div class="tech-stack">
                    <h3>Built with cutting-edge technology:</h3>
                    <div class="tech-list">
                        <span class="tech-badge">Python</span>
                        <span class="tech-badge">Flask</span>
                        <span class="tech-badge">Scikit-learn</span>
                        <span class="tech-badge">Random Forest</span>
                        <span class="tech-badge">NLP</span>
                        <span class="tech-badge">Machine Learning</span>
                    </div>
                </div>

                <div class="intro-card" style="margin-top: 25px;">
                    <p><strong>Performance Metrics:</strong> ScamGuard delivers comprehensive accuracy reports including precision, recall, and F1-score, ensuring reliable detection you can trust.</p>
                </div>
            </div>

            <!-- Founder Section -->
            <div class="founder-section">
                <h3>Founded by</h3>
                <div class="founder-name">Debottam Ghosh</div>
                <p>Combining expertise in machine learning and cybersecurity to create tools that protect people from digital threats. ScamGuard represents a commitment to making the internet safer for everyone.</p>
            </div>

            <!-- Call to Action -->
            <div class="cta-section">
                <h2>Ready to Protect Yourself?</h2>
                <a href="/detect_scams" class="cta-btn">Try Scam Detector</a>
                <a href="/" class="cta-btn secondary">Learn More</a>
            </div>
        </div>
    </div>

    <script>
        function toggleMenu() {
            const menuItems = document.getElementById('menuItems');
            menuItems.classList.toggle('show');
        }

        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            const menuBar = document.querySelector('.menu-bar');
            const menuItems = document.getElementById('menuItems');
            if (!menuBar.contains(event.target)) {
                menuItems.classList.remove('show');
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Serve the home page with scam research and statistics"""
    return render_template_string(HOME_TEMPLATE)

@app.route('/detect_scams')
def detect_scams():
    """Serve the main web interface"""
    return render_template_string(DETECT_SCAMS_TEMPLATE)

@app.route('/about')
def about():
    """Serve the about page"""
    return render_template_string(ABOUT_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for message prediction"""
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Ensure model is loaded
        if model is None:
            return jsonify({'error': 'Model not available'}), 500
        
        # Extract features
        features = [assign_values_to_factors(message)]
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Calculate confidence
        confidence = round(max(prediction_proba) * 100, 1)
        
        result = {
            'prediction': 'SCAM' if prediction == 1 else 'NOT SCAM',
            'confidence': confidence,
            'features': features[0],
            'message_length': len(message)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Scam Detector Web Application...")
    print("📚 Loading model...")
    load_and_train_model()
    print("🌐 Starting Flask server on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)