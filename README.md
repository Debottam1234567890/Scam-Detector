# AI-Powered Scam Message Detector

An intelligent Python-based tool that identifies scam messages using a custom rule-based and machine learning model. Designed to analyze message patterns, assign scores to multiple scam indicators, and classify input as **SCAM** or **REAL** — all while explaining **why**.

## Features

- Classifies input messages as **SCAM** or **REAL**
- Uses a custom-built feature extractor for:
  - Money requests
  - Urgency
  - Grammar issues
  - Suspicious links
  - Upfront payments
  - Celebrity references
  - Reward offers
  - Pressure tactics
  - Unusual contact methods
  - Official appearance fakes
  - Unsecured sources
  - Threats or blackmail
- Machine Learning classifier (Random Forest)
- Accuracy report with precision, recall, and F1-score

## Dataset Structure

Your dataset file `labeled_dataset.csv` should look like this:

```
message,label,urgency,money_request,official_appearance,reward_offer,celebrity_reference,grammar_issues,unusual_contact_method,pressure_to_act,suspicious_link,upfront_payment
"Message text",scam,0.9,1.0,0.7,0.6,0.0,0.4,0.2,0.9,0.0,0.5
```

## How to Run

### 1. Clone this repository

```bash
git clone https://github.com/Debottam1234567890/Scam_Detector.git
cd scam_detector
```

### 2. Set up your Python environment

Make sure you're using Python 3.12 or higher.

```
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
pip install -r requirements.txt
```

If there's no `requirements.txt`, install dependencies manually:

```
pip install scikit-learn pandas
```

### 3. Run the program

```
python3 scam_detector.py
```

You will see a classification report and a prompt for entering messages.

## Sample Usage (CLI)

```
Enter a message to check (or 'quit' to exit):
Congratulations! You've won $5,000! Claim now or lose it forever.

Prediction: SCAM

Enter a message to check (or 'quit' to exit):
```

## Sample Usage (GUI)
```
python3 endpoints.py
```

You will see a website with the same scam detector

## How It Works

Each message is processed by a feature extraction system that searches for 100+ scammy keywords across 12 different psychological and linguistic factors. These values are then used by a Random Forest model to classify the message.

## License

This project is licensed under the MIT License — feel free to use, modify, and share.

## Author

Debottam Ghosh  
GitHub: [@Debottam1234567890](https://github.com/Debottam1234567890)

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/new-keywords`
3. Commit your changes: `git commit -m 'Add new keywords'`
4. Push to the branch: `git push origin feature/new-keywords`
5. Open a pull request

## 📫 Contact

For questions or suggestions, feel free to open an issue or contact via GitHub.
