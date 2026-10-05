# Fake News Detection Platform

A web-based **Fake News Detection Platform** built using Python, Flask, Natural Language Processing (NLP), and Machine Learning.

The application analyzes a news article or claim and predicts whether it is **REAL NEWS** or **FAKE NEWS** based on patterns learned from a labeled training dataset. It also uses the **NewsAPI** service to retrieve related news articles and display live news resources.

> **Disclaimer:** This project is an educational machine-learning application. Its prediction is based on patterns learned from the training dataset and should not be treated as a definitive fact-checking system.

---

## Features

- 📰 Fake news classification using Machine Learning
- 🤖 Logistic Regression classification model
- 🔤 TF-IDF-based text feature extraction
- 📊 Prediction confidence score
- 🌐 Live related-news search using NewsAPI
- 📰 Latest news sidebar
- ⚡ Flask-based web interface
- 🔐 API key stored securely using environment variables
- 📱 Responsive web interface

---

## Tech Stack

### Backend
- Python
- Flask
- Scikit-learn
- Pandas
- Requests
- Python-dotenv

### Machine Learning
- TF-IDF Vectorization
- Logistic Regression
- Supervised text classification

### Frontend
- HTML
- CSS
- Jinja2 Templates

### External API
- NewsAPI

---

## Project Structure

```text
fake-news-detector/
│
├── app.py                 # Flask web application
├── train.py               # Model training script
├── dataset.csv            # Training dataset
├── model.pkl              # Trained Logistic Regression model
├── vectorizer.pkl         # Trained TF-IDF vectorizer
├── requirements.txt       # Python dependencies
├── .env                   # API credentials (not uploaded to GitHub)
├── .gitignore             # Git ignored files
│
├── static/
│   └── style.css          # Application styling
│
└── templates/
    └── index.html         # Main web interface
```

---

## How It Works

The application follows these main steps:

```text
User enters news
       ↓
Text preprocessing
       ↓
TF-IDF Vectorization
       ↓
Logistic Regression Model
       ↓
REAL NEWS / FAKE NEWS
       ↓
Confidence Score
       ↓
NewsAPI searches for related articles
       ↓
Related sources displayed
```

### 1. Text Vectorization

The training text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The vectorizer is configured with English stop-word removal and a maximum vocabulary of 5,000 features.

### 2. Machine Learning Model

A **Logistic Regression** classifier is trained on the TF-IDF features.

The trained model predicts:

- `1` → REAL NEWS
- `0` → FAKE NEWS

The model also provides a probability-based confidence score for the prediction.

### 3. Live News Verification Resources

After making a prediction, the application extracts keywords from the user's input and searches **NewsAPI** for related articles.

If matching articles are available, they are displayed as related resources.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## NewsAPI Configuration

The application uses NewsAPI to retrieve related and latest news articles.

Create a `.env` file in the project root:

```env
NEWS_API_KEY=your_newsapi_key_here
```

Do **not** commit the `.env` file to GitHub.

The repository's `.gitignore` excludes it from version control.

---

## Training the Model

The project already contains `model.pkl` and `vectorizer.pkl`.

If you want to retrain the model using `dataset.csv`, run:

```bash
python train.py
```

This will:

1. Load `dataset.csv`
2. Remove rows containing missing text or labels
3. Convert the text into TF-IDF features
4. Train a Logistic Regression classifier
5. Calculate training-set accuracy
6. Save the trained model as `model.pkl`
7. Save the TF-IDF vectorizer as `vectorizer.pkl`

---

## Running the Application

Start the Flask application:

```bash
python app.py
```

The terminal will provide the local server address.

Open the application in your browser, typically at:

```text
http://127.0.0.1:5000
```

Enter a news article or claim and click **CHECK NEWS**.

---

## Requirements

The project uses the following Python packages:

```text
Flask==3.0.0
pandas==2.1.1
scikit-learn==1.3.1
requests==2.31.0
python-dotenv==1.0.0
```

---

## Limitations

This project has several important limitations:

- The prediction depends heavily on the quality and distribution of the training dataset.
- A high model confidence does **not** mean that a claim is factually true.
- The model identifies patterns in text rather than independently establishing factual truth.
- NewsAPI availability depends on the API key, internet connection, and API limits.
- Related articles retrieved from NewsAPI are presented as resources and are not automatically treated as proof that the original claim is true.
- The current training script reports accuracy on the same dataset used for training, so this value should **not** be interpreted as an unbiased estimate of real-world performance.

---

## Future Improvements

Possible improvements include:

- Train/test dataset splitting
- Cross-validation and additional evaluation metrics
- Precision, recall, F1-score, and confusion matrix
- Larger and more diverse datasets
- Advanced NLP models such as BERT or other transformer-based models
- Improved claim verification using multiple independent sources
- Better keyword extraction
- Source credibility analysis
- User authentication and prediction history
- Deployment using a cloud platform

---

## Disclaimer

This project is developed for **educational and demonstration purposes**.

The prediction generated by the machine-learning model should not be considered a definitive determination of whether a news article or claim is true or false. Users should verify important information using reliable and independent sources.

---

## Author

**Sundaram Pandey**

B.Tech — Computer Science & Engineering

GitHub: `https://github.com/pandey-sundaram`

LinkedIn: `https://www.linkedin.com/in/sundarampandeyy/`