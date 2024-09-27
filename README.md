# Language-Translation-System
Multilingual Content Scraping and Translation: The Language Translation System effectively scrapes textual data from specified web URLs, processing and translating this content into the desired target language, facilitating accessibility for diverse users.  Sentiment Analysis Implementation
!pip install deep-translator textblob transformers torch scikit-learn
import time
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

def moving_welcome_message():
    message = "WELCOME TO TRANSLATION SYSTEM OF WEB SCRAPED DATA"
    for i in range(len(message) + 1):
        print("\r" + message[:i], end="")
        time.sleep(0.1)
    print("\n")

def scrape_multilingual_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = [para.get_text() for para in paragraphs]

        return content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []

def translate_content(content, target_language='en'):
    translated_content = []
    translator = GoogleTranslator(source='auto', target=target_language)

    for text in content:
        try:
            translation = translator.translate(text)
            translated_content.append(translation)
        except Exception as e:
            print(f"Error translating text: {e}")
            translated_content.append(text)

    return translated_content

def preprocess_text(text):
    processed_text = text.lower()
    return processed_text

def train_logistic_regression_model(texts, labels):
    vectorizer = TfidfVectorizer(max_features=5000, preprocessor=preprocess_text)
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return best_model, vectorizer

def predict_sentiment(model, vectorizer, texts):
    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    return predictions

def main():
    moving_welcome_message()

    url = input("Kindly enter the URL to scrape: ")
    target_language = input("Enter the target language code (e.g., 'en' for English, 'hi' for Hindi): ")

    print("\nScraping content from website...")
    content = scrape_multilingual_content(url)

    if content:
        print("\nTranslating content...")
        translated_content = translate_content(content, target_language)

        print("\nTraining sentiment analysis model...")
        train_texts = [
            "I love this product, it is amazing!",
            "This is the worst experience I've ever had.",
            "I am very happy with the service.",
            "I am not satisfied with the quality.",
            "The movie was fantastic, I enjoyed every bit of it.",
            "The book was boring and too long.",
            "I am thrilled with the new features in the update.",
            "The product broke after a week, very disappointed.",
            "Customer support was helpful and resolved my issue quickly.",
            "The food was tasteless and overpriced.",
            "I had a great time at the concert, the band was amazing.",
            "The software is full of bugs and crashes frequently.",
            "Delivery was prompt and the item was as described.",
            "The hotel room was dirty and the service was terrible.",
            "I love the way this app works, it's very intuitive.",
            "Terrible customer service, I waited for hours.",
            "Excellent quality and fast shipping, very satisfied.",
            "Not worth the money, very cheap materials.",
            "Great ambiance and friendly staff, highly recommend.",
            "The movie plot was predictable and dull.",
            "Very comfortable and well-made shoes.",
            "Awful experience, the staff was very rude.",
            "The cake was delicious and beautifully decorated.",
            "The game has too many ads and glitches.",
            "Highly recommended, exceeded my expectations.",
            "The package arrived damaged and late.",
            "Amazing storyline and superb acting.",
            "Worst meal I've had in a long time.",
            "The park is clean and well-maintained, perfect for a family day out.",
            "The battery life of this phone is fantastic.",
            "Very poor build quality, it broke after a few uses.",
            "The scenery was breathtaking, I loved the hike.",
            "The flight was delayed and uncomfortable.",
            "The course was informative and well-structured.",
            "The app constantly crashes, very frustrating.",
            "A delightful experience, will definitely come back.",
            "The noise level was unbearable, couldn't sleep.",
            "Fast and reliable internet connection, very happy.",
            "The instructions were unclear and confusing.",
            "The performance was outstanding, a must-see.",
            "Extremely disappointed, not as described.",
            "I had an awesome experience, will recommend to others."
        ]
        train_labels = [
            "Positive", "Negative", "Positive", "Negative",
            "Positive", "Negative", "Positive", "Negative",
            "Positive", "Negative", "Positive", "Negative",
            "Positive", "Negative", "Positive", "Negative",
            "Positive", "Negative", "Positive", "Negative",
            "Positive", "Negative", "Positive", "Negative",
            "Positive", "Negative", "Positive", "Negative",
            "Positive", "Negative", "Positive", "Negative",
            "Positive", "Negative", "Positive", "Negative",
            "Positive", "Negative", "Positive", "Negative",
            "Positive", "Negative"
        ]

        model, vectorizer = train_logistic_regression_model(train_texts, train_labels)

        print("\nPredicting sentiment of the translated content...")
        predicted_sentiments = predict_sentiment(model, vectorizer, translated_content)

        sentiment_mapping = {
            "Positive": "Positive",
            "Negative": "Negative"
        }

        for i in range(len(predicted_sentiments)):
            if predicted_sentiments[i] not in sentiment_mapping:
                predicted_sentiments[i] = "Neutral"

        print("\nTranslated Content and Predicted Sentiments:")
        for original_text, translated_text, sentiment in zip(content, translated_content, predicted_sentiments):
            print(f"Original: {original_text}")
            print(f"Translated: {translated_text}")
            print(f"Sentiment: {sentiment_mapping.get(sentiment, 'Neutral')}\n")

    else:
        print("No content to translate.")

if __name__ == "__main__":
    main()
