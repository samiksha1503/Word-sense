from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    model = joblib.load('logistic_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    message = request.form['message']
    message_vectorized = vectorizer.transform([message])
    
    prediction = model.predict(message_vectorized)[0]
    result = 'Spam' if prediction == 1 else 'Not Spam'
    
    return render_template('result.html', message=message, result=result)

@app.route('/predict_hate_speech', methods=['POST'])
def predict_hate_speech():
    model = joblib.load('hate_speech_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer1.pkl')
    
    message = request.form['message']
    message_vectorized = vectorizer.transform([message])
    
    prediction = model.predict(message_vectorized)[0]
    result = 'Hate Speech' if prediction == 1 else 'Not Hate Speech'
    
    return render_template('result.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
