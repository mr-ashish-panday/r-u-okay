from flask import Flask, request, render_template
import numpy as np
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__, template_folder='template')

# Load the prediction model (replace with your model loading logic)
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# def load_chunks():
#     with open('data.pkl', 'rb') as pickle_file:
#         while True:
#             try:
#                 chunk = pickle.load(pickle_file)
#                 chunk.append(chunk)
#             except EOFError:
#                 break
#     return chunk

# chunks = load_chunks()  # Load chunks once for efficiency

@app.route('/')
def index():
    return render_template('index.html')

sia=SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    return sia.polarity_scores(text)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST': 
        q1 = request.form['q1']
        q2 = request.form['q2']
        q3 = request.form['q3']
        q4 = request.form['q4']

        vader_scores = [get_vader_sentiment(q1)["compound"], 
                        get_vader_sentiment(q2)["compound"], 
                        get_vader_sentiment(q3)["compound"], 
                        get_vader_sentiment(q4)["compound"]]

        # Convert to numpy array for model input
        # user_input = np.array([vader_scores])  # Assuming model expects a 2D array
        # return {"test" : user_input.tolist()}
        # Model prediction (replace with your actual model's predict method)
        result1 = model.predict(np.array([vader_scores[0]]).reshape(-1, 1))  
        result2 = model.predict(np.array([vader_scores[1]]).reshape(-1, 1))
        result3 = model.predict(np.array([vader_scores[2]]).reshape(-1, 1))
        result4 = model.predict(np.array([vader_scores[3]]).reshape(-1, 1))
        # result2 = model.predict([user_input[1]])  
        # result3 = model.predict([user_input[2]])  
        # result4 = model.predict([user_input[3]])  
        results_list = [result1.tolist(), result2.tolist(), result3.tolist(), result4.tolist()]
        frequency = {0: 0, 1: 0, 2: 0, 3: 0}
        pass_list = []   # Loop through the nested list and count occurrences
        for sublist in results_list:
            for num in sublist:
                if num in frequency:
                    frequency[num] += 1
        for key,value in frequency.items():
              x = round(((value/4.05)*100),2)
              pass_list.append(x)
        # Pass the results to the HTML template
        return render_template('results.html',pass_list = pass_list)
        # results_list = [result1.tolist(), result2.tolist(), result3.tolist(), result4.tolist()]

        # Return the results as a JSON response
        scores = {
            'anxiety': result1.tolist(),
            'depression': result2.tolist(),
            'suicidal': result3.tolist(),
            'normal': result4.tolist(),
        }
        return render_template('results.html', scores=scores)

if __name__ == '__main__':
    app.run(debug=True)