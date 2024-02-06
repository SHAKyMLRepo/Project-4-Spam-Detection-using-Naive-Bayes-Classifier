from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

basepath = os.path.abspath(".")

# Load model
with open(basepath + '/static/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Route for main page
@app.route('/', methods=['POST', 'GET'])
def index():

    # If post
    if request.method == 'POST':
        text = request.form['text']
        predicted_label = request.form['prediction']
        predicted_label = model.predict([text])
        if(predicted_label == 1):
            predicted_label = 'Prediction is SPAM'
        elif(predicted_label == 0):
            predicted_label = 'Prediction is not SPAM'
        print("POST", predicted_label)
        return render_template('index.html', prediction=predicted_label, text=text)

    else:
        print("GET")
        return render_template('index.html', prediction='', input_text='')

if __name__ == '__main__':
    app.run(debug=True)