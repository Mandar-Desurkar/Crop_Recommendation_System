from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the crop recommendation model
crop_recommendation_model_path = 'models/NB.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

app = Flask(__name__)

# Define the crop image URLs and background colors
crop_details = {
    'rice': {'image_url': 'static/images/rice.jpg', 'background_color': '#EAE3DB'},
    'maize': {'image_url': 'static/images/maize.jpg', 'background_color': '#BDC092'},
    'chickpea': {'image_url': 'static/images/chickpea.avif', 'background_color': '#E6C9A9'},
    'kidneybeans': {'image_url': 'static/images/kidneybeans.jpg', 'background_color': '#E7CDD0'},
    'pigeonpeas': {'image_url': 'static/images/pigeonpea.jpg', 'background_color': '#D7CEB3'},
    'mothbeans': {'image_url': 'static/images/mothbeans.jpg', 'background_color': '#E2BB98'},
    'mungbean': {'image_url': 'static/images/mungbeans.jpg', 'background_color': '#BBBB85'},
    'blackgram': {'image_url': 'static/images/blackgram.jpg', 'background_color': '#DDDE7'},
    'lentil': {'image_url': 'static/images/lentil.jpg', 'background_color': '#CFB7A6'},
    'pomegranate': {'image_url': 'static/images/pomegranate.jpg', 'background_color': '#DDDE7'},
    'banana': {'image_url': 'static/images/banana.jpg', 'background_color': '#E8CCAA'},
    'mango': {'image_url': 'static/images/mango.jpg', 'background_color': '#E4EDB6'},
    'grapes': {'image_url': 'static/images/grapes.jpg', 'background_color': '#E9E8A1'},
    'watermelon': {'image_url': 'static/images/watermelon.jpg', 'background_color': '#C1D5B9'},
    'muskmelon': {'image_url': 'static/images/muskmelon.jpg', 'background_color': '#FFE7C9'},
    'apple': {'image_url': 'static/images/apple.jpg', 'background_color': '#D9E1AE'},
    'orange': {'image_url': 'static/images/orange.jpg', 'background_color': '#F8B44A'},
    'papaya': {'image_url': 'static/images/papaya.jpg', 'background_color': '#FFC7B1'},
    'coconut': {'image_url': 'static/images/coconut.jpg', 'background_color': '#EDF29F'},
    'cotton': {'image_url': 'static/images/cotton.jpg', 'background_color': '#EAE3DB'},
    'jute': {'image_url': 'static/images/jute.jpg', 'background_color': '#FFD5A6'},
    'coffee': {'image_url': 'static/images/coffee.jpg', 'background_color': '#D4B39A'}
}

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/crop-prediction')
def crop_prediction():
    return render_template('crop prediction.html')

@app.route('/result', methods=['GET', 'POST'])
def crop_result():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorus'])
        K = int(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph-level'])
        rainfall = float(request.form['rainfall'])

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = crop_recommendation_model.predict(data)
        final_prediction = prediction[0]

        crop_info = crop_details.get(final_prediction, {'image_url': 'static/images/default.jpg', 'background_color': '#ffffff'})
        image_url = crop_info['image_url']
        background_color = crop_info['background_color']

        return render_template('result.html', crop=final_prediction, image_url=image_url, background_color=background_color)
    else:
        return render_template('result.html', crop=None, image_url='', background_color='')

if __name__ == '__main__':
    app.run(debug=True)
