## Chilli Leaf Disease Detection

# 🌿 Chilli Leaf Disease Detection Using Deep Learning

This project is a web-based application that detects diseases in chilli plant leaves using a deep learning model. It processes uploaded leaf images, identifies the disease (if any), and suggests organic pesticide treatments. It also stores the results in a MongoDB database for tracking and analysis.

![Screenshot 2025-03-05 092323](https://github.com/user-attachments/assets/92867f03-e911-4505-b195-11f5bc52ccc4)


## 🚀 Features

- Upload an image of a chilli plant leaf and detect its disease.
- Supports 5 classification categories.
- Provides organic pesticide recommendations based on the diagnosis.
- Built-in image preprocessing with OpenCV.
- Stores results with metadata in MongoDB.
- Simple web interface using Flask.

## 🧠 Model Information

- **Model Type**: Convolutional Neural Network (CNN)
- **Input Size**: 256x256 RGB images
- **Framework**: TensorFlow / Keras
- **Classes**:
  - Whitefly
  - Yellowish
  - Healthy
  - Leaf Curl
  - Leaf Spot

The model is trained on a labelled chilli leaf image dataset from Kaggle and saved as `clm.h5`.

## 🗃️ Dataset

- **Source**: [Kaggle](https://www.kaggle.com/) (please insert the actual link if you have it)
- **Classes**: 5
- **Format**: Image files sorted by disease category

## 🧪 Pesticide Recommendation System

The app recommends organic pesticide treatments for detected diseases:
- **Leaf Curl**: Neem oil, chili-garlic spray, proper watering.
- **Leaf Spot**: Baking soda solution, copper fungicide, neem oil.
- **Whitefly**: Neem oil, garlic spray, yellow sticky traps, insecticidal soap.
- **Yellowish**: Compost tea, Epsom salt, seaweed extract.
- **Healthy**: No treatment required.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow/Keras
- **Image Processing**: OpenCV
- **Database**: MongoDB
- **Frontend**: HTML (via `index.html` template)

## 📦 Installation

### Prerequisites

- Python 3.7+
- MongoDB
- `pip install -r requirements.txt` (dependencies below)

### Required Libraries
flask
tensorflow
opencv-python
numpy
pymongo
werkzeug

Clone & Run

git clone https://github.com/your-username/chilli-leaf-disease-detection.git
cd chilli-leaf-disease-detection
pip install -r requirements.txt
python app.py
MongoDB must be running locally at mongodb://localhost:27017/. Update the URI in the script if needed.

How to Use
Launch the Flask app with python app.py
Open a browser and navigate to http://localhost:5000
Upload a chilli plant leaf image
The app will display:
Predicted disease
Confidence score
Pesticide recommendation

Sample Output (Example)
{
  "prediksi": "Leaf Curl",
  "confidence": 0.9852,
  "gambar_prediksi": "./static/images/uploads/leaf1.jpg",
  "pesticide_suggestion": "Neem Oil, Chili-Garlic Spray, Proper Watering..."
}

🗂 Folder Structure
├── app.py
├── clm.h5
├── static/
│   └── images/uploads/
├── templates/
│   └── index.html
├── requirements.txt

👤 Contributor
[N.Teja Kumr Reddy]

📈 Future Improvements
Add user authentication
Enable cloud storage integration
Add more leaf disease classes
Mobile app version

📝 License
This project is licensed under the MIT License. Feel free to use, modify, and share with attribution.

💡 Acknowledgements
Dataset from Kaggle
TensorFlow for model development
Flask for web integration

