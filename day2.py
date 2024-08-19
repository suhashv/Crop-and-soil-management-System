import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

# Define advanced functions for soil analysis, crop recommendation, and disease prediction

def analyze_soil(nitrogen, phosphorus, potassium, pH, moisture):
    """
    Advanced soil analysis function that provides recommendations based on soil nutrient levels, pH, and moisture.
    """
    recommendations = []

    # Nitrogen analysis
    if nitrogen < 50:
        recommendations.append("Low Nitrogen: Consider adding organic compost or nitrogen-rich fertilizers.")
    elif nitrogen > 200:
        recommendations.append("High Nitrogen: Monitor plant growth for excessive foliage.")

    # Phosphorus analysis
    if phosphorus < 20:
        recommendations.append("Low Phosphorus: Consider adding bone meal or rock phosphate.")
    elif phosphorus > 100:
        recommendations.append("High Phosphorus: Excessive phosphorus may lock out other nutrients.")

    # Potassium analysis
    if potassium < 150:
        recommendations.append("Low Potassium: Use potash fertilizers like wood ash or greensand.")
    elif potassium > 400:
        recommendations.append("High Potassium: May affect the uptake of magnesium and calcium.")

    # pH analysis
    if pH < 5.5:
        recommendations.append("Acidic soil: Consider adding lime to raise pH.")
    elif pH > 7.5:
        recommendations.append("Alkaline soil: Add sulfur or organic matter to lower pH.")

    # Moisture analysis
    if moisture < 30:
        recommendations.append("Low moisture: Ensure regular watering or use mulch to retain moisture.")
    elif moisture > 70:
        recommendations.append("High moisture: Improve drainage to prevent root rot.")

    return recommendations

def recommend_crop(soil_type, climate, season):
    """
    Advanced crop recommendation based on soil type, climate, and season.
    """
    crop_data = {
        'Loamy': {'Warm': {'Summer': ['Tomatoes', 'Corn'], 'Winter': ['Wheat', 'Barley']},
                  'Cool': {'Summer': ['Lettuce', 'Peas'], 'Winter': ['Broccoli', 'Cabbage']}},
        'Clay': {'Warm': {'Summer': ['Rice', 'Soybeans'], 'Winter': ['Carrots', 'Beets']},
                 'Cool': {'Summer': ['Spinach', 'Kale'], 'Winter': ['Garlic', 'Onions']}},
        'Sandy': {'Warm': {'Summer': ['Melons', 'Peppers'], 'Winter': ['Radishes', 'Turnips']},
                  'Cool': {'Summer': ['Potatoes', 'Zucchini'], 'Winter': ['Leeks', 'Brussels Sprouts']}}
    }

    return crop_data.get(soil_type, {}).get(climate, {}).get(season, ["No suitable crops found."])

def fetch_weather_data(location):
    """
    Fetch weather data for a given location using the OpenWeatherMap API.
    """
    api_key = '5742764dfb2717909f87b765437e231e'  # Replace with your API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def identify_disease(image_path, model_path='plant_disease_model.h5'):
    """
    Identify crop disease using a pre-trained deep learning model.
    """
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    return [(label, round(prob * 100, 2)) for _, label, prob in decoded_predictions]

# Console-based interaction
def main():
    print("Welcome to the Advanced Crop and Soil Management System")
    
    # Input soil data
    nitrogen = float(input("Enter Nitrogen content (ppm): "))
    phosphorus = float(input("Enter Phosphorus content (ppm): "))
    potassium = float(input("Enter Potassium content (ppm): "))
    pH = float(input("Enter Soil pH: "))
    moisture = float(input("Enter Soil Moisture content (%): "))
    
    # Analyze soil
    soil_analysis = analyze_soil(nitrogen, phosphorus, potassium, pH, moisture)
    print("\nSoil Analysis Results:")
    for recommendation in soil_analysis:
        print(f"- {recommendation}")

    # Input crop recommendation parameters
    soil_type = input("\nEnter Soil Type (Loamy/Clay/Sandy): ")
    climate = input("Enter Climate (Warm/Cool): ")
    season = input("Enter Season (Summer/Winter): ")

    # Recommend crops
    crop_recommendations = recommend_crop(soil_type, climate, season)
    print("\nRecommended Crops for Your Soil and Climate:")
    for crop in crop_recommendations:
        print(f"- {crop}")

    # Fetch weather data
    location = input("\nEnter your location for weather forecast: ")
    weather_data = fetch_weather_data(location)
    if weather_data:
        print(f"\nWeather Forecast for {location}:")
        print(f"- Temperature: {weather_data['main']['temp']}°C")
        print(f"- Weather: {weather_data['weather'][0]['description']}")
        print(f"- Humidity: {weather_data['main']['humidity']}%")
    else:
        print("Failed to fetch weather data. Please check your location or API key.")

    # Identify disease
    disease_check = input("\nDo you want to identify a crop disease from an image? (yes/no): ")
    if disease_check.lower() == 'yes':
        image_path = input("Enter the path to the crop image: ")
        disease_predictions = identify_disease(image_path)
        print("\nDisease Identification Results:")
        for disease, probability in disease_predictions:
            print(f"- {disease}: {probability}%")

if __name__ == "__main__":
    main()
