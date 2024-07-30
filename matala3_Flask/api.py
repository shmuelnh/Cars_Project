from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from car_data_prep import prepare_data  

app = Flask(__name__)

# Load the model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)
#יצרנו קובץ pkl של מספר העמודות של המודל 
# מכיוון שהמודל צריך לקבל את מספר העמודות המדוייק שהוא התאמן עליו
with open('model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [x for x in request.form.values()]
    
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([features], columns=[
        'manufactor', 'Year', 'model', 'Hand', 'Gear', 'capacity_Engine',
        'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Area', 'City',
        'Pic_num','Color',
        'Km','Cre_date','Repub_date','Description','Test','Supply_score'
    ])
    #כיוון שהמשתמש לא מזין מחיר נוסיף את העמודה הזו כדי שפונקציית הכנת הנתונים תרוץ כראוי
    input_data['Price'] = 0
    
    # Prepare the data using the imported prepare_data function
    processed_data = prepare_data(input_data)

    # נרצה למלא את העמודות החסרות כדי שהמודל יקבל את מספר העמודות הנדרש
    for col in model_columns:
        if col not in processed_data.columns:
            processed_data[col] = 0
    
    processed_data = processed_data.reindex(columns=model_columns)


    X_test = processed_data
   
    # Make prediction
    prediction = model.predict(X_test)

    # Format the output
    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text=f'המחיר החזוי הוא: {output} ש"ח')

if __name__ == "__main__":
    app.run(debug=True)

