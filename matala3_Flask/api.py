from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from car_data_prep import prepare_data  

app = Flask(__name__)

# Load the model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

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

    
    # Prepare the data using the imported prepare_data function
    processed_data = prepare_data(input_data)

   # def complete_missing_columns(new_df, df_sample):
        # Find missing columns
    #    missing_columns = set(new_df.columns) - set(df_sample.columns)
        
        # Add missing columns to the test data with values of 0
    #    for col in missing_columns:
     #       df_sample[col] = 0
        
        # Reorder columns in test data to match training data
      #  df_sample = df_sample.reindex(columns=new_df.columns)
        
      #  return df_sample
    
    for col in model_columns:
        if col not in processed_data.columns:
            processed_data[col] = 0
    
    processed_data = processed_data.reindex(columns=model_columns)


    #processed_data = complete_missing_columns(new_df, processed_data)
    X_test = processed_data
    #X_test = processed_data.drop(['Price'],axis=1)
    #y_test = processed_data['Price']
    # Make prediction
    prediction = model.predict(X_test)

    # Format the output
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Predicted Price: {output}')

if __name__ == "__main__":
    app.run(debug=True)