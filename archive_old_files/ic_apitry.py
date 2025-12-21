# Description: try out function classifier
# to be updated and seen
import joblib
import pandas as pd

# Load the model
clf = joblib.load('interaction_classifier.pkl')

# If you used a scaler, load it as well
# scaler = joblib.load('scaler.pkl')

def predict_interaction(new_data):
    """
    Predicts if there's an interaction based on new data.
    :param new_data: DataFrame with the new input data.
    :return: Prediction results.
    """
    # Preprocess the new data as needed
    # If you used a scaler, apply it here
    # new_data = scaler.transform(new_data)
    
    # Predict using the loaded model
    predictions = clf.predict(new_data)
    return predictions

# Example usage:
new_data = pd.DataFrame({
    'evaluation_species_identified': [1],
    'evaluation_interaction_identified': [1]
})

predictions = predict_interaction(new_data)
print(predictions)


def interactive_prediction():
    """
    Allows user to input data for prediction and returns the result.
    """
    species_identified = int(input("Enter species identification (1 if both species identified, 0 otherwise): "))
    interaction_identified = int(input("Enter interaction identification (1 if interaction is identified, 0 otherwise): "))
    
    # Prepare the data
    new_data = pd.DataFrame({
        'evaluation_species_identified': [species_identified],
        'evaluation_interaction_identified': [interaction_identified]
    })
    
    # Predict using the model
    predictions = predict_interaction(new_data)
    print(f"Prediction: {'Interaction' if predictions[0] == 1 else 'No Interaction'}")

# Example usage:
#interactive_prediction()


from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

# Load the trained model
clf = joblib.load('interaction_classifier.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    species_identified = int(request.form['species_identification'])
    interaction_identified = int(request.form['interaction_identification'])

    # Create a DataFrame for prediction
    new_data = pd.DataFrame({
        'evaluation_species_identified': [species_identified],
        'evaluation_interaction_identified': [interaction_identified]
    })

    # Predict interaction
    prediction = clf.predict(new_data)[0]
    prediction_text = "Interaction" if prediction == 1 else "No Interaction"

    return render_template('result.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)


"""
# Test with known data
test_data = pd.DataFrame({
    'species_identified': [1, 0, 1],
    'interaction_identified': [1, 0, 0]
})

test_predictions = predict_interaction(test_data)
print(test_predictions)
"""