print("Starting app...")

from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model
model = joblib.load("model/random_forest_model.joblib")  # Replace with your trained model path

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    preview_html = None

    if request.method == 'POST':
        # Check if CSV upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            # Predict
            try:
                df['predicted_price'] = model.predict(df)
            except Exception as e:
                return f"Error in prediction: {e}", 500

            # Preview first 5 rows
            preview_html = df.head().to_html(classes='preview-table', index=False)

            # Save CSV to memory for download
            output = BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype='text/csv',
                download_name='predicted_' + file.filename,
                as_attachment=True
            )

        # Check if manual input form
        elif request.form:
            try:
                # Extract form data into a DataFrame
                input_data = {col: [request.form[col]] for col in request.form}
                df_input = pd.DataFrame(input_data)
                
                prediction_result = model.predict(df_input)[0]
            except Exception as e:
                return f"Error in prediction: {e}", 500

    return render_template('index.html', prediction=prediction_result, preview=preview_html)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)

