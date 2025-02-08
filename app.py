from flask import Flask, request, render_template, redirect, flash
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Layer
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import matplotlib.pyplot as plt
import numpy as np
import base64


app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Define path to save uploaded CSV
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.score_dense = Dense(1, activation="tanh")

    def call(self, inputs):
        score = self.score_dense(inputs)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector



# Load your pre-trained model (path to your model file)
model1 = load_model('model/sahamjet1.h5', custom_objects={'AttentionLayer': AttentionLayer})
model = load_model('model/sahamjet2.h5')


def createMultiStepDataset(features, target, lookBack, steps_ahead):
    X, y = [], []
    for i in range(len(features) - lookBack - steps_ahead):
        X.append(features[i:(i + lookBack)])
        y.append(target[(i + lookBack):(i + lookBack + steps_ahead)])
    return np.array(X), np.array(y)

def predict_next_7_days(model, data):
    # Normalisasi data menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Ambil 30 hari terakhir untuk prediksi
    last_30_days = scaled_data[-30:]
    input_data = np.array(last_30_days).reshape(1, 30, -1)  # (1, lookback, features)

    # Prediksi dengan model
    predictions = model.predict(input_data)

    # Invers transformasi hasil prediksi
    predictions = scaler.inverse_transform(predictions)
    return predictions[0]  # Output (7,)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect('/upload')

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect('/upload')

        if file and file.filename.endswith('.csv'):
            # Simpan file di folder 'uploads'
            filepath = f'uploads/{file.filename}'
            file.save(filepath)
            flash('File uploaded successfully')

            # Proses file CSV
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            df['sentiment'] = df['sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
            df['compound_score'] = df['compound_score'].apply(
                lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else x
            )

            # Menyiapkan fitur dan target
            features = df.drop(['Date', 'close'], axis=1).values
            target = df['close'].values

            # Normalisasi data
            scaler_features = MinMaxScaler(feature_range=(0, 1))
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            features_scaled = scaler_features.fit_transform(features)
            target_scaled = scaler_target.fit_transform(target.reshape(-1, 1))

            # Parameter untuk multi-step dataset
            look_back = 30
            steps_ahead = 7
            X, y = createMultiStepDataset(features_scaled, target_scaled, look_back, steps_ahead)

            # Pembagian data training dan testing
            train_size = int(len(X) * 0.8)
            train_X, test_X = X[:train_size], X[train_size:]
            train_y, test_y = y[:train_size], y[train_size:]

            # Reshape data untuk LSTM
            train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], train_X.shape[2]))
            test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], test_X.shape[2]))

            # Model LSTM dengan Attention
            dropout = 0.2
            inputs = Input(shape=(look_back, features.shape[1]))

            lstm_out = LSTM(50, return_sequences=True)(inputs)
            lstm_out = LSTM(50, return_sequences=True)(lstm_out)
            lstm_out = Dropout(dropout)(lstm_out)

            attention_output = AttentionLayer()(lstm_out)

            dense_out = Dense(25, activation="relu")(attention_output)
            dense_out = Dropout(dropout)(dense_out)
            output = Dense(steps_ahead)(dense_out)

            model = Model(inputs=inputs, outputs=output)
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
            model.summary()

            # Training model
            history = model.fit(
                train_X, train_y,
                validation_data=(test_X, test_y),
                epochs=20,
                batch_size=32
            )

            # Plot Loss
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            os.makedirs('static/charts', exist_ok=True)
            loss_chart_path = 'static/charts/loss_chart1.png'
            plt.savefig(loss_chart_path)

            # Plot MAE
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['mae'], label='Train MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.xlabel('Epochs')
            plt.ylabel('Mean Absolute Error')
            plt.legend()
            mae_chart_path = 'static/charts/mae_chart1.png'
            plt.savefig(mae_chart_path)

            # Prediksi selama 7 hari ke depan
            predictions = model.predict(test_X[:1])
            predictions = scaler_target.inverse_transform(predictions)
            predicted_values = predictions[0]

            prediction_chart_path = 'static/charts/prediction_chart1.png'
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, 8), predicted_values, marker='o', label='Predicted Prices')
            plt.title('Predicted Prices for the Next 7 Days')
            plt.xlabel('Days Ahead')
            plt.ylabel('Price')
            plt.legend()
            plt.savefig(prediction_chart_path)

            flash('Model trained successfully. Charts generated.')
            return render_template(
                'hasilTrain.html',
                loss_chart=loss_chart_path,
                mae_chart=mae_chart_path,
                prediction_chart=prediction_chart_path
            )
        else:
            flash('Only CSV files are allowed!')
            return redirect('/upload')
    return render_template('upload.html')


@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        # Periksa keberadaan file
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # Periksa apakah ada nama file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Simpan file jika ada
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                # Baca file CSV
                df = pd.read_csv(filepath)
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
                df['sentiment'] = df['sentiment'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})
                df['compound_score'] = df['compound_score'].apply(
                    lambda x: float(str(x).replace(',', '.')) if isinstance(x, str) else x
                )

                # Siapkan fitur dan target (harga close)
                features = df.drop(['Date', 'close'], axis=1).values
                close_prices = df['close'].values

                scaler_features = MinMaxScaler(feature_range=(0, 1))
                scaler_target = MinMaxScaler(feature_range=(0, 1))
                features_scaled = scaler_features.fit_transform(features)
                target_scaled = scaler_target.fit_transform(close_prices.reshape(-1, 1))

                # Prediksi 7 hari ke depan
                if len(features_scaled) < 30:
                    flash('Data tidak mencukupi untuk prediksi (minimal 30 baris).')
                    return redirect(request.url)

                input_data = features_scaled[-30:].reshape(1, 30, features_scaled.shape[1])
                predictions = model.predict(input_data)
                predictions = scaler_target.inverse_transform(predictions).flatten()

                # Gabungkan data historis dengan prediksi
                all_close_prices = np.concatenate([close_prices, predictions])
                all_days = list(range(1, len(all_close_prices) + 1))

                # Plot grafik
                plt.figure(figsize=(12, 6))
                plt.plot(all_days, all_close_prices, label='Harga Historis dan Prediksi', marker='o', color='blue')
                plt.axvline(x=len(close_prices), color='gray', linestyle='--', label='Awal Prediksi')
                plt.title('Harga Saham dan Prediksi (7 Hari)')
                plt.xlabel('Hari ke-')
                plt.ylabel('Harga Saham')
                plt.legend()
                plt.grid()

                # Simpan chart sebagai gambar
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                chart_url = base64.b64encode(img.getvalue()).decode()
                plt.close()

                predictions_data = [{'day': i + 1, 'predicted_price': pred} for i, pred in enumerate(predictions)]
                flash('Prediction completed and chart generated!')
                return render_template('hasilTest.html', chart_url=chart_url, predictions=predictions_data)

            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)

    return render_template('test.html')



if __name__ == '__main__':
    app.run(debug=True)
