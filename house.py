import pickle
import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

house_model = pickle.load(open('house.sav', 'rb'))

st.title('PREDIKSI HARGA RUMAH')

test_inputs = np.array([
    [1360,2,1,1981,0.5996366396268326,0,5],
    [1269,2,2,2006,3.5550397628502823,1,9]
])
test_actuals = np.array([5, 6])  


Square_Footage = st.text_input('Luas Persegi')
Num_Bedrooms = st.text_input('Jumlah Kamar')
Num_Bathrooms= st.text_input('Jumlah Kamar mandi')
Year_Built = st.text_input('Tahun Pembangunan')
Lot_Size = st.text_input('Luas Halaman')
Garage_Size = st.text_input('Luas Garasi')
Neighborhood_Quality = st.text_input('Kualitas Tetangga')

House_Price = ''
mae, mse, rmse = '', '', ''

if st.button('Predict Price'):
    
    try:
        input_data = np.array([[float(Square_Footage), float(Num_Bedrooms), float(Num_Bathrooms),
                                float(Year_Built), float(Lot_Size), float(Garage_Size),
                                float(Neighborhood_Quality)]])
        
        prediction = house_model.predict(input_data)
        
        House_Price = f'Prediksi Harga-nya adalah: {prediction[0]}'
        
        test_predictions = house_model.predict(test_inputs)
        
        mae_value = mean_absolute_error(test_actuals, test_predictions)
        mae = f'MAE (Mean Absolute Error): {mae_value:.2f}'
        
        mse_value = mean_squared_error(test_actuals, test_predictions)
        mse = f'MSE (Mean Squared Error): {mse_value:.2f}'
        
        rmse_value = np.sqrt(mse_value)
        rmse = f'RMSE (Root Mean Squared Error): {rmse_value:.2f}'
        
    except ValueError:
        House_Price = "Tolong masukkan data yang benar."


st.success(House_Price)

if mae and mse and rmse:
    st.write(mae)
    st.write(mse)
    st.write(rmse)
