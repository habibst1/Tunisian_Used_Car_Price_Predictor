# 🚗 Car Price Prediction

An end-to-end **machine learning application** that predicts the prices of used cars in **Tunisia**.  
It combines **web scraping, preprocessing, model training, and a Streamlit app** to deliver real-time predictions. 

---

## 💡 Use Case

This project can be integrated as a **feature for websites that sell used cars in Tunisia**.  
It allows users to estimate a fair selling or buying price based on the car’s brand, model, year, mileage, and other characteristics.  

- 🏢 **Car marketplaces** → provide transparent price estimates.  
- 👤 **Individual sellers** → set competitive prices for their listings.  
- 👥 **Buyers** → evaluate whether a listed price is reasonable.  

---

## ⚙️ How It Works
 
1. **Data Collection**  
   - Scrapes used car listings from [automobile.tn](https://www.automobile.tn/fr/occasion).  
   - Extracts fields: *Brand, Model, Kilométrage, Année, Boîte, Carburant, Puissance fiscale, Prix*.  

2. **Data Preprocessing**  
   - Splits **Brand** and **Model**.  
   - Cleans and normalizes categorical and numerical data.  
   - Removes outliers and saves a structured dataset.  

3. **Model Training**  
   - Trains multiple ML models (Linear Regression, KNN, Random Forest).  
   - **Random Forest** chosen for best accuracy.  
   - Saves trained model for deployment.  

4. **Prediction App (Streamlit)**  
   - User selects **brand and model**.  
   - Inputs car details: mileage, year, fiscal power, gearbox, fuel type.  
   - Model outputs **predicted price in TND**.  

---

## 🛠️ Tech Stack

- **Python**  
- **Web Scraping** → `requests`, `BeautifulSoup`  
- **Data Processing** → `pandas`, `numpy`  
- **Machine Learning** → `scikit-learn` (Random Forest)  
- **Visualization** → `matplotlib`, `seaborn`  
- **Deployment** → `Streamlit`  
- **Serialization** → `joblib`  

---

## 📸 Examples 

![image](https://github.com/user-attachments/assets/41da4fc2-fe5b-4a06-9784-37205a6fab0f)


---


