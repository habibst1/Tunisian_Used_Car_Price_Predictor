import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

brand_models = {
    "Alfa Romeo": ["GIULIETTA", "MITO", "GIULIA PACK PREMIUM"],
    "Audi": ["A5 COUPÉ", "A3", "A3 SPORTBACK", "Q5", "Q3 SPORTBACK", "Q8 S-LINE TDI", "Q2", "Q3 ADVANCED PLUS", "Q3 2.0 TDI", "A3 BERLINE", "Q3", "A5", "A4", "Q8", "A5 CABRIOLET", "A3 SPORTBACK S-LINE", "A5 SPORTBACK", "A6", "A4 BUSINESS PLUS", "A5 SPORTBACK S-LINE SELECTION", "Q3 AMBIENTE", "Q4 E-TRON", "A3 SPORTBACK SPORT", "Q7", "Q8 E-TRON", "TT", "A1 SPORTBACK", "A7 SPORTBACK QUATTRO V6", "A4 AMBITION LUXE", "S5", "A4 BUSINESS", "Q5 SPORTBACK S-LINE", "Q5 PRESTIGE PLUS", "Q5 FULL PACKAGE KHALIJIA", "A4 AMBIENTE", "A3 BERLINE BUSINESS", "A6 S-LINE"],
    "BAIC YX": ["KENBO S3"],
    "BMW": ["SÉRIE 4 GRAN COUPÉ", "X5", "SÉRIE 4 COUPÉ", "SÉRIE 2 GRAN COUPÉ PACK SPORT", "SÉRIE 3 CONFORT SPORT LINE", "SÉRIE 4 COUPÉ PACK SPORT M", "X4", "SÉRIE 3", "SÉRIE 1 PACK M", "SÉRIE 6", "SÉRIE 5", "SÉRIE 1", "SÉRIE 7", "SÉRIE 5 LUXURY LINE", "SÉRIE 2 GRAN COUPÉ", "X6", "X2 LOUNGE", "SÉRIE 5 BUSINESS LINE PLUS", "SÉRIE 4 GRAN COUPÉ PACK M SPORT", "SÉRIE 3 ACCESS PLUS LUXURY LINE", "SÉRIE 1 3P", "SÉRIE 2 ACTIVE TOURER", "SÉRIE 3 PACK SPORT", "SÉRIE 3 COUPÉ", "X3", "SÉRIE 1 SPORT LINE", "SÉRIE 1 ACCESS", "X1", "SÉRIE 4 COUPÉ SPORT LINE", "SÉRIE 3 LUXURY LINE", "SÉRIE 4 COUPÉ LUXURY LINE", "X3 PACK SPORT M LIMITED EDITION", "X1 1.5L", "X2", "SÉRIE 5 PACK SPORT M", "X2 PACK M PRO", "SÉRIE 2 ACTIVE TOURER BUSINESS LINE", "SÉRIE 1 LOUNGE", "SÉRIE 2 GRAN COUPÉ SPORT LINE", "SÉRIE 2 GRAN COUPÉ PACK SPORT M", "X3 XLINE", "X3 BUSINESS LINE", "SÉRIE 4 GRAN COUPÉ CONFORT LUXURY LINE", "SÉRIE 4 GRAN COUPÉ CONFORT SPORT LINE", "SÉRIE 3 316I", "SÉRIE 5 EXECUTIVE LUXURY LINE"],
    "BYD": ["F3 GLX-I", "SONG", "TANG SUPREME"],
    "Chery": ["TIGGO 7 PRO", "TIGGO 2 .", "TIGGO 8 PRO", "TIGGO 2", "TIGGO 3 CONFORT", "TIGGO 3X", "E3", "TIGGO 8", "TIGGO 4 PREMIUM", "QQ"],
    "Chevrolet": ["SUBURBAN", "CRUZE", "SONIC", "CAPTIVA 7 PLACES", "CAPTIVA 5 PLACES", "SONIC 4P 2021"],
    "Citroën": ["C3 SHINE", "BERLINGO BVA 5 PLACES", "BERLINGO VAN BVA TOUTE OPTIONS", "C3", "DS3", "BERLINGO", "C4 CACTUS", "C3 C-SÉRIES ÉDITION SPECIAL", "C4", "JUMPY FOURGON", "JUMPY COMBI", "C5", "SAXO", "C1", "DS4 SPORT CHIC", "BERLINGO MULTISPACE", "NEMO ATTRACTION", "C2", "JUMPY COMBI MULTISPACE", "C1 GRIS ANTHRACITE", "DS5"],
    "Cupra": ["LEON SPORT", "Cupra_LEON", "FORMENTOR"],
    "DFSK": ["GLORY 580"],
    "DS": ["7 CROSSBACK", "4 CROSSBACK", "DS_5"],
    "Dacia": ["DUSTER TECHROAD", "Dacia_DUSTER", "DOKKER VAN", "SANDERO STEPWAY", "SANDERO STEPWAY STEPWAY", "DOKKER VAN VAN LUX", "SANDERO", "LOGAN"],
    "Dodge": ["RAM RUMBLE BEE"],
    "Dongfeng": ["FORTHING T5 EVO", "SX3", "AX4", "CAPTAIN E"],
    "Fiat": ["500X", "PANDA CITY CROSS", "TIPO 5 PORTES", "TIPO BERLINE", "500 DOLCEVITA", "FIORINO", "500", "DUCATO", "PANDA", "DOBLO", "FIORINO MULIJET 2", "GRANDE PUNTO", "500 POP", "LINEA", "TIPO BERLINE EXCELLENTE FINITION", "TIPO BERLINE ENTRY", "LINEA EXCELLENTE", "PANDA CITY CROSS POP"],
    "Ford": ["ECOSPORT TITANIUM PLUS", "MUSTANG", "ECOSPORT", "RANGER", "MONDEO", "FOCUS", "KUGA", "FOCUS RS", "FUSION SE", "FIESTA TITANIUM PLUS", "FIESTA", "TRANSIT", "ECOSPORT TREND", "FIESTA TREND", "FUSION", "KA", "FOCUS TOUTE OPTION", "MONDEO TITANIUM", "ECOSPORT TITANIUM PLUS SANS ROUE", "PUMA", "FOCUS ECO BOOST", "KUGA TREND PLUS"],
    "Foton": ["TUNLAND SIMPLE CABINE"],
    "GMC": ["SIERRA AMERICAIN", "ACADIA AWD"],
    "Geely": ["GX3"],
    "Great Wall": ["M4", "M4 COMFORT"],
    "Haval": ["JOLION", "H6 LUXURY", "H9 VIN", "H6", "JOLION STANDARD", "H2"],
    "Honda": ["CR-V", "ACCORD", "CITY", "CIVIC", "HR-V EX", "CIVIC RS", "CIVIC TYPE R RS"],
    "Hyundai": ["GRAND I10", "I20", "TUCSON", "I20 HIGH GRADE", "CRETA", "KONA", "VELOSTER", "CRETA HIGH GRADE", "GRAND I10 HIGH GRADE", "ACCENT", "H-1", "IX35", "I10", "KONA N LINE", "GRAND I10 GLS", "TUCSON LIMITED", "GRAND I10 SEDAN", "TUCSON HTRAC 2.0L", "GALLOPER", "ACCENT GLS", "SANTA FE CRÉATIVE"],
    "Infiniti": ["FX"],
    "Isuzu": ["D-MAX SIMPLE CABINE", "D-MAX DOUBLE CABINE V-CROSS", "D-MAX DOUBLE CABINE GT", "D-MAX DOUBLE CABINE", "MU-X 2.5L", "NKR 66 DOUBLE CABINE"],
    "Iveco": ["DAILY"],
    "Jaguar": ["X-TYPE BVA", "F-PACE", "XE", "XF LUXURY", "XJL", "XF", "F-PACE R-DYNAMIC S", "XE PRESTIGE", "XF PRESTIGE", "XJ"],
    "Jeep": ["RENEGADE", "WRANGLER", "COMPASS", "GRAND CHEROKEE", "COMPASS NC", "WRANGLER RUBICON", "GRAND CHEROKEE LIMITED", "COMPASS LIMITED", "CHEROKEE LIMITED"],
    "KIA": ["RIO 5P", "RIO 5P GX", "SELTOS", "PICANTO", "SPORTAGE", "STONIC", "RIO BERLINE BVA", "RIO BERLINE", "SPORTAGE BLACK EDITION", "CERATO", "PICANTO SMART", "CEE'D", "RIO 5P SMART", "NIRO HYBRIDE", "XCEED SX", "SORENTO", "CARENS", "QUORIS", "RIO BERLINE LX", "OPTIMA GT LINE", "SPORTAGE FULL OPTION SAUF TOIT", "STONIC SX", "XCEED", "STONIC GT-LINE", "SORENTO L X", "SPORTAGE SPORT", "SPORTAGE AWD"],
    "Lada": ["4X4", "NIVA"],
    "Land Rover": ["RANGE ROVER EVOQUE", "RANGE ROVER SPORT", "DISCOVERY SPORT R-DYNAMIC S", "RANGE ROVER VELAR", "DISCOVERY SPORT HSE", "RANGE ROVER EVOQUE DYNAMIC SE", "DISCOVERY", "DEFENDER 110", "RANGE ROVER SUPERCHARGED", "FREELANDER", "RANGE ROVER SPORT HSE", "DISCOVERY SE", "DEFENDER 110 P400 X", "RANGE ROVER EVOQUE PURE", "RANGE ROVER VELAR R-DYNAMIC S"],
    "MG": ["HS", "ZS", "MG_6", "GS", "MG_5", "MG_3", "GT", "HS TROPHY", "ONE LUXURY+", "GT CONFORT PLUS", "GT LUXURY"],
    "Mahindra": ["KUV 100 K8", "PICK-UP SC", "KUV 100 K6+", "SCORPIO SUV"],
    "Mazda": ["Mazda_6", "Mazda_3", "BT-50", "CX-9", "CX-3", "CX-7", "3 PROPRE", "3 CORE+ GRADE", "2 SEDAN SÉLECTION PLUS", "CX-5", "3 SEDAN CORE+ GRADE"],
    "Mercedes-Benz": ["CLASSE C AMG", "GLC COUPÉ 400E 4MATIC AMG PREMIUM PLUS", "GLA AMG", "CLASSE C AMG +", "CLASSE C", "CLASSE E AVANTGARDE", "CLA AMG", "CLA 200 KIT AMG", "CLASSE E", "CLASSE E COUPÉ 250 CABRIOLET KIT AMG", "CLASSE A EXECUTIVE", "GLA 200 KIT AMG", "CLA PROGRESSIVE", "CLA 250E PACK AMG", "CLASSE A AMG", "CLASSE E COUPÉ AMG", "GLA", "CLASSE A BUSINESS", "CLASSE C AVANTGARDE", "CLASSE X", "CLA KIT AMG", "CLASSE C COUPÉ AMG", "GLC AMG", "GLB 200 D", "CLASSE B", "CLS AMG", "GLC COUPÉ AMG", "CLASSE E BUSINESS", "CLA", "CLASSE S", "CLASSE E AMG", "GLC", "CLASSE E COUPÉ", "CLASSE A", "GLE COUPÉ AMG", "ML", "CLS", "CLASSE A URBAN", "CLASSE G", "GLC COUPÉ", "EQA EDITION", "CLASSE GL AMG", "GLC DESIGNO", "CLA URBAN", "CLASSE C KITT AMG", "CLASSE C MERCEDES C180", "CLASSE E E300 KITT AMG LINE 4 MATIC", "CLASSE C C180 KITT AMG", "GLE", "CLASSE A BERLINE A 250 E KIT AMG", "CLA 250 E", "GLC COUPÉ GLC 300 E KIT AMG", "CLASSE A BERLINE AMG", "GLA PROGRESSIVE", "EQA AMG LINE PHASE 2 2024", "CLA PACK NIGHT", "CLASSE E ELEGANCE", "ML BVA", "CLASSE E E 200 D", "CLASSE E 220 CDI AMG", "CLASSE C PREMIUM", "ML AMG", "CLASSE C CLASSIC EDITION C", "CLASSE E 220D KIT AMG", "CLASSE A BERLINE", "GLK", "CLASSE C BUSINESS"],
    "Mini": ["3 PORTES", "JOHN COOPER WORKS", "5 PORTES", "COUNTRYMAN", "CLUBMAN", "3 PORTES INTACTE", "3 PORTES CLASSIC", "COOPER S COOPER", "COUPÉ COOPER", "5 PORTES PACK CHILI", "COOPER S"],
    "Mitsubishi": ["ASX 4WD", "PAJERO", "ATTRAGE GLS", "ATTRAGE", "L200 SPORTERO", "MIRAGE GLX"],
    "Nissan": ["NAVARA", "QASHQAI", "JUKE", "NAVARA DOUBLE CABINE", "JUKE NISMO-R", "PATROL", "MICRA", "TERRANO", "JUKE TEKNA PERSO"],
    "Opel": ["CORSA", "COMBO CARGO", "ASTRA EDITION PLUS", "ASTRA ELEGANCE", "MOKKA ELEGANCE", "CALIBRA TURBO 4X4", "GRANDLAND ÉLÉGANCE BUSINESS"],
    "Peugeot": ["PARTNER 650 KG", "308", "RIFTER", "2008 ALLURE", "2008", "3008", "E-208 GT", "3008 ALLURE", "408", "208", "3008 GT-LINE", "3008 GT", "PARTNER", "508", "PARTNER 1000 KG", "206 LOOK", "301 ALLURE VISIO", "PARTNER ORIGIN UTILITAIRE", "LANDTREK SIMPLE CABINE", "508 ALLURE", "LANDTREK SIMPLE CABINE BENNE TÔLE PLIÉE", "2008 ALLURE BLANC NACRÉ + TOIT", "208 ALLURE", "EXPERT", "4008 LUXE", "206+", "301 ALLURE", "LANDTREK DOUBLE CABINE U", "307", "208 VRAIE GT", "206+ POPULAIRE", "208 PACK EDITION", "3008 PREMIUM BLANC NACRÉ", "504", "207", "2008 ACTIVE PM + TOIT", "2008 ACTIVE PACK", "2008 ALLURE BLANC NACRÉ", "208 ACTIVE", "4008", "307 PHASE 2", "206 BERLINE", "LANDTREK DOUBLE CABINE", "308 ALLURE", "BOXER", "5008", "RIFTER GT LINE", "407", "3008 IMPECCABLE"],
    "Porsche": ["CAYENNE COUPÉ", "CAYENNE", "MACAN", "CAYENNE V6", "PANAMERA", "PANAMERA S", "PANAMERA PLATINUM EDITION", "CAYENNE S DIESEL", "CAYENNE PLATINIUM ÉDITION"],
    "Renault": ["CLIO LIFE PLUS", "LAGUNA COUPÉ", "CLIO 4", "KADJAR", "KADJAR INTENS", "CLIO", "TWINGO", "MEGANE SEDAN", "CLIO DYNAMIQUE", "MEGANE", "EXPRESS", "MEGANE DYNAMIQUE", "CLIO ZEN", "CAPTUR ZEN", "ARKANA INTENSE", "AUSTRAL", "VELSATIS EXPRESSION", "GRAND ESPACE", "CLIO INTENS", "SYMBOL", "CAPTUR", "EXPRESS VAN", "SYMBOL CONFORT", "KWID POPULAIRE", "FLUENCE DYNAMIQUE", "CLIO EDITION", "Renault_DUSTER"],
    "Seat": ["IBIZA", "Seat_LEON", "ARONA", "IBIZA SOL", "IBIZA POPULAIRE", "IBIZA FR PLUS", "ATECA STYLE", "IBIZA SC", "LEON 1.2", "ATECA FR"],
    "Skoda": ["OCTAVIA CONFORT", "KUSHAQ AMBITION", "OCTAVIA", "SUPERB AMBASSADOR", "KAMIQ STYLE", "FABIA", "RAPID", "RAPID STYLE", "SCALA", "OCTAVIA ESSENCE"],
    "Smart": ["FORTWO", "FORFOUR"],
    "Ssangyong": ["ACTYON SPORTS 4*4", "ACTYON SPORTS", "TIVOLI", "TIVOLI CLASSY", "KORANDO", "TIVOLI CLASSIC MY2020", "TIVOLI CLASSY SECURE", "MUSSO LUXE", "MUSSO CRM", "TIVOLI CONFORT MY2020", "KORANDO LIMITED", "KORANDO CLASSIQUE"],
    "Suzuki": ["BALENO", "VITARA 4*4", "CELERIO", "SWIFT", "JIMNY 3 PORTES", "VITARA", "GRAND VITARA", "VITARA GLX SR", "JIMNY 3 PORTES GL", "CIAZ", "S-PRESSO GL", "SWIFT GLX PACK", "S-PRESSO GL PACK ALU"],
    "Toyota": ["AGYA COMFORT", "LAND CRUISER 79 SIMPLE CABINE", "HILUX DOUBLE CABINE DOUBLE CABINE BUSINESS", "AGYA BVA", "COROLLA", "C-HR", "HILUX DOUBLE CABINE", "AGYA", "YARIS MÉTALLISÉ", "YARIS", "PRADO", "LAND CRUISER", "HILUX SIMPLE CABINE", "RAV 4 HYBRIDE SUV", "PRADO BUSINESS", "RAV 4", "FORTUNER 7 PLACES", "HILUX DOUBLE CABINE DOUBLE CABINE PREMIUM"],
    "Volkswagen": ["POLO", "GOLF 8", "CADDY", "PASSAT R-LINE", "T-ROC IMPORTÉE TN250", "PASSAT GTE", "PASSAT", "GOLF 7 R-LINE", "AMAROK DOUBLE CABINE", "GOLF 8 R-LINE", "GOLF 7 JOIN", "T-ROC", "POLO R-LINE", "TIGUAN", "GOLF 7", "GOLF 7 SMARTLINE", "AMAROK", "T-ROC CONFORTLINE", "T-CROSS", "GOLF 6", "GOLF 7 GTD", "JETTA", "GOLF 7 TOIT PANORAMIQUE", "GOLF 7 ALLSTAR", "CRAFTER", "TOUAREG", "GOLF 7 SOUND", "TIGUAN UNITED", "POLO PRESTIGE", "GOLF 7 À", "GOLF 6 STYLE", "T-ROC R-LINE", "TIGUAN PRESTIGE", "TIGUAN R-LINE", "TIGUAN LIFE", "GOLF 7 EXCELLENTE ÉTAT", "GOLF 7 TRENDLINE", "GOLF 5", "SCIROCCO", "PASSAT B6", "GOLF 7 EXCELLENT ÉTAT D’ORIGINE", "JETTA TRENDLINE", "GOLF 7 MÉTALLISÉ", "PASSAT SMARTLINE", "POLO CONFORTLINE"],
    "Volvo": ["XC60 R", "S80", "S60", "XC60 R-DESIGN", "S80 CUIR BOIS NORDIQUE", "XC60", "XC90", "XC60 ULTIMATE DARK"],
    "Wallyscar": ["719 ATMOS", "IRIS"],
    "ZXAUTO": ["GRAND TIGER DOUBLE CABINE"]
}


# App title
st.title("Car Price Prediction")

# --- Brand selection ---
brand = st.selectbox("Select Brand", sorted(brand_models.keys()), key="brand_select")

# --- Model selection tied to brand ---
if "model_selected" not in st.session_state:
    st.session_state.model_selected = ""

available_models = brand_models.get(brand, [])
model = st.selectbox("Select Model", sorted(available_models), key="model_select")

# --- Input form ---
with st.form(key="car_form"):
    kilometrage = st.number_input("Kilométrage (km)", min_value=0, step=1000, value=50000)
    annee = st.number_input("Année", min_value=1980, max_value=2025, step=1, value=2018)
    puissance_fiscale = st.number_input("Puissance fiscale (CV)", min_value=1, step=1, value=6)
    boite = st.selectbox("Boîte", ["Automatique", "Manuelle"])
    carburant = st.selectbox("Carburant", ["Essence", "Diesel","Hybride"])
    
    submit_button = st.form_submit_button("Predict Price")

# --- Load model ---
try:
    model_rf = joblib.load("RandomForest")
except FileNotFoundError:
    st.error("Model file 'RandomForest.joblib' not found. Please ensure it is in the correct directory.")
    st.stop()

# --- Predict price ---

if submit_button:
    if model not in available_models:
        st.error("Please select a valid brand and model.")
    else:
        input_data = {
            "Brand": brand,
            "Model": model,
            "Kilométrage": kilometrage,
            "Année": annee,
            "Puissance fiscale": puissance_fiscale,
            "Boîte": boite,
            "Carburant": carburant
        }
        
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, drop_first=False)

        # Load the column names from X_columns.txt
        with open('X_columns.txt', 'r', encoding='utf-8') as file:
            expected_columns = [line.strip() for line in file]

        # Align the columns of input_df with the expected columns
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)

        try:
            prediction = model_rf.predict(input_df)
            predicted_price = prediction[0]
            predicted_price = np.exp(predicted_price)
            
            # Enhanced price display using Streamlit's markdown with custom styling
            st.markdown(
                """
                <div style='text-align: center; padding: 20px; background-color: #f0f8ff; 
                           border-radius: 10px; margin: 20px 0;'>
                    <h2 style='color: #1e90ff; font-size: 28px; margin-bottom: 10px;'>
                        Predicted Price
                    </h2>
                    <p style='font-size: 48px; font-weight: bold; color: #006400; 
                             margin: 0;'>
                        {:,.2f} TND
                    </p>
                </div>
                """.format(predicted_price),
                unsafe_allow_html=True
            )
            
            # Additional info below the main display
            st.info(f"Prediction for {brand} {model} ({annee}) with {kilometrage:,} km")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")