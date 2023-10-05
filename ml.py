# Import python libraries
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


def ml():

    st.image('images/logistic_regression.jpg')
    st.title("Predictions with Linear Regression Model")

    # loading the model
    models_path = 'models/'
    model_name = models_path + 'LRM_model.pkl'
    loaded_model = pickle.load(open(model_name, 'rb'))

    # loading the scaler
    transformers_path = 'transformers/'
    transformer_name = transformers_path + 'standard_scaler.pkl'
    loaded_transformer = pickle.load(open(transformer_name, 'rb'))

    # loading the encoder
    encoders_path = 'encoders/'
    encoder_name = encoders_path + 'one_hot_encoder.pkl'
    loaded_encoder = pickle.load(open(encoder_name, 'rb'))

    # Lists of accptable values
    valid_country_codes = ['IND', 'USA', 'Other', 'CHN', 'HKG', 'CAN', 'CHL', 'GBR', 'FRA',
       'AUS', 'ROM', 'KOR', 'NLD', 'DNK', 'NOR', 'COL', 'ESP', 'BEL',
       'IRL', 'ITA', 'SWE', 'SGP', 'RUS', 'NZL', 'CHE', 'BRA', 'SVN',
       'JOR', 'HUN', 'JPN', 'DEU', 'NGA', 'ISR', 'FIN', 'CRI', 'IDN',
       'PRT', 'ARG', 'TWN', 'THA', 'UKR', 'LTU', 'ISL', 'MEX', 'TUR',
       'URY', 'AUT', 'ZAF', 'PHL', 'MYS', 'PER', 'POL', 'VNM', 'UGA',
       'HRV', 'EST', 'LBN', 'BGR', 'SVK', 'LUX', 'CZE', 'ARE', 'SAU',
       'PAK', 'LVA', 'GHA', 'TAN', 'PRI', 'GRC', 'BLR', 'BMU', 'LIE',
       'SLV', 'GEO', 'GTM']
    valid_industry = ['Social Media', 'Other', 'Web', 'Analytics', 'Internet',
       'Search Engine', 'B2C', 'Services', 'Management', 'Social']

    # Get input values.
    funding_amount = st.number_input("Please enter the total funding the company already has in USD: ", 0, 1000000000000)
    country_code = st.selectbox("Please enter the country code: ", valid_country_codes, key = "3")
    funding_rounds = st.number_input ("Please enter the number of funding rounds: ", 0, 100)
    year = st.number_input("Please enter the year of the last funding: ", 2000, 2024 )
    lapse_months = st.number_input("Please enter the number of months between the first funding round and the last one: ", 0, 276)
    industry_cluster = st.selectbox("Please select the industry: ",valid_industry, key = '2')
    

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Get Your Prediction"):

        X = pd.DataFrame({'funding_total_usd':[funding_amount],
                      'country_code':[country_code],
                      'funding_rounds':[funding_rounds],
                      'last_funding_at':[year],
                      'lapse_months':[lapse_months],
                      'cluster_industry':[industry_cluster]
                     })

        numerical = X.select_dtypes(include = np.number)
        categorical = X.select_dtypes(include = object)

        column_names = ['country_code_ARE', 'country_code_ARG', 'country_code_AUS',
       'country_code_AUT', 'country_code_BEL', 'country_code_BGR',
       'country_code_BLM', 'country_code_BLR', 'country_code_BMU',
       'country_code_BRA', 'country_code_CAN', 'country_code_CHE',
       'country_code_CHL', 'country_code_CHN', 'country_code_COL',
       'country_code_CRI', 'country_code_CZE', 'country_code_DEU',
       'country_code_DNK', 'country_code_ESP', 'country_code_EST',
       'country_code_FIN', 'country_code_FRA', 'country_code_GBR',
       'country_code_GEO', 'country_code_GHA', 'country_code_GRC',
       'country_code_GTM', 'country_code_HKG', 'country_code_HRV',
       'country_code_HUN', 'country_code_IDN', 'country_code_IND',
       'country_code_IRL', 'country_code_ISL', 'country_code_ISR',
       'country_code_ITA', 'country_code_JOR', 'country_code_JPN',
       'country_code_KOR', 'country_code_LBN', 'country_code_LIE',
       'country_code_LTU', 'country_code_LUX', 'country_code_LVA',
       'country_code_MAF', 'country_code_MEX', 'country_code_MYS',
       'country_code_NGA', 'country_code_NLD', 'country_code_NOR',
       'country_code_NZL', 'country_code_Other', 'country_code_PAK',
       'country_code_PER', 'country_code_PHL', 'country_code_POL',
       'country_code_PRI', 'country_code_PRT', 'country_code_ROM',
       'country_code_RUS', 'country_code_SAU', 'country_code_SGP',
       'country_code_SLV', 'country_code_SOM', 'country_code_SVK',
       'country_code_SVN', 'country_code_SWE', 'country_code_SYC',
       'country_code_TAN', 'country_code_THA', 'country_code_TUR',
       'country_code_TWN', 'country_code_UGA', 'country_code_UKR',
       'country_code_URY', 'country_code_USA', 'country_code_VNM',
       'country_code_ZAF', 'cluster_industry_Analytics',
       'cluster_industry_B2C', 'cluster_industry_Internet',
       'cluster_industry_Management', 'cluster_industry_Other',
       'cluster_industry_Search Engine', 'cluster_industry_Services',
       'cluster_industry_Social', 'cluster_industry_Social Media',
       'cluster_industry_Web']
        cat_transformed = loaded_encoder.transform(categorical).toarray()
        categorical = pd.DataFrame(cat_transformed)
        categorical.columns = column_names
        

        # Joning dataframes
        X = pd.concat([numerical, categorical], axis=1)
        # Scaling data
        X_scaled = loaded_transformer.transform(X)

        # Making predictions
        prediction = loaded_model.predict(X_scaled)
        prediction_probs = loaded_model.predict_proba(X_scaled)


        if prediction == 1:
            st.success("The model predicts a status of 'Yes' with a probability of {:.2f}".format(prediction_probs[0, 1]))
        else:
            st.error("The model predicts a status of 'No' with a probability of {:.2f}".format(prediction_probs[0, 0]))
