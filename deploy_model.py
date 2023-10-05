###############################
# This program lets you       #
# - Create a dashboard        #
# - Evevry dashboard page is  #
# created in a separate file  #
###############################

# Python libraries
import streamlit as st
from PIL import Image
import sklearn

# User module files
from ml import ml
from nn_adam import nn_adam
from nn_sgd import nn_sgd


def main():

    #############
    # Main page #
    #############

    options = ['Home','Logistic Regression Predictions', 'Artificial Neural Network Adam Predictions','Aritifical Neural Network SGD Predictions']
    choice = st.sidebar.selectbox("Menu",options, key = '1')

    if ( choice == 'Home' ):
      st.title("Hey VC lazy guy, here again? Let's figure out if that company worths an investment")
      st.image('./images/vc_meme.jpeg')
      st.text("Disclaimer: this is a trial version 1.0.0 with Crunchbase public data up to 2016. It's not recommended to invest only guiding in the model predictions")
      pass

    elif ( choice == 'Logistic Regression Predictions' ):
      ml()
    
    elif (choice == 'Artificial Neural Network Adam Predictions'):
      nn_adam()

    elif (choice == 'Aritifical Neural Network SGD Predictions'):
      nn_sgd()

    else:
      st.stop()


main()
