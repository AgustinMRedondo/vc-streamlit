###############################
# This program lets you       #
# - Create a dashboard        #
# - Evevry dashboard page is  #
# created in a separate file  #
###############################

# Python libraries
import streamlit as st
from PIL import Image

# User module files
from ml import ml
from nn_adam import nn_adam
from nn_sgd import nn_sgd

import sys
import pkg_resources

print("Python version:", sys.version)
print("Installed packages:", [package.key for package in pkg_resources.working_set])

def main():

    #############
    # Main page #
    #############

    options = ['Home','Logistic Regression Predictions', 'Artificial Neural Network Adam Predictions','Aritifical Neural Network SGD Predictions']
    choice = st.sidebar.selectbox("Menu",options, key = '1')

    if ( choice == 'Home' ):
      st.title("Using AI and data to invest in start-up")
      st.text ("Contact: agustin@redondoarena.com")
      st.text ("Choose the model that you prefer on the left side, and check what the models predict about your companies")  
      st.image('./images/portada.png')
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
