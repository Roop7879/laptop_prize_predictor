import streamlit as st
import pickle
import numpy as np
st.title('Laptop Prize Predictor')

# import models
pipe = pickle.load(open('pipe1.pkl','rb'))
data = pickle.load(open('df1.pkl','rb'))

#Company	
company = st.selectbox("Brand",data.Company.unique())

# TypeName	
typename= st.selectbox("TypeName",data.TypeName.unique())

# Ram
ram=st.selectbox("Ram(in GB)",data.Ram.unique())

# Gpu
gpu=st.selectbox("GPU",data.Gpu.unique())

# OpSys	
opsys=st.selectbox("OS",data.OpSys.unique())

# Weight	
weight = st.number_input('Weight ')

# IPS	
ips=st.selectbox("IPS",['No','Yes'])

# Touchscreen
touchcreen=st.selectbox("Touchscreen",['No','Yes'])	

#screeen_size
screen_size= st.number_input('Screen Size')

#Resolution
resolution = st.selectbox("Resolution",['1280x800','1366x768','1600x900','1920x1080','2256x1504','2736x1824','2560x1440','3200x1800','3840x2160'])

# Processor_Name	
cpu=st.selectbox("CPU",data.Processor_Name.unique())

# HDD
hdd=st.selectbox("HDD(in GB)",data.HDD.unique())
# SSD	
ssd=st.selectbox("SSD(in GB)",data.SSD.unique())


if st.button("Predict"):
    if touchcreen == 'Yes':
        touchcreen = 1
    else:
        touchcreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_resol = int(resolution.split('x')[0])
    Y_resol = int(resolution.split('x')[1])

    ppi = ((X_resol)**2 + (Y_resol)**2)**0.5 / screen_size

    query = np.array([company, typename, ram, gpu, opsys, weight, ips, touchcreen, ppi, cpu, hdd, ssd])
    query = query.reshape(1, 12)

    st.title("The predicted price of this configuration is "+ str(int(np.exp(pipe.predict(query))))) 
