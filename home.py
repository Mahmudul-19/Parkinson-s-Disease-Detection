import streamlit as st

def app():
    col1, col2, col3 = st.columns([1.5, 2, 1])

# Display the image in the center column
    with col2:
        st.image("E:\\3-2 Sem\Parkinson\parkinson.jpg", caption="Parkinson's Awareness", use_column_width=False)
    st.title("*Parkinson's Disease Detection Application*")
    st.markdown("Welcome to the web-application of Parkinson's disease Detection. This application is not only used to detect the Parkinson's disease but also shows the data plot visualization of dataset, trains the dataset and shows the performance of training model.There is also given some frequently asked question.")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
    with col3:
        st.subheader("*:green[Training]*")
        st.markdown("\nTraining option is used to upload the dataset to train a model so that the app can predict a human suffered from Parkinson's or not. After completing the training, it also shows the performance of the training model.")
    
    with col1:
        st.subheader("*:orange[Visualization]*")
        st.markdown("After Selecting the dataset, Visualization option shows main three feature of databar according to healthy person and parkinson's patients and also shows the data plot of all the features. ")
    with col2:
        st.subheader("*:red[Prediction]*")
        st.markdown("\nPrediction option takes a mp3 audio formated input and shows the 22 attributes of the audio and shows the prediction according the training model. ")
    with col4:
        st.subheader("*:violet[FAQ]*")
        st.markdown("\nFAQ option shows some questions about the application and their topic. and the answers are also be given ")
        
    col1, col2, col3 = st.columns([1.5, 1.5,  1])
    
    with col2:
        st.subheader(":green[Dataset Generator]")
        st.markdown("In this option multiple audio file can upload. After uploading the files, it analyze the voice audio and generate a dataset of csv format.")