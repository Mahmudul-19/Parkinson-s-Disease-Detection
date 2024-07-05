
import streamlit as st

# Function to display the announcement
def app():
    st.title("Parkinson's Disease Detection")
    
    # Display the announcement
    announcement = """
    <div style='padding: 20px; background-color: #ffcccc; text-align: center; border: 2px solid #ff0000; border-radius: 10px;'>
        <h2 style='color: #ff0000;'>Negative, No Parkinson's Found</h2>
    </div>
    """
    announcement = """
    <div style='
        padding: 20px; 
        background-color: #b3ffb3; 
        text-align: center; 
        border: 2px solid #00cc00; 
        border-radius: 10px; 
        margin-top: 20px;'>
        <h2 style='color: #00cc00;'>Positive, Parkinson's Detected</h2>
    </div>
    """
    st.markdown(announcement, unsafe_allow_html=True)

app()
