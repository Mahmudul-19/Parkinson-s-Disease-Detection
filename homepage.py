import streamlit as st
from streamlit_option_menu import option_menu


import training, visualization, prediction, faq, home, new_training

st.set_page_config(
    page_title= "Parkinson's Disease",

)


            
class MultiApp:
    def __init__(self):
        self.apps = []
    def add_app(self, title, function):
        self.apps.append({
            "title" : title,
            "function" : function
        })
        
    def run():
       
        with st.sidebar:
            app = option_menu(
                menu_title= 'Features',
                options=['Home','Training', 'Visualization', 'Prediction', 'Dataset Generator','FAQ'],
                
                default_index= 0,
                styles={
                    "container": {"padding": "5!important","background-color":'black'},
        "icon": {"color": "white", "font-size": "23px"}, 
        "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
        "nav-link-selected": {"background-color": "#02ab21"},
                }
        )
        
        if app == "Home":
            home.app()
        if app == "Training":
            training.app()
        if app == "Visualization":
            visualization.app()    
        if app == "Prediction":
            prediction.app() 
        if app == "Dataset Generator":
            new_training.app() 
        if app == "FAQ":
            faq.app()      
    
    
    run()
    