# Real-Time Parkinson's Disease Detection


## Project Overview
This project is a Parkinson's disease detection system built using machine learning and Streamlit. The application allows users to test if a person has Parkinson's disease or not, add training data dynamically, and evaluate the model's performance. The model uses the LogisticRegression algorithm from scikit-learn but not using library, and additional features are extracted from the dataset generator(feature name) by using raw data to improve detection accuracy. 

## Features
1. **Visualization :** Shows plots of different attributes of the dataset  that can be selected by the user .
2. **Training :** Train the model and display the scores of the performances of the model, the confusion matrix and the ROC curve.
3. **Prediction :** Users can input a mp3 audio file of a person's voice and the system will classify it as  Parkinson's or not and also display the value of 22 attributes of person's voice like frequency, jitter etc.
4. **Add Training Data or Dataset Generator :** Users can add new emails and label them as spam or not spam, which will be used to retrain the model.
5. **Evaluate Model :** Users can evaluate the model's performance on a test dataset.
6. **FAQ :** There are some sample question on regards of the project.

## Libraries and Tools Used
- **Streamlit :** A framework for building interactive web applications.
- **Pandas :** A data manipulation and analysis library for Python.
- **NumPy :** A library for numerical operations in Python.
- **IO :**The io module provides Python's main facilities for dealing with various types of I/O.
- **Nolds :** A small numpy-based library that pro- vides an implementation and a learning resource for non- linear measures for dynamical systems based on one- dimensional time series.
- **OS :** The OS module in Python is an indispensable tool for handling file-related tasks for programmers.
- **Librosa :** Librosa is a library that is used for analyzing the behavior of audio. It helps in loading audio files, extracting the characteristics of the music, and visualizing audio data.
- **Matplotlib :** Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
- **Seaborn :** Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
  - **Features Used :**
  - **Pydub :** Provides a gold mine of tools for manipulating audio files.  
- **Model Training :** Logistic Regression is employed to build and train the Parkinson's disease detection model.
- **Pickle :** A module for serializing and deserializing Python objects. It is used for saving and loading the trained model and vectorizer.

## Installation Instructions
To get a local copy of the project up and running, follow these simple steps:

1. **Clone the Repository :**
   
   ```bash
   git clone https://github.com/ShahiduzzamanSajid/SPAM-HAM-Mail-Detection.git
   cd SPAM-HAM-Mail-Detection

 2. **Install the Required Libraries :**
   To install all the necessary libraries and dependencies for the project, you can use the **requirements.txt** file. Run the following command:

    ```bash
    pip install -r requirements.txt
    ```
    
    Alternatively, you can install the libraries manually with the following commands:

    ```bash
    pip install pandas
    pip install numpy
    pip install scikit-learn
    pip install imbalanced-learn
    pip install streamlit

 3. **Run the Streamlit App :**
  
     ```bash
     streamlit run Mail_detection_latest.py
     

## Contributing
Contributions to improve Real-Time Spam ham Detection are welcome. To contribute, follow these steps :

1. Fork the repository on GitHub.
2. Create a new branch with a descriptive name.
3. Make your changes and commit them with clear comments.
4. Push your changes to your fork.
5. Open a pull request, explaining the changes made.

## License
This project is licensed under the MIT License. 
