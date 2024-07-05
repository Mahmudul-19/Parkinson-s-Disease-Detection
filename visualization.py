import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns


def app():
    uploaded_file = st.file_uploader("Please Choose the dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        targetCol = 'status'                                                    # defining target column
        targetColDf = df.pop(targetCol)                                     # popping target column from loanData df
        df.insert(len(df.columns),targetCol, targetColDf)               # inserting target column to last column

    # deleting variables that were used for changing column position of target column
        del targetCol 
        del targetColDf

    # converting column names into lower case
        df.columns = [c.lower() for c in df.columns]
    # replacing spaces in column names with '_'
        df.columns = [c.replace(' ', '_') for c in df.columns]
    # replacing ':' in column names with '_'
        df.columns = [c.replace(':', '_') for c in df.columns]
    # replacing '(' in column names with '_'
        df.columns = [c.replace('(', '_') for c in df.columns]
    # replacing ')' in column names with '' i.e blank
        df.columns = [c.replace(')', '') for c in df.columns]
    # replacing '%' in column names with 'in_percent'
        df.columns = [c.replace('%', 'in_percent') for c in df.columns]
        df.info()
        def skew(data):
            """
        Calculate the skewness of a dataset.

        Parameters:
        data (array-like): Input data.

        Returns:
        float: Skewness of the input data.
        """
            data = np.asarray(data)
            n = data.shape[0]

            mean = np.mean(data)
            std_dev = np.std(data, ddof=0)

            skewness = (1 / n) * np.sum(((data - mean) / std_dev) ** 3)
    
            return skewness

        def plot_data(df, plot_type, grid_size, fig_size, y = None):
            fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=fig_size)
            column_names = df.select_dtypes(exclude='object').columns
            for i, column_name in enumerate(column_names):
                row = i // grid_size[1]
                col = i % grid_size[1]
                ax = axes[row, col]
                if plot_type == 'hist':
                    sns.histplot(df[column_name], kde=True, color='darkblue', ax=ax)
                elif plot_type == 'boxplot':
                    sns.boxplot(y=df[column_name], x=y, color='red', ax=ax)
                else:
                    raise ValueError("Input value for the parameter 'plot_type' should be 'hist' or 'boxplot'.")
                ax.set_xlabel(column_name, fontsize=16)
            plt.tight_layout()
            return fig
    
        numeric_columns = df.select_dtypes(include=np.number)
        skewness = numeric_columns.skew().sort_values(ascending=False)
        #print(skewness)
    
        st.title("Distribution of Three Main Feature")
        feature = 'mdvp_fo_hz'
        meanData = 'Mean : ' + str(round(df[feature].mean(),4))        # variable to contain mean of the attribute
        skewData = 'Skewness : ' + str(round(df[feature].skew(),4))    # variable to contain skewness of the attribute
        plt.figure(figsize=(10,5))                                         
    # seaborn distplot to examine distribution of the feature of healthy patient
        fig = sns.histplot(df[df['status'] == 0][feature], bins=30, kde=True, label='Healthy')
    # seaborn distplot to examine distribution of the feature of Parkinson's patient
        fig = sns.histplot(df[df['status'] == 1][feature], bins=30, kde=True, label='Parkinson\'s')
        plt.legend()
        plt.title("Distribution of feature : "+feature+" having "+meanData+" and "+skewData)                    # setting title of the figure
        st.pyplot(plt)
    
        feature = 'mdvp_fhi_hz'
        meanData = 'Mean : ' + str(round(df[feature].mean(),4))        # variable to contain mean of the attribute
        skewData = 'Skewness : ' + str(round(df[feature].skew(),4))
        plt.figure(figsize=(10,5))                                         # setting figure size with width = 10 and height = 5
# seaborn distplot to examine distribution of the feature of healthy patient
        fig = sns.histplot(df[df['status'] == 0][feature], bins=30, kde=True, label='Healthy')
# seaborn distplot to examine distribution of the feature of Parkinson's patient
        fig = sns.histplot(df[df['status'] == 1][feature], bins=30, kde=True, label='Parkinson\'s')
        plt.legend()
        plt.title("Distribution of feature : "+feature+" having "+meanData+" and "+skewData)                    # setting title of the figure
        st.pyplot(plt)
    
        feature = 'mdvp_flo_hz'
        plt.figure(figsize=(10,5))                                         # setting figure size with width = 10 and height = 5
    # seaborn distplot to examine distribution of the feature of healthy patient
        fig = sns.histplot(df[df['status'] == 0][feature], bins=30, kde=True, label='Healthy')
    # seaborn distplot to examine distribution of the feature of Parkinson's patient
        fig = sns.histplot(df[df['status'] == 1][feature], bins=30, kde=True, label='Parkinson\'s')
        plt.legend()            # seaborn distplot to examine distribution of the feature
        plt.title("Distribution of feature : "+feature+" having "+meanData+" and "+skewData)   # setting title of the figure
        st.pyplot(plt)
    
        col1, col2, col3 = st.columns([0.5,2,0.5])
        with col2:
            st.title("*Data Plots of All Feature*")
        fig = plot_data(df, plot_type = 'hist', grid_size = (8,3), fig_size = (12, 20))
        st.pyplot(fig)
    
        plt.figure(figsize=[15, 8], dpi=100)
        plt.title("Correlation Graph", fontsize=20)
        def drop_name_column_if_exists(df):
    # Check if 'name' column exists and drop it
            if 'name' in df.columns:
                df = df.drop(columns=['name'])
            return df
        
        data = drop_name_column_if_exists(df)
    # Create a blue color map
        cmap = sns.color_palette("Blues")
        col1, col2, col3 = st.columns([0.2,2,0.2])
        with col2:
            st.title("*Correlation Graph of Dataset*")
        sns.heatmap(data.corr(), annot=True, cmap=cmap)
        st.pyplot(plt)