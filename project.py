import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Interactive Data Exploration with Streamlit")

st.markdown("""
Welcome to this interactive data exploration app built with **Streamlit**, **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn**.  
Upload your CSV file to get insights and visualizations of your data!
""")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File loaded successfully!")
        
        # Show dataframe shape and head
        st.subheader("Dataset Overview")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head())

        # Show descriptive statistics
        st.subheader("Descriptive Statistics")
        st.write(df.describe())

        # Show missing values summary
        st.subheader("Missing Values Count")
        missing_counts = df.isnull().sum()
        st.write(missing_counts[missing_counts > 0] if any(missing_counts > 0) else "No missing values detected.")

        # Select numeric columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) > 0:
            st.subheader("Correlation Heatmap")
            corr = df[numeric_cols].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            st.pyplot()

            st.subheader("Distribution Plots")
            selected_cols = st.multiselect("Select numeric columns to plot distributions", numeric_cols, default=numeric_cols[:3])
            for col in selected_cols:
                plt.figure(figsize=(6, 4))
                sns.histplot(df[col].dropna(), kde=True, color='blue')
                plt.title(f'Distribution and KDE of {col}')
                st.pyplot()
            
            st.subheader("Scatterplot Matrix (Pairplot)")
            if len(selected_cols) > 1:
                pairplot_fig = sns.pairplot(df[selected_cols].dropna())
                st.pyplot(pairplot_fig)
            else:
                st.info("Select at least two numeric columns for pairplot.")

            # Show basic numpy statistics
            st.subheader("Additional Statistics (Using NumPy)")
            for col in selected_cols:
                col_data = df[col].dropna()
                stats = {
                    'Mean': np.mean(col_data),
                    'Median': np.median(col_data),
                    'Std Dev': np.std(col_data),
                    'Variance': np.var(col_data),
                    'Min': np.min(col_data),
                    'Max': np.max(col_data),
                    '25th Percentile': np.percentile(col_data, 25),
                    '50th Percentile': np.percentile(col_data, 50),
                    '75th Percentile': np.percentile(col_data, 75),
                }
                st.write(f"**Statistics for {col}:**")
                st.json({k: float(f'{v:.4f}') for k,v in stats.items()})
                # Categorical Data Analysis
                st.subheader("Categorical Data Analysis")
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    selected_cat_col = st.selectbox("Select a categorical column to visualize", categorical_cols)
                    plt.figure(figsize=(8, 6))
                    sns.countplot(data=df, x=selected_cat_col)
                    plt.title(f'Count Plot of {selected_cat_col}')
                    st.pyplot()
                else:
                    st.warning("No categorical columns found in this dataset.")

        else:
            st.warning("No numeric columns found in this dataset for correlation or distribution plots.")
    except Exception as e:
        st.error(f"Error loading or processing the file: {e}")
else:
    st.info("Please upload a CSV file to get started.")

