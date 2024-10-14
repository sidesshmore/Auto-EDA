import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import io

# Set page config
st.set_page_config(page_title="AutoML EDA Pipeline")

# Title and description
st.title("Auto Pipeline for Exploratory Data Analysis (EDA) 😎⚙️")
st.write("Upload your CSV file to get started with automatic EDA!")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data(file):
    df = pd.read_csv(file)
    return df

# Function to display basic information
def show_basic_info(df):
    st.write("### Basic Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write("### Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(col_info)

# Function to display basic statistics
def show_basic_statistics(df):
    st.write("### Basic Statistics")
    st.write(df.describe().T)

# Function to visualize missing data
def show_missing_data(df):
    st.write("### Missing Data Visualization")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

    if not missing_data.empty:
        fig = px.bar(x=missing_data.index, y=missing_data.values,
                     labels={'x': 'Features', 'y': 'Number of Missing Values'},
                     title='Missing Values by Feature',
                     color=missing_data.values,
                     color_continuous_scale='Viridis')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig)
    else:
        st.write("No missing data found!")

# Function to display correlation heatmap
def show_correlation_heatmap(df):
    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig = px.imshow(corr, color_continuous_scale='RdYlBu_r', aspect="auto")
    fig.update_layout(title='Correlation Heatmap')
    st.plotly_chart(fig)

# Function to show distribution plots
def show_distribution_plots(df):
    st.write("### Distribution Plots")
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}",
                           color_discrete_sequence=['rgba(0, 128, 255, 0.7)'])
        fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
        st.plotly_chart(fig)

# Function to show scatter plots
def show_scatter_plots(df):
    st.write("### Scatter Plots")
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Default selection
    default_x = numeric_columns[0]  
    default_y = numeric_columns[1]  

    x_axis = st.selectbox("Select X-axis", numeric_columns, index=0)
    y_axis = st.selectbox("Select Y-axis", numeric_columns[numeric_columns != x_axis], index=1)

    fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot: {x_axis} vs {y_axis}",
                     color=df.columns[0], color_continuous_scale='Viridis')
    st.plotly_chart(fig)


# Function to perform and visualize PCA
def perform_pca(df):
    st.write("### Principal Component Analysis (PCA)")
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(numeric_df)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(imputed_data)
    
    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Create a scree plot
    fig = px.line(x=range(1, len(explained_variance_ratio) + 1), y=np.cumsum(explained_variance_ratio),
                  labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance Ratio'},
                  title='PCA Scree Plot')
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="80% explained variance")
    fig.update_traces(line_color='rgb(0, 128, 255)', line_width=2)
    st.plotly_chart(fig)

def show_categorical_pie_charts(df):
    st.write("### Categorical Data Distribution")
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    if len(categorical_columns) == 0:
        st.write("No categorical columns found!")
    else:
        for col in categorical_columns:
            st.write(f"#### Distribution of {col}")
            data = df[col].value_counts()

            fig = px.pie(values=data.values, names=data.index, title=f"Distribution of {col}")
            st.plotly_chart(fig)

# Function to generate a comprehensive report
def generate_report(df):
    report = io.StringIO()
    report.write("Exploratory Data Analysis Report\n\n")
    
    report.write("1. Basic Information\n")
    report.write(f"Number of rows: {df.shape[0]}\n")
    report.write(f"Number of columns: {df.shape[1]}\n\n")
    
    report.write("2. Column Information\n")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    report.write(col_info.to_string(index=False))
    report.write("\n\n")
    
    report.write("3. Basic Statistics\n")
    report.write(df.describe().T.to_string())
    report.write("\n\n")
    
    report.write("4. Missing Data\n")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    report.write(missing_data.to_string())
    report.write("\n\n")
    
    report.write("5. Correlation Matrix\n")
    numeric_df = df.select_dtypes(include=[np.number])
    report.write(numeric_df.corr().to_string())
    report.write("\n\n")
    
    return report.getvalue()

if uploaded_file is not None:
    # Load and preprocess the data
    df = load_and_preprocess_data(uploaded_file)
    
    # Display data preview
    st.write("### Preview of the Dataset", df.head())
    
    # Show basic information
    show_basic_info(df)
    
    # Show pie chart for categorical data
    show_categorical_pie_charts(df)
    
    # Show basic statistics
    show_basic_statistics(df)
    
    # Show missing data visualization
    show_missing_data(df)
    
    # Show correlation heatmap
    show_correlation_heatmap(df)
    
    # Show distribution plots
    show_distribution_plots(df)
    
    # Show scatter plots
    show_scatter_plots(df)
    
    # Generate and download the report
    if st.button("Generate and Download Report"):
        report = generate_report(df)
        st.download_button(
            label="Download Report",
            data=report,
            file_name='eda_report.txt',
            mime='text/plain'
        )
else:
    st.write("Awaiting CSV file upload...")