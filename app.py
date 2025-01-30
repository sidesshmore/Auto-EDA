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
st.set_page_config(page_title="AutoML EDA Pipeline", layout="wide")

# Title and description
st.title("Auto Pipeline for Exploratory Data Analysis (EDA) ⚙️")

# Function to load sample data
@st.cache_data
def load_sample_data(dataset_name):
    if dataset_name == "Iris Dataset":
        return pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    elif dataset_name == "Titanic Dataset":
        return pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    elif dataset_name == "Boston Housing Dataset":
        return pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
    return None

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
    st.write("### Scatter Plots (You can change the X and Y axis)")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) >= 2:
        # Default selection
        x_axis = st.selectbox("Select X-axis", numeric_columns, index=0)
        y_axis = st.selectbox("Select Y-axis", numeric_columns[numeric_columns != x_axis], index=1)
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot: {x_axis} vs {y_axis}",
                         color=df.columns[0], color_continuous_scale='Viridis')
        st.plotly_chart(fig)
    else:
        st.write("Not enough numeric columns for scatter plot")

# Function to perform and visualize PCA
def perform_pca(df):
    st.write("### Principal Component Analysis (PCA)")
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        st.write("Not enough numeric columns for PCA")
        return
    
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

# Create tabs for demo data and file upload
st.write("Choose a sample dataset or upload your own CSV file!")
tab1, tab2 = st.tabs(["📊 Demo Datasets", "📁 Upload Your Data"])

with tab1:
    demo_dataset = st.selectbox(
        "Select a sample dataset",
        ["Select a dataset...", "Iris Dataset", "Titanic Dataset", "Boston Housing Dataset"]
    )
    
    if demo_dataset != "Select a dataset...":
        df = load_sample_data(demo_dataset)
        st.success(f"Loaded {demo_dataset} successfully!")
        
        # Add dataset descriptions
        if demo_dataset == "Iris Dataset":
            st.info("""
            The Iris dataset contains measurements for 150 iris flowers from three different species.
            Features include: sepal length, sepal width, petal length, and petal width.
            """)
        elif demo_dataset == "Titanic Dataset":
            st.info("""
            The Titanic dataset contains information about passengers aboard the Titanic.
            Features include: age, sex, passenger class, survival status, and more.
            """)
        elif demo_dataset == "Boston Housing Dataset":
            st.info("""
            The Boston Housing dataset contains information about housing in Boston.
            Features include: crime rate, room numbers, property tax rates, and more.
            """)

with tab2:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_and_preprocess_data(uploaded_file)

# Main analysis section
if 'df' in locals():
    # Display data preview
    st.write("### Preview of the Dataset", df.head())
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Show basic information
        show_basic_info(df)
        
        # Show basic statistics
        show_basic_statistics(df)
    
    with col2:
        # Show missing data visualization
        show_missing_data(df)
        
        # Show correlation heatmap
        show_correlation_heatmap(df)
    
    # Show pie chart for categorical data
    show_categorical_pie_charts(df)
    
    # Show distribution plots
    show_distribution_plots(df)
    
    # Show scatter plots
    show_scatter_plots(df)
    
    # Perform PCA
    perform_pca(df)
    
    # Generate and download report
    if st.button("Generate and Download Report"):
        report = generate_report(df)
        st.download_button(
            label="Download Report",
            data=report,
            file_name='eda_report.txt',
            mime='text/plain'
        )
else:
    st.write("Please select a demo dataset or upload your own CSV file to begin the analysis.")

# Footer
st.write("### Connect with Me!")
st.write("[LinkedIn](https://www.linkedin.com/in/sidessh/) | [Twitter](https://x.com/SidesshMore)")