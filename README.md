

# Auto EDA 📊

> An automated Exploratory Data Analysis tool that transforms your raw data into actionable insights within seconds.

![AutoEda](https://github.com/user-attachments/assets/2d7944d9-cbea-4f5c-845f-07336dfa8cb1)

## 🌟 Features

- **Interactive Data Analysis**: Upload your CSV file or use pre-loaded datasets for instant analysis
- **Comprehensive Visualizations**: Automatic generation of relevant plots and charts
- **Statistical Insights**: Detailed statistical analysis of your data
- **Missing Data Analysis**: Visual and numerical representation of missing data patterns
- **Correlation Analysis**: Interactive correlation heatmaps and scatter plots
- **Distribution Analysis**: Automated distribution plots for numerical variables
- **PCA Analysis**: Principal Component Analysis with interactive scree plots
- **Downloadable Reports**: Generate and download comprehensive EDA reports

## 🚀 Live Demo

[Try Auto EDA](https://auto-eda.streamlit.app/)

## 💻 Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Statistical Analysis**: SciPy

## 🛠️ Installation & Setup

1. Clone the repository
```bash
git clone https://github.com/sidesshmore/Auto-EDA.git
cd Auto-EDA
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run app.py
```

## 📊 Sample Datasets

The application comes with three pre-loaded datasets:
- **Iris Dataset**: Perfect for classification tasks
- **Titanic Dataset**: Great for both classification and EDA
- **Boston Housing Dataset**: Ideal for regression analysis

## 🔍 Key Components

- **Data Loading**: Support for CSV files with automatic type inference
- **Basic Statistics**: Summary statistics, data types, missing values
- **Visualization Suite**: 
  - Distribution plots
  - Correlation heatmaps
  - Scatter plots
  - Missing data visualizations
  - Categorical data pie charts
- **Advanced Analysis**: 
  - Principal Component Analysis (PCA)
  - Automated report generation

## 💡 Usage

1. Visit the application URL or run locally
2. Choose between demo datasets or upload your own CSV file
3. Explore automatically generated visualizations and insights
4. Download comprehensive reports for further analysis
