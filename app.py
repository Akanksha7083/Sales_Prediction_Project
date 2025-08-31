 import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from model_code import load_data, feature_engineering, prepare_data, train_model, evaluate_model

# Streamlit configuration
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the light neutral color theme
st.markdown("""
    <style>
        .main {
            background-color: #f5f5dc; /* Beige background */
        }
        .stButton > button {
            background-color: #d3b692; /* Light brown buttons */
            color: white;
            border-radius: 10px;
            width: 150px;
            height: 40px;
        }
        .stButton > button:hover {
            background-color: #c49a6c; /* Light coffee on hover */
            color: white;
        }
        h1, h3, h5, .stMarkdown {
            color: #5b4636; /* Light coffee/brown text */
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title('📊 Walmart Sales Forecasting')

# Load and process data
df = load_data()
df = feature_engineering(df)

# Display data overview
st.write("### 🗂 Data Overview")
st.dataframe(df.head(), height=300)
st.write("### 📄 Columns in the Dataset")
st.write(df.columns)

# Create additional features
st.write("### 📅 Number of NaT values in 'Date'")
nat_count = df['Date'].isnull().sum()
st.write(f"Number of missing dates: {nat_count}")

# Create a dropdown for selecting a store
store_list = df['Store'].unique()  # Get unique stores
selected_store = st.selectbox("Select a Store to Plot Sales", store_list)

# Filter the data based on the selected store
filtered_df = df[df['Store'] == selected_store]

# Plot Weekly Sales Over Time
st.write(f"### 📈 Weekly Sales Over Time for Store {selected_store}")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered_df['Date'],
    y=filtered_df['Weekly_Sales'],
    mode='lines',
    line=dict(color='#C49A6C')  # Light coffee color for the line
))
fig.update_layout(
    title=f'Weekly Sales Over Time for {selected_store}',
    xaxis_title='Date',
    yaxis_title='Weekly Sales',
    template='plotly_white',  # Light, neutral background
    plot_bgcolor='#f8f1e3',   # Light beige plot background
    paper_bgcolor='#f8f1e3'   # Light beige background
)
st.plotly_chart(fig)

# Check the correlation between features
st.write("### 🔍 Correlation Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')  # Using light blue-green for a neutral tone
plt.title('Correlation Heatmap')
plt.tight_layout()
st.pyplot(plt)

# Prepare the data for modeling
X_train, X_test, y_train, y_test = prepare_data(df)

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
y_pred, mae, rmse, mape, accuracy = evaluate_model(model, X_test, y_test)

st.write("### 📊 Model Performance Metrics")
st.metric(label="Mean Absolute Error", value=f"{mae:.2f}")
st.metric(label="Root Mean Squared Error", value=f"{rmse:.2f}")
st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{mape:.2f}%")
st.metric(label="Accuracy", value=f"{accuracy:.2f}%")

# Visualize the predictions vs actual sales
st.write("### 🔮 Actual vs Predicted Sales")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(len(y_test))),
    y=y_test.values,
    mode='lines',
    name='Actual Sales',
    line=dict(color='#5b4636')  # Darker brown color
))
fig.add_trace(go.Scatter(
    x=list(range(len(y_pred))),
    y=y_pred,
    mode='lines',
    name='Predicted Sales',
    line=dict(color='#C49A6C')  # Light coffee color
))
fig.update_layout(
    title='Actual vs Predicted Sales',
    xaxis_title='Test Sample',
    yaxis_title='Weekly Sales',
    template='plotly_white',  # Light background
    plot_bgcolor='#f8f1e3',
    paper_bgcolor='#f8f1e3'
)
st.plotly_chart(fig)                                                      # Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load and clean the dataset
def load_data():
    df = pd.read_csv(r'C:\Users\muska\OneDrive\Desktop\python\Walmart_sales.csv')

    # Convert 'Date' to datetime format with specified format
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

    # Handle missing values in 'Date' column by dropping rows with NaT values
    df = df.dropna(subset=['Date'])
    
    # Fill missing numerical values with column mean (e.g., CPI)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].mean())

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove outliers using the IQR method for 'Weekly_Sales'
    Q1 = df['Weekly_Sales'].quantile(0.25)
    Q3 = df['Weekly_Sales'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Weekly_Sales'] < (Q1 - 1.5 * IQR)) | (df['Weekly_Sales'] > (Q3 + 1.5 * IQR)))]

    return df

# Feature engineering
def feature_engineering(df):
    # Create additional date-based features
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    
    # Lag features: last week's and last month's sales
    df['Weekly_Sales_Lag_1'] = df['Weekly_Sales'].shift(1)
    df['Weekly_Sales_Lag_4'] = df['Weekly_Sales'].shift(4)
    
    # Rolling average feature: 4-week and 12-week averages
    df['Weekly_Sales_Rolling_4'] = df['Weekly_Sales'].rolling(window=4).mean()
    df['Weekly_Sales_Rolling_12'] = df['Weekly_Sales'].rolling(window=12).mean()

    # Fill any remaining NaNs created by lag and rolling features with 0
    df = df.fillna(0)
    
    return df

# Prepare the data for modeling
def prepare_data(df):
    features = ['Store', 'Month', 'Week', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                'Weekly_Sales_Lag_1', 'Weekly_Sales_Lag_4', 'Weekly_Sales_Rolling_4', 'Weekly_Sales_Rolling_12']
    X = df[features]
    y = df['Weekly_Sales']
    # One-hot encode categorical variables (if any)
    X = pd.get_dummies(X, drop_first=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with Random Forest and hyperparameter tuning
def train_model(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

# Make predictions and evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    # Calculate MAPE and Accuracy
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape
    return y_pred, mae, rmse, mape, accuracy

# Visualize the predictions vs actual sales
def plot_predictions(y_test, y_pred):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Sales', color='blue')
    plt.plot(y_pred, label='Predicted Sales', color='orange')
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Test Sample')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load and clean data
    df = load_data()
    
    # Feature engineering
    df = feature_engineering(df)

    # Prepare data for training
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    y_pred, mae, rmse = evaluate_model(model, X_test, y_test)
    print(f'Mean Absolute Error: {mae}')
    print(f'Root Mean Squared Error: {rmse}')

    # Plot actual vs predicted sales
    plot_predictions(y_test, y_pred)