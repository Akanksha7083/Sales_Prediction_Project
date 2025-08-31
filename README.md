**Walmart Sales Forecasting**
This project predicts weekly sales across Walmart stores using machine learning. By analyzing historical data, applying feature engineering, and training a Random Forest model, it provides accurate sales forecasts and actionable insights for better inventory planning and promotions.

**Features**
Data Cleaning: Missing values handled, duplicates removed, and outliers filtered.
Feature Engineering: Created date-based features (Month, Week), lag features, and rolling averages.
Modeling: Random Forest Regressor optimized with GridSearchCV.
Evaluation Metrics: MAE, RMSE, MAPE, and accuracy.
Interactive Dashboard: Built with Streamlit for data exploration and visualization.

**Dataset**
Source: Walmart weekly sales dataset (Walmart_Sales.csv).
Columns include: Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, and Unemployment.

**Tech Stack**
Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
Machine Learning: Random Forest Regressor
Visualization: Matplotlib, Seaborn, Plotly
Deployment: Streamlit

**Results**
Achieved strong performance with low error rates (MAE, RMSE).
Interactive dashboard allows:
Sales trend analysis per store
Correlation heatmap of features
Actual vs. Predicted sales comparison

**How to Run**
Clone the repository
git clone https://github.com/your-username/walmart-sales-forecasting.git
cd walmart-sales-forecasting


**Install dependencies**
pip install -r requirements.txt
Run Streamlit app
streamlit run app.py

**Future Work**
Try deep learning models (LSTMs/GRUs) for sequential forecasting.
Add more external factors (e.g., holidays, weather trends).
