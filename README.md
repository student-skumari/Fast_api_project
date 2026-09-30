# Fast_api_project
california_housing_prediction
California House Price Prediction API 🏠

This project is a Machine Learning-based REST API developed using FastAPI to predict California house prices. The project uses a trained Random Forest Regressor model to generate house price predictions based on important housing and location features.

The API accepts information such as median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, and longitude. Users can make predictions for a single house through a JSON request or upload a CSV file to generate predictions for multiple houses at once.

The project uses Pandas for data processing, Pydantic for input validation, Joblib for loading the trained machine learning model, and FastAPI for building the API endpoints. It also includes interactive API documentation through Swagger UI, making it easy to test the prediction endpoints.

Key Features
🏠 California house price prediction
🤖 Random Forest Regressor machine learning model
⚡ FastAPI REST API
📊 CSV file upload and batch prediction
✅ Pydantic data validation
🐼 Pandas data processing
📖 Interactive Swagger API documentation
💾 Joblib model loading
🔧 Git and Git LFS for project management
Technologies

Python | FastAPI | Pandas | Scikit-learn | Random Forest | Joblib | Pydantic | Git | GitHub | Git LFS

This project demonstrates how a trained Machine Learning model can be integrated into a production-style API and made accessible through simple HTTP requests.
