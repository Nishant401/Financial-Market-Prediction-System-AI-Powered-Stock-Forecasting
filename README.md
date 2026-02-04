# Financial-Market-Prediction-System-AI-Powered-Stock-Forecasting
 Developed an AI-powered stock prediction system using Temporal Fusion Transformer (TFT) for accurate multi-horizon stock price forecasting. Features an intuitive GUI for stock selection, custom timeframes, and interactive visualization of predictions with performance metrics.
Financial Market Prediction Using Temporal Fusion Transformer (TFT)

This project focuses on developing an advanced financial market prediction system using the Temporal Fusion Transformer (TFT) to forecast future stock price movements. The model is trained and evaluated using AAPL (Apple Inc.) stock market data, a highly liquid and well-studied equity, making it suitable for robust time-series forecasting.

Historical stock data was collected programmatically using yfinance, enabling reliable access to Apple Inc. (AAPL) price data including Open, High, Low, Close, Adjusted Close, and Trading Volume. Additional technical indicators such as moving averages, RSI, and volatility measures were engineered to enrich the feature set and capture market behavior.

The Temporal Fusion Transformer architecture combines LSTM-based temporal encoders, attention mechanisms, and gated residual networks, allowing the model to:

Capture both short-term price variations and long-term market trends

Dynamically select relevant features across time steps

Provide model interpretability through attention weights and feature importance analysis

The complete pipeline includes data extraction via yfinance, data preprocessing and normalization, feature engineering, model training and validation, and performance evaluation using MAE, RMSE, and MAPE. This project highlights strong proficiency in deep learning–based time-series forecasting, financial data analysis, and end-to-end ML pipeline development for real-world financial applications.
