import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import os

# Add a title at the beginning of the UI
st.title('Real-Time Stock Data & Charting Tool')

# Function to retrieve S&P 500 stock tickers
def get_sp500_stocks():
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 
        'TSLA', 'BRK.B', 'JNJ', 'V', 'JPM',
        # Add other tickers as necessary
    ]

def calculate_risk_score(answers):
    score = sum(answers)
    return score

def risk_tolerance_quiz():
    st.title("Risk Tolerance Quiz")
    
    questions = [
        "How would you describe your current investment knowledge?",
        "What is your investment time horizon?",
        "How do you react to market fluctuations?",
        "What is your primary investment goal?",
        "How comfortable are you with the idea of losing money?"
    ]

    options = {
        "How would you describe your current investment knowledge?": ["Novice", "Intermediate", "Expert"],
        "What is your investment time horizon?": ["Less than 1 year", "1-5 years", "5+ years"],
        "How do you react to market fluctuations?": ["Panic", "Stay calm", "View it as an opportunity"],
        "What is your primary investment goal?": ["Preserve capital", "Generate income", "Grow wealth"],
        "How comfortable are you with the idea of losing money?": ["Very uncomfortable", "Neutral", "Comfortable"],
    }

    answers = []
    for question in questions:
        answer = st.radio(question, options[question], key=question)
        answer_values = {
            "Novice": 1, "Intermediate": 2, "Expert": 3,
            "Less than 1 year": 1, "1-5 years": 2, "5+ years": 3,
            "Panic": 1, "Stay calm": 2, "View it as an opportunity": 3,
            "Preserve capital": 1, "Generate income": 2, "Grow wealth": 3,
            "Very uncomfortable": 1, "Neutral": 2, "Comfortable": 3
        }
        answers.append(answer_values[answer])

    if st.button("Submit"):
        risk_score = calculate_risk_score(answers)
        st.write("Your Risk Tolerance Score is:", risk_score)
        
        if risk_score <= 8:
            risk_tolerance = "Low"
        elif risk_score <= 12:
            risk_tolerance = "Medium"
        else:
            risk_tolerance = "High"
        
        st.write(f"Risk level: {risk_tolerance}")
        return risk_tolerance

# Get the user's risk level and start the quiz if necessary
risk_level = st.selectbox("What is your risk level?", ["High", "Medium", "Low", "No idea, Quiz me"])

if risk_level == "No idea, Quiz me":
    chosen_risk_level = risk_tolerance_quiz()
    risk_level = chosen_risk_level

# Load stocks and categorize them by risk level into CSVs
def categorize_stocks_by_risk():
    sp500_stocks = get_sp500_stocks()
    
    # Create empty lists for categorized stocks
    low_risk_stocks = []
    medium_risk_stocks = []
    high_risk_stocks = []

    # Simulated categorization logic for illustration
    for stock in sp500_stocks:
        if stock in ['AAPL', 'MSFT']:  # Example: High risk
            high_risk_stocks.append(stock)
        elif stock in ['JNJ', 'PFE']:  # Example: Medium risk
            medium_risk_stocks.append(stock)
        else:  # Default to low risk
            low_risk_stocks.append(stock)

    # Saving to CSV files
    pd.DataFrame(low_risk_stocks, columns=["Ticker"]).to_csv("low_risk_stocks.csv", index=False)
    pd.DataFrame(medium_risk_stocks, columns=["Ticker"]).to_csv("medium_risk_stocks.csv", index=False)
    pd.DataFrame(high_risk_stocks, columns=["Ticker"]).to_csv("high_risk_stocks.csv", index=False)

    return low_risk_stocks, medium_risk_stocks, high_risk_stocks

# Call the categorization function
low_risk, medium_risk, high_risk = categorize_stocks_by_risk()

# Investment Amount input
investment_amount = st.number_input('Enter amount for investment', min_value=1, value=1000, step=100)

# Time Frame Selection
selected_timeframe = st.selectbox("Select your investment time frame:", [
    "Day trading (Inactive)", 
    "Short-term trading (2-30 days)",  
    "Medium-term investing (31-90 days) (Inactive)",  
    "Long-term investing (90+ days) (Inactive)" 
])

# Initialize days variable
days = 0
if selected_timeframe == "Short-term trading (2-30 days)":
    days = st.number_input("Enter number of days (2-30)", min_value=2, max_value=30, value=20)
else:
    st.warning("Please select 'Short-term trading (2-30 days)' as the active strategy for now.")

# After inputs are provided
if days > 0 and investment_amount > 0:
    # Determine the list of stocks to analyze based on the user's risk level
    if risk_level == "High":
        stocks_to_analyze = high_risk
    elif risk_level == "Medium":
        stocks_to_analyze = medium_risk
    else:
        stocks_to_analyze = low_risk

    stock_forecasts = []
    
    for ticker in stocks_to_analyze:
        data = yf.Ticker(ticker)
        hist = data.history(period='max')

        if not hist.empty:
            now = pd.Timestamp.now(tz='America/New_York')
            last_60_days = now - pd.Timedelta(days=60)
            hist = hist[hist.index > last_60_days]

            # Prepare data for Prophet
            prophet_data = hist.reset_index()[['Date', 'Close']]
            prophet_data.columns = ['ds', 'y']
            prophet_data['ds'] = prophet_data['ds'].dt.tz_localize(None)

            # Fit the Prophet model
            model = Prophet(daily_seasonality=True)
            model.fit(prophet_data)

            # Make future predictions for the specified period
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)

            # Calculate residuals for analysis
            forecast_actual = forecast[['ds', 'yhat']]
            forecast_actual = forecast_actual.set_index('ds')
            combined = pd.concat([prophet_data.set_index('ds'), forecast_actual], axis=1)
            combined.columns = ['actual', 'predicted']
            combined['residuals'] = combined['actual'] - combined['predicted']

            # Using the first forecasted value for the buy price 
            # and the last forecasted value for the sell price
            buy_price = forecast['yhat'].iloc[0]  # Price at the start of the forecast
            sell_price = forecast['yhat'].iloc[-1]  # Price at the end of the forecast
            buy_date = forecast['ds'].iloc[0]  # Buy date is the first forecasted date
            sell_date = forecast['ds'].iloc[-1]  # Sell date is the last forecasted date

            # Ensure that the sell date comes after the buy date is always true here
            if buy_price < sell_price:
                expected_return = (sell_price - buy_price) / buy_price
                num_shares = investment_amount // buy_price

                # Store forecast results
                stock_forecasts.append((
                    ticker,
                    buy_price,
                    sell_price,
                    expected_return,
                    num_shares,
                    buy_date,
                    sell_date,
                    combined
                ))

    # Create a DataFrame for the analysis
    forecast_df = pd.DataFrame(stock_forecasts, columns=['Ticker', 'Buy Price', 'Sell Price', 'Expected Return', 'Shares', 'Buy Date', 'Sell Date', 'Combined'])
    forecast_df = forecast_df.sort_values(by='Expected Return', ascending=False).head(3)

    # Display results
    st.subheader('Top 3 Stock Recommendations')
    st.write(forecast_df[['Ticker', 'Buy Price', 'Sell Price', 'Expected Return', 'Shares', 'Buy Date', 'Sell Date']])

    # Residual analysis for each recommended stock
    for index, row in forecast_df.iterrows():
        combined = row['Combined']
        
        mean_residual = combined['residuals'].mean()
        std_residual = combined['residuals'].std()
        max_residual = combined['residuals'].max()
        min_residual = combined['residuals'].min()
        
        if abs(mean_residual) < 0.1 * combined['actual'].mean():
            bias_message = "The model has a small bias in its predictions."
        else:
            bias_message = "The model shows significant bias in its predictions."
        
        variability_message = f"The standard deviation of the residuals is {std_residual:.2f}, indicating variability in the predictions."
        
        if std_residual < 2:
            prediction_message = "The model is a good fit with minimal residuals. The forecast is accurate."
        else:
            prediction_message = "The model is a good fit with minimal residuals. The forecast is not accurate. I suggest using a different model or seek expert advice."

        # Display residual analysis
        st.subheader(f'Residuals Analysis for {row["Ticker"]}')
        st.markdown(f"* **Mean Residual:** {mean_residual:.2f}")
        st.markdown(f"* **Max Residual:** {max_residual:.2f}")
        st.markdown(f"* **Min Residual:** {min_residual:.2f}")
        st.markdown(f"* **Standard Deviation of Residuals:** {std_residual:.2f}")
        
        # Display the automated messages
        st.markdown(f"\n### Summary:\n- {bias_message}\n- {variability_message}\n- {prediction_message}")

        # Plot the actual vs predicted for the selected stock
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined.index, y=combined['actual'], mode='lines+markers', name='Actual Price'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Price', line=dict(dash='dash')))
        
        fig.update_layout(
            title=f'{row["Ticker"]} Closing Prices and Predictions',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='closest'
        )
        
        # Show the plot for each stock
        st.plotly_chart(fig)

else:
    st.warning("Please ensure you have selected a timeframe and entered a valid investment amount.")

# New Question at the end
st.subheader("How do you want to proceed?")
option = st.selectbox("Choose an option:", ["Take me to a trading platform to invest.", "Take me to an expert to consult."])
