import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import time
import base64

# Set page configuration at the very beginning
st.set_page_config(
    page_title="Stock Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for background image
def set_background():
    """Set background image for main content and sidebar"""
    main_bg = "y.jpg"  # Your background image file
    main_bg_ext = "jpg"
    
    try:
        # Read the image file and encode it to base64
        with open(main_bg, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
            
        # Apply the background to both main area and sidebar
        st.markdown(
            f"""
            <style>
            .reportview-container {{
                background: url(data:image/{main_bg_ext};base64,{img_data});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            .stApp {{
                background: url(data:image/{main_bg_ext};base64,{img_data});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            .sidebar .sidebar-content {{
                background: url(data:image/{main_bg_ext};base64,{img_data});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Background image '{main_bg}' could not be loaded: {e}")
        st.info("The app will continue to function without the background image.")

# Title of the app
st.title('Stock Price Predictor')

# Initialize session state variables
if 'progress_state' not in st.session_state:
    st.session_state['progress_state'] = {
        'data_loaded': False,
        'preprocessed': False,
        'model_trained': False,
        'visualized': False
    }

if 'data' not in st.session_state:
    st.session_state['data'] = None

if 'future_predictions' not in st.session_state:
    st.session_state['future_predictions'] = {
        'days_ahead': 30,
        'generated': False,
        'fig': None  # Store the future prediction figure
    }

# Function to show loading animation
def show_loading_animation(placeholder, duration=1.5):
    """Show loading animation using a Giphy GIF"""
    giphy_url = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeGtvZGI5b3J4dmc1a2tpdjVmdDFibnBpeWY1bGYyN2kxcmpxMmk2aiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/vevRcKoKB1PMxEvH0Q/giphy.gif"
    
    with placeholder.container():
        st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; height: 150px;">
                <img src="{giphy_url}" alt="Loading..." style="max-height: 100%; max-width: 100%;">
            </div>
        """, unsafe_allow_html=True)
        
        time.sleep(duration)

# Create main content area and sidebar
main_content = st.container()
sidebar = st.sidebar

# File uploader in the sidebar
uploaded_file = sidebar.file_uploader("Upload your CSV file", type="csv")

# Function to load data
def load_data(file):
    """Load and display the data."""
    loading_placeholder = main_content.empty()
    show_loading_animation(loading_placeholder)
    
    try:
        data = pd.read_csv(file)
        st.session_state['data'] = data
        st.session_state['progress_state']['data_loaded'] = True
        loading_placeholder.empty()
        with main_content:
            st.success('✅ Data loaded successfully!')
            st.subheader("Dataset Preview")
            st.dataframe(data.head(), use_container_width=True)
    except Exception as e:
        loading_placeholder.empty()
        main_content.error(f"Error loading data: {e}")

def preprocess_data():
    """Preprocess the data."""
    loading_placeholder = main_content.empty()
    show_loading_animation(loading_placeholder)
    
    try:
        data = st.session_state['data'].copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
        st.session_state['data'] = data
        st.session_state['progress_state']['preprocessed'] = True
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines+markers',
            name='Closing Prices',
            line=dict(color='blue'),
            marker=dict(color='blue', size=4)
        ))

        fig.update_layout(
            title="Stock Closing Prices Over Time",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            showlegend=True,
            xaxis_rangeslider_visible=True,
            dragmode='zoom',
            height=500
        )
        
        loading_placeholder.empty()
        with main_content:
            st.success('✅ Data preprocessed successfully!')
            st.subheader("Preprocessed Data")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state['preprocessed_fig'] = fig
    except Exception as e:
        loading_placeholder.empty()
        main_content.error(f"Error preprocessing data: {e}")

def train_model():
    """Train the linear regression model."""
    loading_placeholder = main_content.empty()
    show_loading_animation(loading_placeholder)
    
    try:
        data = st.session_state['data']
        data['days'] = (data.index - data.index.min()).days
        X = data['days'].values.reshape(-1, 1)
        y = data['close'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = model.score(X_test, y_pred)

        st.session_state['model'] = model
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['metrics'] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'slope': model.coef_[0],
            'intercept': model.intercept_
        }
        
        st.session_state['progress_state']['model_trained'] = True
        
        loading_placeholder.empty()
        with main_content:
            st.success('✅ Model trained successfully!')
            st.subheader("Model Performance Metrics")
            
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Mean Squared Error", f"{mse:.4f}")
            with metrics_cols[1]:
                st.metric("Root MSE", f"{rmse:.4f}")
            with metrics_cols[2]:
                st.metric("R² Score", f"{r2:.4f}")
                
            st.info(f"Price = {model.coef_[0]:.4f} × Days + {model.intercept_:.4f}")
    except Exception as e:
        loading_placeholder.empty()
        main_content.error(f"Error training model: {e}")

def generate_future_predictions(days_ahead):
    """Generate and display future predictions."""
    try:
        model = st.session_state['model']
        data = st.session_state['data']
        max_days = data['days'].max()
        
        future_days = np.array(range(max_days+1, max_days+days_ahead+1)).reshape(-1, 1)
        future_prices = model.predict(future_days)
        
        future_fig = go.Figure()
        
        # Historical data
        future_fig.add_trace(go.Scatter(
            x=data['days'].values,
            y=data['close'].values,
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue')
        ))
        
        # Future predictions with a different style
        future_fig.add_trace(go.Scatter(
            x=future_days.flatten(),
            y=future_prices,
            mode='lines+markers',
            name='Future Predictions',
            line=dict(color='red', dash='dash'),
            marker=dict(color='red', size=6)
        ))

        # Add shaded area for future predictions
        future_fig.add_trace(go.Scatter(
            x=np.concatenate([data['days'].values, future_days.flatten()]),
            y=np.concatenate([data['close'].values, future_prices]),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',  # Semi-transparent red fill
            line=dict(color='red'),
            name="Future Area"
        ))

        future_fig.update_layout(
            title=f'Future Stock Price Predictions (Next {days_ahead} Days)',
            xaxis_title='Days since Start',
            yaxis_title='Stock Price ($)',
            template="plotly_dark",
            showlegend=True,
            xaxis_rangeslider_visible=True,
            dragmode='zoom',
            height=500
        )
        
        return future_fig, future_prices
    except Exception as e:
        st.error(f"Error generating future predictions: {e}")
        return None, None

def visualize_results():
    """Visualize the results using Plotly."""
    loading_placeholder = main_content.empty()
    show_loading_animation(loading_placeholder)
    
    try:
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        X_test = st.session_state['X_test']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X_test.flatten(), 
            y=y_test, 
            mode='markers', 
            name='Actual Prices',
            marker=dict(color='blue', size=8)
        ))

        fig.add_trace(go.Scatter(
            x=X_test.flatten(), 
            y=y_pred, 
            mode='lines', 
            name='Predicted Prices',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title='Stock Price: Actual vs Predicted',
            xaxis_title='Days since Start',
            yaxis_title='Stock Price ($)',
            template="plotly_dark",
            showlegend=True,
            legend=dict(orientation="h", y=1.1),
            xaxis_rangeslider_visible=True,
            dragmode='zoom',
            height=600
        )

        st.session_state['progress_state']['visualized'] = True
        
        loading_placeholder.empty()
        with main_content:
            st.success('✅ Results visualized successfully!')
            st.subheader("Prediction Results")
            st.plotly_chart(fig, use_container_width=True)

            # Generate and display predictions for multiple future periods
            st.subheader("Future Price Predictions")

            # Display multiple future prediction charts (30, 100, 120, 365 days)
            future_fig_30, future_prices_30 = generate_future_predictions(30)
            future_fig_100, future_prices_100 = generate_future_predictions(100)
            future_fig_120, future_prices_120 = generate_future_predictions(120)
            future_fig_365, future_prices_365 = generate_future_predictions(365)

            st.plotly_chart(future_fig_30, use_container_width=True)
            st.plotly_chart(future_fig_100, use_container_width=True)
            st.plotly_chart(future_fig_120, use_container_width=True)
            st.plotly_chart(future_fig_365, use_container_width=True)

            # Allow user to download the predictions as CSV
            future_data = pd.DataFrame({
                'Days': np.concatenate([np.arange(len(future_prices_30)), np.arange(len(future_prices_100)), np.arange(len(future_prices_120)), np.arange(len(future_prices_365))]),
                'Predicted Prices': np.concatenate([future_prices_30, future_prices_100, future_prices_120, future_prices_365])
            })

            # Download button
            st.download_button(
                label="Download Predictions Data",
                data=future_data.to_csv(index=False),
                file_name="future_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        loading_placeholder.empty()
        main_content.error(f"Error visualizing results: {e}")

# Apply the background
set_background()

# Create a workflow sidebar
with sidebar:
    st.subheader("Workflow Steps")
    
    step1 = st.button('1️⃣ Load Data', use_container_width=True)
    if step1:
        if uploaded_file is not None:
            # Clear previous results by resetting progress state
            for key in st.session_state['progress_state']:
                if key != 'data_loaded':
                    st.session_state['progress_state'][key] = False
            # Reset future predictions
            st.session_state['future_predictions']['generated'] = False
            load_data(uploaded_file)
        else:
            st.warning('Please upload a CSV file first!')

    step2 = st.button('2️⃣ Preprocess Data', use_container_width=True, 
                    disabled=not st.session_state['progress_state']['data_loaded'])
    if step2:
        # Clear subsequent steps
        for key in ['preprocessed', 'model_trained', 'visualized']:
            if key != 'preprocessed':
                st.session_state['progress_state'][key] = False
        preprocess_data()

    step3 = st.button('3️⃣ Train Model', use_container_width=True, 
                    disabled=not st.session_state['progress_state']['preprocessed'])
    if step3:
        # Clear subsequent steps
        for key in ['model_trained', 'visualized']:
            if key != 'model_trained':
                st.session_state['progress_state'][key] = False
        train_model()

    step4 = st.button('4️⃣ Visualize Results', use_container_width=True, 
                    disabled=not st.session_state['progress_state']['model_trained'])
    if step4:
        visualize_results()
    
    # Add a progress tracker
    st.subheader("Progress")
    progress_statuses = {
        'Data Loading': st.session_state['progress_state']['data_loaded'],
        'Preprocessing': st.session_state['progress_state']['preprocessed'],
        'Model Training': st.session_state['progress_state']['model_trained'],
        'Visualization': st.session_state['progress_state']['visualized']
    }
    
    for step, completed in progress_statuses.items():
        if completed:
            st.success(f"✅ {step} completed")
        else:
            st.info(f"⏳ {step} pending")

# Instructions in main area when no data is loaded
if not st.session_state['progress_state']['data_loaded']:
    with main_content:
        st.info("👈 Please upload a CSV file and follow the steps in the sidebar to analyze your stock data.")
        st.markdown("""
        ### Expected CSV Format:
        Your CSV should include at least these columns:
        - `date`: Date in YYYY-MM-DD format
        - `close`: Closing price of the stock
        
        ### Example:
        ```
        date,close,volume,open,high,low
        2020-01-01,145.23,10000000,144.50,146.20,143.75
        2020-01-02,147.85,12500000,145.30,148.00,145.00
        ```
        """)
