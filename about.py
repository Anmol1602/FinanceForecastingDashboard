import streamlit as st

def main():
    st.title("ℹ️ About the Financial Forecasting Dashboard")
    st.markdown("---")
    
    st.subheader("🔍 How LSTM Models Work")
    st.write("""
    - LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) capable of learning long-term dependencies.
    - Our models take past stock, crypto, and forex prices to predict future trends.
    - We normalize the data, pass it through LSTM layers, and optimize using Adam optimizer.
    """)

    st.subheader("📊 Dataset Details")
    st.write("""
    - Data is collected from various financial sources and consists of historical prices of stocks, cryptocurrencies, and forex pairs.
    - Data preprocessing involves scaling using MinMaxScaler and creating input sequences for LSTM training.
    """)

    st.subheader("📈 Model Performance")
    st.write("The models were trained and evaluated using loss metrics:")
    
    st.markdown("""
    - **Stock Model**  
      - Final loss: `0.0063`  
      - Validation loss: `0.0087`
    - **Crypto Model**  
      - Final loss: `0.0067`  
      - Validation loss: `0.0116`
    - **Forex Model**  
      - Final loss: `0.0029`  
      - Validation loss: `0.0006`
    """)

if __name__ == "__main__":
    main()
