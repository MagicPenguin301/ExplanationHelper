import streamlit as st
import utils
import pandas as pd
import eda

@utils.after_model_loaded
def run():
    if f := st.file_uploader("Upload a dataset."):
        df = pd.read_json(f, lines=True)
        text_col, cat_col = None, None
        try:
            text_col = utils.config["eda"]["text_col"]
            cat_col = utils.config["eda"]["cat_col"]
        except Exception as e:
            st.warning(f"Failed to read config {e}")
            text_col = "text"
            cat_col = "category"
        eda.token_length_distribution(df, text_col)
        eda.category_balance(df, cat_col)
            
    

if __name__ == "__main__":
    st.title("EDA")
    run()    