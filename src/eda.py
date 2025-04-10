import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import utils

def category_balance(df: pd.DataFrame, column: str):
    category_counts = df[column].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(category_counts.index, category_counts.values, color="skyblue")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Category Balance for {column}")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()


def token_length_distribution(df: pd.DataFrame, column: str):
    df['token_length'] = df[column].apply(lambda x: len(utils.tokenizer.tokenize(x)))
    plt.figure(figsize=(10, 6))
    plt.hist(df['token_length'], bins=50, color='skyblue')
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.title(f"Token Length Distribution for {column}")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

