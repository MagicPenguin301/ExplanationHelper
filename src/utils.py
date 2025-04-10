import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml, streamlit as st, torch
import transformers

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

tokenizer = None
model =  None
classifier = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_data
def read_data(file_path):
    # For simplicity, the used field names are hard-coded here.
    # Alter this function to fit a custom dataset. 
    data = pd.read_json(file_path, encoding="utf-8", lines=True)
    data["concatenated"] = data.apply(
        (
            lambda row: f"{row['headline']}\n{row['short_description']}"
        ),
        axis=1
    )
    return data["concatenated"].to_list(), data["category"].to_list()


def init_model():
    global model, tokenizer, classifier
    path = config["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    classifier = transformers.pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,
    return_all_scores=False
    )

def after_model_loaded(func):
    def wrapper(*args, **kwargs):
        if not model or not tokenizer:
            st.warning("The model has not been loaded!")
            st.stop()
        return func(*args, **kwargs)
    return wrapper

if __name__ == "__main__":
    # only for debug
    data = read_data(r"data\News_Category_Dataset_v3.json")
    print(data.head(2)[["headline", "short_description"]])

# write files
