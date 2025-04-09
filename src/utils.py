import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

tokenizer = None
model =  None

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
    global model, tokenizer
    path = (
    config["model"]["hf_path"]
    if config["model"]["from_hf"]
    else config["model"]["local_path"]
    )
    # by default, it's "Yueh-Huan/news-category-classification-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)

if __name__ == "__main__":
    # only for debug
    data = read_data(r"data\News_Category_Dataset_v3.json")
    print(data.head(2)[["headline", "short_description"]])

# write files
