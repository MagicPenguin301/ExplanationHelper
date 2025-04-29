from anchor.anchor_text import AnchorText
import spacy
import utils
import torch
import streamlit as st

class_names = list(utils.model.config.id2label.values())

nlp = spacy.load("en_core_web_sm")


def predict_lr(text):
    inputs = utils.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(utils.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = utils.model(**inputs)
        preds = outputs.logits.argmax(dim=1)  # <-- pick the class with highest probability
    print(preds.cpu().numpy())
    return preds.cpu().numpy()  # e.g., array([5])



def anchor_explain(text):
    explainer = AnchorText(nlp, class_names, use_unk_distribution=False)
    exp = explainer.explain_instance(text, predict_lr)
    # st.write(exp.names())
    # st.write(exp.precision())
    def show_in_streamlit(explainer: AnchorText, **kwargs):
        out = explainer.as_html(**kwargs)  # same as before
        st.components.v1.html(out, height=600, scrolling=True)
    show_in_streamlit(explainer, exp=exp.exp_map)