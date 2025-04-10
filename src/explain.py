import captum, shap, torch, transformers
from lime.lime_text import LimeTextExplainer
import utils
from matplotlib import pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from expl_eval import EvalForSHAP, EvalForSaliency, EvalForLIME
import pandas as pd
from lime_fix import get_lime_attributions


def show_hint(text, desired_label_i):
    predicted = utils.classifier(text)[0]["label"]

    if predicted != utils.model.config.id2label[desired_label_i]:
        st.info(
            f"The sample text is predicted as **{predicted}**.\n"
            "You may choose it in the label field above."
        )
    else:
        st.success(f"**{predicted}** is exactly the predicton of the model.")
    if utils.model.config.id2label[desired_label_i] != predicted:
        st.write("You may choose it in the label field above.")


def shap_explain(text, infidelity=False):
    try:
        pred = utils.classifier
        explainer = shap.Explainer(pred)
        shap_values = explainer([text])
        shap_html = shap.plots.text(shap_values[0], display=False)
        components.html(shap_html, height=300, scrolling=True)
        return shap_values

    except Exception as e:
        st.error(f"An error occurred during SHAP explanation: {e}")


def lime_explain(text, desired_label_i, infidelity=False):
    def predict_fn(texts):
        # Tokenize the texts
        encodings = utils.tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt"
        )
        outputs = utils.model(**encodings)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs.detach().numpy()

    try:
        show_hint(text, desired_label_i)
        explainer = LimeTextExplainer(
            class_names=[str(i) for i in range(utils.model.config.num_labels)],
            # in order to calculate infidelity we need to disable it.
            feature_selection="none" if infidelity else "auto",
            # but it doesn't work
            bow=False,
        )

        # Explain the instance
        exp = explainer.explain_instance(
            text,
            predict_fn,
            num_features=len(utils.tokenizer.tokenize(text)),
            labels=[desired_label_i],
        )
        # st.write(explainer.feature_selection)
        # Extract feature names and their importance scores
        feature_names = [
            feature for feature, score in exp.as_list(label=desired_label_i)
        ]
        scores = [score for feature, score in exp.as_list(label=desired_label_i)]

        # scores, encoding = get_lime_attributions(text, desired_label_i)
        # feature_names = utils.tokenizer.convert_ids_to_tokens(encoding)
        # Plot the explanation
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, scores, color="skyblue")
        plt.xlabel("Contribution Score")
        label_str = utils.model.config.id2label[desired_label_i]
        plt.title(f"LIME Explanation for Label {label_str}")
        st.pyplot(plt)
        plt.clf()

        return scores
    except Exception as e:
        st.error(f"An error occurred during LIME explanation: {e}")


def saliency_explain(text, desired_label_i):
    # sourcery skip: merge-nested-ifs
    show_hint(text, desired_label_i)

    inputs = utils.tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Ensure input_ids has a batch dimension
    input_ids = input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)
    input_embeddings = utils.model.bert.embeddings(input_ids)
    input_embeddings.requires_grad_()

    def predict(embeddings):
        inputs_embeds = embeddings
        attention_mask = inputs.get("attention_mask")
        token_type_ids = inputs.get("token_type_ids")
        return utils.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits

    from captum.attr import Saliency

    saliency = Saliency(predict)
    attributions = saliency.attribute(input_embeddings, target=desired_label_i)

    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    tokens = utils.tokenizer.convert_ids_to_tokens(input_ids[0])

    # no [CLS] and [SEP]
    tokens = tokens[1:-1]
    attributions = attributions[1:-1]

    # token_attributions = list(zip(tokens, attributions.tolist()))
    # st.write("Token Attributions:", token_attributions)
    def visualize_saliency_graph(tokens, attributions):
        plt.rcParams.update({"font.size": 10})
        fig, ax = plt.subplots(
            figsize=(8, len(tokens) * 0.6), dpi=150
        )  # Adjust figure size (width, height)

        normalized_attributions = (attributions - attributions.min()) / (
            attributions.max() - attributions.min()
        )

        colors = plt.cm.viridis(normalized_attributions)

        ax.barh(
            np.arange(len(tokens)),
            normalized_attributions,
            color=colors,
            align="center",
        )  # Use barh
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)  # Tokens on y-axis
        ax.set_xlabel("Normalized Saliency Score", fontsize=10)  # Score on x-axis
        ax.set_title(
            f"Saliency Map ({utils.model.config.id2label[desired_label_i]})",
            fontsize=12,
        )
        ax.invert_yaxis()  # To display the first token at the top
        plt.tight_layout()
        st.pyplot(fig)

    visualize_saliency_graph(tokens, attributions)

    return attributions


def explain(text: str, approach: str, label_i, infidelity=False):
    # st.write(type(approach))
    evals = {}
    match approach:
        case "SHAP":
            shap_values = shap_explain(text)
            if infidelity:
                evals["infidelity"] = [
                    EvalForSHAP.compute_infidelity(text, shap_values)
                ]
        case "LIME":
            scores = lime_explain(text, label_i, infidelity)
            if infidelity:
                st.info(
                    "Sorry, I haven't figured out how to correctly implement "
                    "evaluations for LIME because "
                    "`feature_selection` in `LimeTextExplainer` doesn't work as intended. "
                    "I'm trying to implement a LIME for text classification myself. See `lime_fix.py` "
                )
                # evals["infidelity"] = [
                #     EvalForLIME.compute_infidelity(text, scores, label_i)
                # ]
        case "Saliency":
            attrs = saliency_explain(text, label_i)
            if infidelity:
                evals["infidelity"] = [
                    EvalForSaliency.compute_infidelity(text, attrs, label_i)
                ]
        case _:
            st.error(f"Unexpected explanation approach: {approach}")

    if evals:
        df = pd.DataFrame(evals)
        st.write(df, hide_index=True)
