import captum, shap, lime, torch, transformers
import utils
from matplotlib import pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

def shap_explain(text):
    try:
        pred = utils.classifier
        explainer = shap.Explainer(pred)
        shap_values = explainer([text])
        shap_html = shap.plots.text(shap_values[0], display=False)
        components.html(shap_html, height=300, scrolling=True)

    except Exception as e:
        st.error(f"An error occurred during SHAP explanation: {e}")


def lime_explain(text):
    try:
        from lime.lime_text import LimeTextExplainer
        import matplotlib.pyplot as plt

        # Create a prediction function for LIME
        def predict_proba(texts):
            inputs = utils.tokenizer(
                texts, padding=True, return_tensors="pt"
            )  # use utils.tokenizer
            with torch.no_grad():
                outputs = utils.model(**inputs)  # use utils.model
            probas = torch.softmax(outputs.logits, dim=-1).numpy()
            return probas

        explainer = LimeTextExplainer(
            class_names=["Negative", "Positive"]
        )  # Replace with your class names
        exp = explainer.explain_instance(text, predict_proba, num_features=42)

        # Extract feature names and scores for visualization
        feature_names = [feature for feature, score in exp.as_list()]
        scores = [score for feature, score in exp.as_list()]

        # Create a bar chart
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, scores, color="skyblue")
        plt.xlabel("Contribution Score")
        plt.title("LIME Explanation")
        st.pyplot(plt)
        plt.clf()

    except Exception as e:
        st.error(f"An error occurred during LIME explanation: {e}")


def ig_explain(text):
    """Explain the given text using Integrated Gradients and visualize the results.

    This function uses the Integrated Gradients explainer from the captum library to explain the predictions of a model on the given text and then visualizes the attributions.

    Args:
        text (str): The text to explain.

    """
    try:
        # Tokenize the input text and convert to tensor
        input_ids = utils.tokenizer(text, return_tensors="pt")["input_ids"]

        # Create a baseline tensor (e.g., embedding of zero tokens)
        baseline = utils.tokenizer("", return_tensors="pt")["input_ids"]

        # Initialize the Integrated Gradients explainer with the model
        ig = captum.attr.IntegratedGradients(utils.model)

        # Get the integrated gradients attributions
        attributions = ig.attribute(input_ids, baselines=baseline)

        # Convert the input text to a list of tokens
        tokenized_text = utils.tokenizer.tokenize(text)

        # Convert attributions to numpy array and take the absolute value for visualization
        attr_numpy = attributions.detach().numpy().squeeze(0)
        abs_attr = abs(attr_numpy)

        # Visualize the attributions using a bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(tokenized_text, abs_attr)
        plt.xlabel("Tokens")
        plt.ylabel("Attribution Magnitude")
        plt.title("Integrated Gradients Explanation")
        plt.xticks(
            rotation=45, ha="right"
        )  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        st.pyplot(plt)
        plt.clf()

    except Exception as e:
        st.error(f"An error occurred during Integrated Gradients explanation: {e}")


def explain(text: str, approach: str):
    # st.write(type(approach))
    match approach:
        case "SHAP":
            shap_explain(text)
        case "LIME":
            lime_explain(text)
        case "Integrated Gradients":
            ig_explain(text)
        case _:
            pass
