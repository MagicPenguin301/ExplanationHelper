import captum


def explain(text: str, approach: str):
    def shap_explain(text):
        pass

    def lime_explain(text):
        pass

    def ig_explain(text):
        pass

    match approach:
        case "SHAP":
            shap_explain(text)
        case "LIME":
            lime_explain(text)
        case "Integrated Gradients":
            ig_explain(text)
        case _:
            pass
