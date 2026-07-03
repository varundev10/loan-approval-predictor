import os

from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_data():
    if request.method == "GET":
        return render_template("home.html")

    try:
        data = CustomData(
            no_of_dependents=int(request.form.get("no_of_dependents", 0)),
            education=request.form.get("education", "Graduate"),
            self_employed=request.form.get("self_employed", "No"),
            income_annum=float(request.form.get("income_annum", 0)),
            loan_amount=float(request.form.get("loan_amount", 0)),
            loan_term=float(request.form.get("loan_term", 0)),
            cibil_score=float(request.form.get("cibil_score", 0)),
            residential_assets_value=float(request.form.get("residential_assets_value", 0)),
            commercial_assets_value=float(request.form.get("commercial_assets_value", 0)),
            luxury_assets_value=float(request.form.get("luxury_assets_value", 0)),
            bank_asset_value=float(request.form.get("bank_asset_value", 0)),
        )

        pred_df = data.get_data_as_data_frame()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(pred_df)[0]
        result = "Approved" if str(prediction).upper() in {"Y", "1", "APPROVED"} else "Rejected"
        return render_template("home.html", results=result)
    except Exception as exc:
        return render_template("home.html", error=str(exc))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
