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
            gender=request.form.get("gender", "Male"),
            married=request.form.get("married", "Yes"),
            dependents=request.form.get("dependents", "0"),
            education=request.form.get("education", "Graduate"),
            self_employed=request.form.get("self_employed", "No"),
            applicant_income=float(request.form.get("applicant_income", 0)),
            coapplicant_income=float(request.form.get("coapplicant_income", 0)),
            loan_amount=float(request.form.get("loan_amount", 0)),
            loan_amount_term=float(request.form.get("loan_amount_term", 360)),
            credit_history=float(request.form.get("credit_history", 1)),
            property_area=request.form.get("property_area", "Semiurban"),
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
