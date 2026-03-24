"""
疾病预测 Web 应用
提供前端页面供用户填写特征值并获取预测结果
"""
from pathlib import Path
from flask import Flask, render_template, request, jsonify

from predict import (
    load_model,
    load_ct_transform,
    predict,
    MODEL_PATH,
    CT_TRANSFORM_PATH,
    FEATURE_COLUMNS,
)

app = Flask(__name__)

_model = None
_ct_transform = None


def get_model_and_ct():
    global _model, _ct_transform
    if _model is None and MODEL_PATH.exists():
        _model = load_model()
    if _ct_transform is None and CT_TRANSFORM_PATH.exists():
        _ct_transform = load_ct_transform()
    return _model, _ct_transform


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    model, ct_transform = get_model_and_ct()
    if model is None:
        return jsonify({"error": "模型未加载，请先运行 python train_model.py"}), 500
    if ct_transform is None:
        return jsonify({"error": "ct.transform 未加载，请先运行 python train_model.py"}), 500

    try:
        data = dict(request.get_json() or request.form)
        # 将 exposure_RC_continuous 映射为 exposure.RC_continuous
        if "exposure_RC_continuous" in data and "exposure.RC_continuous" not in data:
            data["exposure.RC_continuous"] = data.pop("exposure_RC_continuous")
        features = {k: v for k, v in data.items() if k in FEATURE_COLUMNS}
        print(features)
        print("====")
        print(len(features), len(FEATURE_COLUMNS))
        if len(features) != len(FEATURE_COLUMNS):
            missing = set(FEATURE_COLUMNS) - set(features.keys())
            return jsonify({"error": f"缺少特征: {missing}"}), 400
        result = predict(model, ct_transform, features)
        return jsonify(result)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
