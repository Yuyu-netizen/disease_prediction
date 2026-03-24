"""
疾病预测应用程序
根据用户输入的特征，使用预训练 XGBoost 模型预测是否有病
"""
import xgboost as xgb
import joblib
import pandas as pd
from pathlib import Path

# 特征列名（需与训练时一致）
FEATURE_COLUMNS = [
    "Gender_discrete",
    "Age_continuous",
    "BMI_continuous",
    "blood_continuous",
    "exposure.RC_continuous",
]

# Gender 取值映射：Female=0, Male=1
GENDER_MAP = {"female": 0.0, "male": 1.0, "0": 0.0, "1": 1.0}

MODEL_PATH = Path(__file__).parent / "xgboost_predict.json"
CT_TRANSFORM_PATH = Path(__file__).parent / "ct_transform.joblib"


def load_model() -> xgb.XGBClassifier:
    """加载预训练模型"""
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def load_ct_transform():
    """加载 ct 特征归一化器"""
    return joblib.load(CT_TRANSFORM_PATH)



def predict(model: xgb.XGBClassifier, ct_transform, features: dict) -> dict:
    """
    根据输入特征进行预测（先经 ct.transform 归一化）
    :param model: 加载的 XGBoost 模型
    :param ct_transform: 加载的 ct 归一化器
    :param features: 特征字典，包含各特征值（Gender 可为 Female/Male 或 0/1）
    :return: {"has_disease": bool, "probability": float}
    """
    X = pd.DataFrame([{k: features[k] for k in FEATURE_COLUMNS}])
    print(X)
    X_transformed = ct_transform.transform(X)
    print(X_transformed)
    pred = model.predict(X_transformed)[0]
    proba = model.predict_proba(X_transformed)[0]
    return {
        "has_disease": bool(pred),
        "probability": float(proba[1]),
        "label": "有病" if pred else "无病",
    }


def get_user_input() -> dict:
    """从命令行获取用户输入"""
    print("=" * 50)
    print("疾病预测系统 - 请输入以下特征")
    print("=" * 50)
    features = {}
    g = input("性别 (Female/Male): ").strip() or "Female"
    features["Gender_discrete"] = GENDER_MAP.get(g.lower(), 0.0)
    features["Age_continuous"] = float(
        input("年龄 (连续值, 如 45): ").strip() or "40")
    features["BMI_continuous"] = float(
        input("BMI 体重指数 (如 24.5): ").strip() or "24")
    features["blood_continuous"] = float(
        input("血压相关指标 (如 120): ").strip() or "120")
    features["exposure.RC_continuous"] = float(
        input("暴露指数 exposure.RC_continuous (如 3.0): ").strip() or "3")
    return features


def main():
    model_path = Path(MODEL_PATH)
    ct_path = Path(CT_TRANSFORM_PATH)
    if not model_path.exists():
        print("未找到预训练模型，请先运行: python train_model.py")
        return
    if not ct_path.exists():
        print("未找到 ct.transform，请先运行: python train_model.py")
        return

    model = load_model()
    ct_transform = load_ct_transform()
    features = get_user_input()
    result = predict(model, ct_transform, features)

    print("\n" + "=" * 50)
    print("预测结果")
    print("=" * 50)
    print(f"预测: {result['label']}")
    print(f"患病概率: {result['probability']:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
