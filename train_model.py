"""
训练 XGBoost 模型用于疾病预测
特征: Gender_discrete, Age_continuous, BMI_continuous, blood_continuous, exposure.RC_continuous
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# 特征列名
FEATURE_COLUMNS = [
    "Gender_discrete",
    "Age_continuous",
    "BMI_continuous",
    "blood_continuous",
    "exposure.RC_continuous",
]

MODEL_PATH = Path(__file__).parent / "xgboost_predict.json"
CT_TRANSFORM_PATH = Path(__file__).parent / "ct_transform.joblib"


def generate_training_data(n_samples: int = 1000) -> tuple[pd.DataFrame, pd.Series]:
    """生成用于训练的合成数据（若无真实数据，可替换为加载实际CSV）"""
    np.random.seed(42)
    X = pd.DataFrame({
        "Gender_discrete": np.random.randint(0, 2, n_samples),
        "Age_continuous": np.random.uniform(18, 80, n_samples),
        "BMI_continuous": np.random.uniform(15, 40, n_samples),
        "blood_continuous": np.random.uniform(70, 180, n_samples),
        "exposure.RC_continuous": np.random.uniform(0, 10, n_samples),
    })
    # 简化的目标：基于规则合成二分类标签
    y = (
        (X["Age_continuous"] > 50) & (X["BMI_continuous"] > 28) |
        (X["blood_continuous"] > 140) & (X["exposure.RC_continuous"] > 5)
    ).astype(int)
    # 添加噪声
    flip = np.random.random(n_samples) < 0.1
    y[flip] = 1 - y[flip]
    return X, y


def train_and_save(
    model_path: Path = MODEL_PATH,
    ct_path: Path = CT_TRANSFORM_PATH,
) -> tuple[xgb.XGBClassifier, ColumnTransformer]:
    """训练模型并保存（含 ct 特征归一化器）"""
    print("生成训练数据...")
    X, y = generate_training_data(2000)

    print("拟合并保存 ct 特征归一化器...")
    ct = ColumnTransformer(
        [("scale", StandardScaler(), FEATURE_COLUMNS)],
        remainder="passthrough",
    )
    X_transformed = ct.fit_transform(X)

    print("训练 XGBoost 模型...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_transformed, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    joblib.dump(ct, ct_path)
    print(f"模型已保存至: {model_path}")
    print(f"ct.transform 已保存至: {ct_path}")

    return model, ct


if __name__ == "__main__":
    train_and_save()
