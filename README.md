# 疾病预测应用程序

基于 XGBoost 的疾病预测应用，根据用户输入的特征预测是否有病。

## 特征说明

| 特征名 | 说明 | 示例 |
|--------|------|------|
| Gender_discrete | 性别 (0=女, 1=男) | 0 或 1 |
| Age_continuous | 年龄 | 45 |
| BMI_continuous | 体重指数 | 24.5 |
| blood_continuous | 血压相关指标 | 120 |
| exposure.RC_continuous | 暴露指数 | 3.0 |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用步骤

### 1. 训练/生成模型（若尚无预训练模型）

```bash
python train_model.py
```

将生成 `xgboost_model.json` 文件。

### 2. 使用已有模型进行预测

若你已有训练好的 XGBoost 模型和 ct 归一化器：
- 将模型保存为 `xgboost_model.json`
- 将 ct.transform 保存为 `ct_transform.joblib`（使用 `joblib.dump(ct, "ct_transform.joblib")`）

放在项目根目录，或修改 `predict.py` 中的 `MODEL_PATH` 和 `CT_TRANSFORM_PATH`。

### 3. 运行预测

**命令行方式：**
```bash
python predict.py
```
按提示输入各项特征即可得到预测结果。

**Web 前端方式：**
```bash
python app.py
```
在浏览器打开 http://127.0.0.1:5000 ，在页面中填写特征值并点击「预测」。

## 程序化调用

```python
from predict import load_model, predict

model = load_model()
features = {
    "Gender_discrete": 1,
    "Age_continuous": 55,
    "BMI_continuous": 28.5,
    "blood_continuous": 145,
    "exposure_RC_continuous": 6.0,
}
result = predict(model, features)
print(result)  # {"has_disease": True/False, "probability": 0.xx, "label": "有病"/"无病"}
```
