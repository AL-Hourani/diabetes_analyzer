# app/utils.py

import torch
import pandas as pd
import joblib
from app.model import MultiLabClassifier

reference_ranges = {
    "glucose": (70, 99),
    "ldl": (0, 100),
    "triglycerides": (0, 150),
    "hdl": (40, 60),
    "cholesterol": (125, 200),
    "creatinine": (0.6, 1.3),
    "bun": (7, 20),
    "hemoglobin": (13.5, 17.5),
    "wbc": (4.5, 11.0),
    "rbc": (4.7, 6.1),
    "platelets": (150, 450),
}

# تحميل scaler
scaler = joblib.load("scaler.pkl")

# تحميل النموذج
input_size = len(reference_ranges)
output_size = len(reference_ranges)

model = MultiLabClassifier(input_size, output_size)
model.load_state_dict(torch.load("lab_model.pt", map_location=torch.device("cpu")))
model.eval()

def interpret_prediction(pred_tensor):
    labels = ["أقل من الطبيعي", "طبيعي", "أعلى من الطبيعي"]
    return [labels[p.argmax().item()] for p in pred_tensor]

def predict_analysis(input_dict):
    # استخدم فقط الخصائص المدعومة والمرتبطة بالنموذج
    valid_features = [key for key in input_dict if key in reference_ranges]
    
    # إذا لم يدخل المستخدم أي قيم صحيحة، أعطه رسالة
    if not valid_features:
        return {"error": "لم يتم إدخال أي تحاليل مدعومة."}
    
    # جهز فقط الخصائص المدخلة
    x_partial = {key: input_dict[key] for key in valid_features}

    # اكمل باقي الخصائص بـ 0
    full_input = {key: x_partial.get(key, 0) for key in reference_ranges}

    x_df = pd.DataFrame([full_input])
    x_scaled = scaler.transform(x_df)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        out = model(x_tensor)

    predictions = interpret_prediction(out[0])

    # أرجع فقط التوقعات التي طلبها المستخدم
    result = {
        test: pred
        for test, pred in zip(reference_ranges.keys(), predictions)
        if test in input_dict
    }
    return result

