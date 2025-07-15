


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib 



reference_ranges = {
    "glucose": (70, 99),
    "urea": (7, 20),  # أو BUN
    "estrogen": (15, 350),  # تختلف حسب الجنس ووقت الدورة الشهرية، هذا متوسط عام
    "creatinine": (0.6, 1.3),
    "calcium": (8.6, 10.2),
    "total_cholesterol": (125, 200),
    "lipoprotein": (0, 30),  # Lipoprotein(a), يُفضل أقل من 30 mg/dL
    "albumin": (3.4, 5.4),
    "bilirubin": (0.1, 1.2),
    "potassium": (3.5, 5.1),
    "ph": (7.35, 7.45),  # الدم
    "alt": (7, 56),
    "ast": (10, 40),
    "magnesium": (1.7, 2.2),
    "ldl": (0, 100),
    "triglycerides": (0, 150),
    "hemoglobin": (13.5, 17.5),
    "hdl": (40, 60),
    "cholesterol": (125, 200),  # Total cholesterol (مرادف لـ total_cholesterol)
    "bun": (7, 20),  # مرادف لـ urea
    "wbc": (4.5, 11.0),
    "rbc": (4.7, 6.1),
    "platelets": (150, 450),
    "uric_acid": (3.5, 7.2),
}


class LabTestClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, 3)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


models = {}
scalers = {}


samples = []
labels = []

for _ in range(1000):
    sample = {}
    label = {}

    for test, (low, high) in reference_ranges.items():
        value = np.random.uniform(low * 0.5, high * 1.5)
        sample[test] = value

        if value < low:
            label[test] = 0
        elif value > high:
            label[test] = 2
        else:
            label[test] = 1

    samples.append(sample)
    labels.append(label)

df_X = pd.DataFrame(samples)
df_y = pd.DataFrame(labels)


for test_name in reference_ranges:
    X = df_X[[test_name]].values
    y = df_y[[test_name]].values.ravel()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = LabTestClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(300):
        optimizer.zero_grad()
        out = model(x_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()

    models[test_name] = model
    scalers[test_name] = scaler


torch.save({k: v.state_dict() for k, v in models.items()}, "multi_models.pt")
joblib.dump(scalers, "multi_scalers.pkl")



def predict_dynamic(input_dict):
    labels = ["أقل", "طبيعي", "أعلى"]

    model_states = torch.load("multi_models.pt", map_location="cpu")
    scalers = joblib.load("multi_scalers.pkl")

    results = {}

    for test, value in input_dict.items():
        if test not in model_states:
            results[test] = "غير مدعوم"
            continue

        # تحميل النموذج
        model = LabTestClassifier()
        model.load_state_dict(model_states[test])
        model.eval()

        # تحجيم وتوقع
        scaler = scalers[test]
        x_scaled = scaler.transform([[value]])
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            out = model(x_tensor)
            pred = torch.argmax(out, dim=1).item()
            results[test] = labels[pred]

    return results


if __name__ == "__main__":
    sample_input = {
        "glucose": 85,
        "ldl": 130,
        "hdl": 50,
        "unknown_test": 10  
    }

    prediction = predict_dynamic(sample_input)
    print("Prediction:", prediction)