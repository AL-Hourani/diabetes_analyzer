import torch
import joblib
import torch.nn as nn

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


model_states = torch.load("multi_models.pt", map_location="cpu")
scalers = joblib.load("multi_scalers.pkl")

labels = ["أقل من الحد الطبيعي", "قيمة طبيعية نوعا ما ", "أعلى من الحد الطبيعي"]

def predict_dynamic(input_dict):
    results = {}

    for test, value in input_dict.items():
        if test not in model_states:
            results[test] = "غير مدعوم"
            continue

        model = LabTestClassifier()
        model.load_state_dict(model_states[test])
        model.eval()

        scaler = scalers[test]
        try:
            x_scaled = scaler.transform([[float(value)]])
        except:
            results[test] = "قيمة غير صالحة"
            continue
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            out = model(x_tensor)
            pred = torch.argmax(out, dim=1).item()
            results[test] = labels[pred]

    return results
