import torch
import torch.nn as nn
import torch.optim as optim
import ipywidgets as widgets
from IPython.display import display


data = [
    [60, 0.1],  
    [70, 0.2],  
    [80, 0.4],  
    [90, 0.5],  
    [100, 0.7], 
    [110, 0.9],  
]

X = torch.tensor([[x[0]] for x in data], dtype=torch.float32)  
y = torch.tensor([[x[1]] for x in data], dtype=torch.float32)  


class HeartAttackRiskModel(nn.Module):
    def __init__(self): 
        super(HeartAttackRiskModel, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.output(x))  
        return x


model = HeartAttackRiskModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


epochs = 500
for epoch in range(epochs):
    model.train()
    
   
    outputs = model(X)
    loss = criterion(outputs, y)
    
   
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


def predict_heart_attack_risk(heart_rate):
    model.eval()
    heart_rate_tensor = torch.tensor([[heart_rate]], dtype=torch.float32)
    with torch.no_grad():
        risk = model(heart_rate_tensor).item()
    return risk


def suggest_health_improvements(heart_rate):
    risk = predict_heart_attack_risk(heart_rate)
    risk_percentage = risk * 100
    
    if risk < 0.3:
        exercise = "Light aerobic exercises like walking, cycling, and swimming are recommended."
        food = "Eat a balanced diet with plenty of fruits, vegetables, and whole grains. Avoid high-fat, high-sodium foods."
    elif 0.3 <= risk < 0.6:
        exercise = "Moderate-intensity exercises such as jogging, brisk walking, and resistance training are beneficial."
        food = "Include lean proteins, such as chicken and fish, and reduce intake of processed foods and sugars."
    else:
        exercise = "Consult a healthcare professional before starting any exercise regime. Gentle activities like yoga and stretching are helpful."
        food = "Follow a heart-healthy diet rich in omega-3 fatty acids, whole grains, and healthy fats. Avoid red meat and trans fats."
    
    return f"Heart Attack Risk for {heart_rate} bpm: {risk_percentage:.2f}%\n\nRecommended Exercises: {exercise}\nRecommended Foods: {food}"


heart_rate_input = widgets.FloatText(value=70, description='Heart Rate (bpm):', step=1, style={'description_width': 'initial'})
output_text = widgets.Output()


def on_button_click(b):
    with output_text:
        output_text.clear_output()
        heart_rate = heart_rate_input.value
        result = suggest_health_improvements(heart_rate)
        print(result)


button = widgets.Button(description="Check Risk")
button.on_click(on_button_click)


title = widgets.HTML(value="<h2>Heart Attack Risk Predictor</h2>")
description = widgets.HTML(value="<p>Enter your heart rate (in bpm) to get an estimate of your risk for a heart attack and receive health improvement suggestions.</p>")


display(title, description, heart_rate_input, button, output_text)
