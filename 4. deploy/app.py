# app.py
import os
from flask import Flask, request, render_template, send_file
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
from grad_cam_utils import generate_grad_cam

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("resnet50_model.pth", map_location=device))
model.to(device)
model.eval()

# Class labels
class_names = ["Arjun Diseased", "Arjun Healthy", "Pomegranate Diseased", "Pomegranate Healthy"]

# Transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part!"
    file = request.files['file']
    if file.filename == '':
        return "No selected file!"
    
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]

    # Grad-CAM if diseased
    if "Diseased" in label:
        cam_path = generate_grad_cam(model, input_tensor, predicted.item(), image)
        return render_template('index.html', prediction=label, cam_path=cam_path)
    else:
        return render_template('index.html', prediction=label, cam_path=None)

@app.route('/cam')
def cam():
    return send_file("static/cam_output.jpg", mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
