import os
import json
import torch
import timm
import io
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# --- 配置 ---
MODEL_PATH = 'weights/best_plantnet_model.pt'
MAPPING_PATH = 'weights/id_mapping.json'
NAMES_PATH = 'data/plantnet300K_species_names.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 加载数据与模型 ---
with open(MAPPING_PATH, 'r') as f:
    sid_to_idx = json.load(f)
    idx_to_sid = {int(v): k for k, v in sid_to_idx.items()}

with open(NAMES_PATH, 'r') as f:
    species_names = json.load(f)

num_classes = len(sid_to_idx)
model = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '未收到文件'})
    
    file = request.files['file']
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            # 计算 Softmax 概率
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            # 获取 Top-5 结果
            top5_prob, top5_indices = torch.topk(probabilities, 5)

        results = []
        for i in range(5):
            prob = top5_prob[i].item()
            idx = top5_indices[i].item()
            sid = idx_to_sid[idx]
            name = species_names.get(sid, f"未知 ID: {sid}")
            results.append({
                'name': name,
                'confidence': f"{prob * 100:.2f}%",
                'rank': i + 1
            })

        return jsonify({'success': True, 'predictions': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)