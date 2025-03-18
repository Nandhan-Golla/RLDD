import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import re
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pubchempy as pcp
import matplotlib.pyplot as plt
import PyPDF2
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import logging
from torchvision import models, transforms

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force CPU explicitly
device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Flask app setup
app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Custom feature extractor (simplified for debugging)
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=256)
        try:
            logger.info("Initializing CustomFeatureExtractor...")
            # Load BERT without pre-trained weights as a fallback
            self.bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=False).to(device)
            self.bert.eval()
            logger.info("BERT initialized.")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            logger.info("Tokenizer initialized.")
            # Load ResNet without pre-trained weights as a fallback
            self.resnet = models.resnet18(pretrained=False).to(device)
            self.resnet.fc = torch.nn.Identity()
            self.resnet.eval()
            logger.info("ResNet initialized.")
            self.image_transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info("Image transform initialized.")
            # Transformer initialization (where error likely occurs)
            self.transformer = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(d_model=774, nhead=6, dim_feedforward=512), num_layers=2
            ).to(device)
            logger.info("Transformer initialized.")
            self.fc = torch.nn.Linear(774, 256).to(device)
            logger.info("Fully connected layer initialized.")
        except Exception as e:
            logger.error(f"Error in CustomFeatureExtractor init: {str(e)}", exc_info=True)
            raise

    def forward(self, observations):
        try:
            batch_size = observations.shape[0]
            bert_input = observations[:, :-4].reshape(batch_size, 1, 768)
            extra_features = observations[:, -4:].to(device)
            img = torch.zeros(224, 224, 3, dtype=torch.uint8)  # Placeholder image
            img_tensor = self.image_transform(img).unsqueeze(0).to(device).repeat(batch_size, 1, 1, 1)
            with torch.no_grad():
                img_features = self.resnet(img_tensor).reshape(batch_size, 1, -1)[:, :, -2:]
            combined = torch.cat((bert_input, extra_features.unsqueeze(1), img_features), dim=2)
            transformer_out = self.transformer(combined.transpose(0, 1)).transpose(0, 1)
            return self.fc(transformer_out.squeeze(1))
        except Exception as e:
            logger.error(f"Error in CustomFeatureExtractor forward: {str(e)}", exc_info=True)
            raise

# Data pipeline
class DataPipeline:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.bert_model.eval()

    def process_blood_report(self, report):
        inputs = self.tokenizer(report, return_tensors='pt', truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.pooler_output.squeeze().cpu().numpy()

    def extract_patient_data(self, file_path):
        text = self.read_file(file_path)
        try:
            hemoglobin = float(re.search(r"Hemoglobin:\s*(\d+\.\d+)", text).group(1))
            glucose = float(re.search(r"Glucose:\s*(\d+\.\d+)", text).group(1))
            age = float(re.search(r"Age:\s*(\d+)", text).group(1))
            weight = float(re.search(r"Weight:\s*(\d+\.\d+)", text).group(1))
            return text, hemoglobin, glucose, age, weight
        except AttributeError:
            raise ValueError("File missing required data: Hemoglobin, Glucose, Age, Weight")

    def read_file(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                return f.read()
        elif file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        else:
            raise ValueError("Unsupported file format. Use .txt or .pdf")

# Load TDC dataset
def load_tdc_data():
    return [
        "CCO", "CCN", "CCC", "C=O", "C#N", "CC=O", "CC#N", "COC", "CCOC", "CN",
        "CNC", "CCNC", "C(=O)O", "CC(=O)O", "C#CC", "CC#CC", "COC=O", "CCOC=O",
        "CN(C)C", "CCN(C)C"
    ]

# Decode SMILES with confidence
def decode_smiles(smiles, model, state):
    mol = Chem.MolFromSmiles(smiles)
    mol_formula = rdMolDescriptors.CalcMolFormula(mol) if mol else "Invalid SMILES"
    compounds = pcp.get_compounds(smiles, 'smiles')
    drug_name = compounds[0].iupac_name if compounds else "Unknown"
    with torch.no_grad():
        logits, _ = model.policy.predict_values(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
        confidence = torch.softmax(logits, dim=-1).max().item()
    return drug_name, mol_formula, confidence

# Visualization
def plot_patient_trajectory(history, rewards, filename="static/patient_trajectory.png"):
    hgb_history, glu_history = zip(*history)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(hgb_history, label="Hemoglobin")
    plt.plot(glu_history, label="Glucose")
    plt.axhspan(12, 16, alpha=0.2, color='green', label="Hgb Ideal")
    plt.axhspan(70, 110, alpha=0.2, color='blue', label="Glu Ideal")
    plt.legend()
    plt.title("Patient State Trajectory")
    plt.subplot(2, 1, 2)
    plt.plot(rewards, label="Reward")
    plt.legend()
    plt.title("Reward Over Time")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Flask routes
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.info(f"File uploaded: {filename}")

            pipeline = DataPipeline()
            try:
                blood_report, hemoglobin, glucose, age, weight = pipeline.extract_patient_data(file_path)
                blood_features = pipeline.process_blood_report(blood_report)
                new_state = np.concatenate([blood_features, [hemoglobin, glucose, age, weight]]).astype(np.float32)

                logger.info("Loading PPO model...")
                try:
                    model = PPO.load("ppo_luminarix_advanced.zip", device=device, map_location=device, custom_objects={
                        "policy_kwargs": {"features_extractor_class": CustomFeatureExtractor}
                    })
                    logger.info("Model loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load model: {str(e)}", exc_info=True)
                    raise

                state = new_state
                history = [(hemoglobin, glucose)]
                rewards = []
                steps = []
                for step in range(5):
                    action, _ = model.predict(state, deterministic=True)
                    drug_effects = [
                        [-0.5, -5.0, 0.1], [0.5, -3.0, 0.2], [-1.0, 2.0, 0.3], [1.0, 0.0, 0.1],
                        [0.0, -10.0, 0.4], [-0.2, 5.0, 0.2], [0.8, -2.0, 0.1], [-0.8, -1.0, 0.3],
                        [0.3, 3.0, 0.2], [0.0, 0.0, 0.0], [1.2, -4.0, 0.5], [-0.7, 6.0, 0.3],
                        [0.4, -1.5, 0.1], [-1.5, 3.0, 0.4], [0.6, -2.5, 0.2], [-0.3, 4.0, 0.3],
                        [0.9, -3.5, 0.1], [-0.9, 1.0, 0.2], [0.2, 2.0, 0.3], [-0.4, -6.0, 0.4]
                    ]
                    hemoglobin += drug_effects[action][0] * (weight / 70)
                    glucose += drug_effects[action][1] * (weight / 70)
                    side_effect = drug_effects[action][2] * (age / 50)
                    hemoglobin_error = max(0, 12 - hemoglobin) + max(0, hemoglobin - 16)
                    glucose_error = max(0, 70 - glucose) + max(0, glucose - 110)
                    reward = -(hemoglobin_error + glucose_error) - side_effect
                    if reward > -0.5:
                        reward += 1.0
                    state[-4:] = [hemoglobin, glucose, age, weight]
                    history.append((hemoglobin, glucose))
                    rewards.append(reward)
                    steps.append(f"Step {step+1}: Drug {action}, Reward {reward:.2f}, Hgb {hemoglobin:.1f}, Glu {glucose:.1f}")
                    if reward > 0:
                        break
                
                drug_list = load_tdc_data()
                recommended_drug_smiles = drug_list[action]
                drug_name, mol_formula, confidence = decode_smiles(recommended_drug_smiles, model, new_state)
                plot_patient_trajectory(history, rewards)

                return render_template('result.html', 
                                       steps=steps,
                                       smiles=recommended_drug_smiles,
                                       drug_name=drug_name,
                                       formula=mol_formula,
                                       confidence=confidence,
                                       plot_url='/static/patient_trajectory.png')
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}", exc_info=True)
                flash(f"Error: {str(e)}")
                return redirect(request.url)
    return render_template('upload.html')

@app.route('/static/<path:filename>')
def serve_plot(filename):
    return send_file(os.path.join('static', filename))

if __name__ == "__main__":
    os.makedirs('static', exist_ok=True)
    model_path = "ppo_luminarix_advanced.zip"
    if not os.path.exists(model_path):
        logger.error(f"Model file '{model_path}' not found. Run train_luminarix.py first.")
        print(f"Error: Model file '{model_path}' not found. Run train_luminarix.py first.")
    else:
        logger.info("Starting Flask server...")
        app.run(debug=True)