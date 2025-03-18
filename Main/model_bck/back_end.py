import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import re
from stable_baselines3 import PPO
import os
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pubchempy as pcp
import matplotlib.pyplot as plt
import PyPDF2
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Flask app setup
app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Data pipeline
class DataPipeline:
    def __init__(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
            self.bert_model.eval()
            logger.info("BERT model initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing BERT: {str(e)}", exc_info=True)
            raise

    def process_blood_report(self, report):
        try:
            inputs = self.tokenizer(report, return_tensors='pt', truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            return outputs.pooler_output.squeeze().cpu().numpy()
        except Exception as e:
            logger.error(f"Error processing blood report: {str(e)}", exc_info=True)
            raise

    def extract_patient_data(self, file_path):
        text = self.read_file(file_path)
        try:
            hemoglobin = float(re.search(r"Hemoglobin:\s*(\d+\.\d+)", text).group(1))
            glucose = float(re.search(r"Glucose:\s*(\d+\.\d+)", text).group(1))
            age = float(re.search(r"Age:\s*(\d+)", text).group(1))
            weight = float(re.search(r"Weight:\s*(\d+\.\d+)", text).group(1))
            logger.info(f"Extracted patient data: Hgb={hemoglobin}, Glu={glucose}, Age={age}, Weight={weight}")
            return text, hemoglobin, glucose, age, weight
        except AttributeError:
            raise ValueError("File missing required data: Hemoglobin, Glucose, Age, Weight")
        except Exception as e:
            logger.error(f"Error extracting patient data: {str(e)}", exc_info=True)
            raise

    def read_file(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        try:
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
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}", exc_info=True)
            raise

# Load TDC dataset (placeholder; ideally save this during training)
def load_tdc_data():
    return [
        "CCO", "CCN", "CCC", "C=O", "C#N", "CC=O", "CC#N", "COC", "CCOC", "CN",
        "CNC", "CCNC", "C(=O)O", "CC(=O)O", "C#CC", "CC#CC", "COC=O", "CCOC=O",
        "CN(C)C", "CCN(C)C"
    ]

# Decode SMILES with confidence
def decode_smiles(smiles, model, state):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol_formula = rdMolDescriptors.CalcMolFormula(mol) if mol else "Invalid SMILES"
        compounds = pcp.get_compounds(smiles, 'smiles')
        drug_name = compounds[0].iupac_name if compounds else "Unknown"
        with torch.no_grad():
            logits, _ = model.policy.predict_values(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
            confidence = torch.softmax(logits, dim=-1).max().item()
        logger.info(f"Decoded SMILES: {smiles} -> {drug_name}, {mol_formula}, Confidence={confidence}")
        return drug_name, mol_formula, confidence
    except Exception as e:
        logger.error(f"Error decoding SMILES {smiles}: {str(e)}", exc_info=True)
        raise

# Visualization
def plot_patient_trajectory(history, rewards, filename="static/patient_trajectory.png"):
    try:
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
        logger.info(f"Trajectory plot saved as {filename}")
    except Exception as e:
        logger.error(f"Error plotting trajectory: {str(e)}", exc_info=True)
        raise

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

            # Process file and predict
            pipeline = DataPipeline()
            try:
                blood_report, hemoglobin, glucose, age, weight = pipeline.extract_patient_data(file_path)
                blood_features = pipeline.process_blood_report(blood_report)
                new_state = np.concatenate([blood_features, [hemoglobin, glucose, age, weight]]).astype(np.float32)

                logger.info("Loading LUMINARIX model...")
                model = PPO.load("ppo_luminarix_advanced.zip", device=device)
                logger.info("Model loaded successfully.")

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
                    logger.info(steps[-1])
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
    try:
        return send_file(os.path.join('static', filename))
    except Exception as e:
        logger.error(f"Error serving plot {filename}: {str(e)}", exc_info=True)
        flash(f"Error serving plot: {str(e)}")
        return redirect(url_for('upload_file'))

if __name__ == "__main__":
    os.makedirs('static', exist_ok=True)
    model_path = "ppo_luminarix_advanced.zip"
    if not os.path.exists(model_path):
        logger.error(f"Trained model '{model_path}' not found. Run train_luminarix.py first.")
        print(f"Error: Trained model '{model_path}' not found. Run train_luminarix.py first.")
    else:
        try:
            logger.info("Starting Flask server...")
            app.run(debug=True)
        except Exception as e:
            logger.error(f"Error starting Flask server: {str(e)}", exc_info=True)
            raise