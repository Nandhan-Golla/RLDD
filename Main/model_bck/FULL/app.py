import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import re
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import base64
import logging
import requests
import pubchempy as pcp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cpu")
logger.info(f"Using device: {device}")

app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
class SimplePolicy(nn.Module):
    def __init__(self, input_dim=4, output_dim=20):
        super(SimplePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)
MODEL_PATH = "simple_policy.pth"
DRUG_LIST_PATH = "drug_list.npy"
DRUG_EFFECTS_PATH = "drug_effects.npy"
DRUG_NAMES_PATH = "drug_names.txt"

if not all(os.path.exists(p) for p in [MODEL_PATH, DRUG_LIST_PATH, DRUG_EFFECTS_PATH, DRUG_NAMES_PATH]):
    logger.error(f"Required files missing: {MODEL_PATH}, {DRUG_LIST_PATH}, {DRUG_EFFECTS_PATH}, {DRUG_NAMES_PATH}. Ensure all files exist.")
    raise FileNotFoundError("Model or data files missing. Run training script and create drug_names.txt.")

drug_list = np.load(DRUG_LIST_PATH, allow_pickle=True).tolist()
drug_effects = np.load(DRUG_EFFECTS_PATH, allow_pickle=True).tolist()
with open(DRUG_NAMES_PATH, 'r') as f:
    drug_names = [line.strip() for line in f.readlines()]
num_drugs = len(drug_list)
policy = SimplePolicy(input_dim=4, output_dim=num_drugs).to(device)
try:
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy.eval()
    logger.info(f"Loaded pre-trained model from {MODEL_PATH} with {num_drugs} drugs")
except Exception as e:
    logger.error(f"Failed to load {MODEL_PATH}: {str(e)}", exc_info=True)
    raise

if len(drug_names) != num_drugs:
    logger.error(f"Drug names count ({len(drug_names)}) does not match drug list count ({num_drugs})")
    raise ValueError("Mismatch between drug_names.txt and drug_list.npy")

class DataPipeline:
    def extract_patient_data(self, file_path):
        with open(file_path, 'r') as f:
            text = f.read()
        hemoglobin = float(re.search(r"Hemoglobin:\s*(\d+\.\d+)", text).group(1))
        glucose = float(re.search(r"Glucose:\s*(\d+\.\d+)", text).group(1))
        age = float(re.search(r"Age:\s*(\d+)", text).group(1))
        weight = float(re.search(r"Weight:\s*(\d+\.\d+)", text).group(1))
        return np.array([hemoglobin, glucose, age, weight], dtype=np.float32)

def simulate_treatment(state):
    hemoglobin, glucose, age, weight = state
    history = [(hemoglobin, glucose)]
    rewards = []
    actions = []
    with torch.no_grad():
        for _ in range(5):
            action_logits = policy(torch.tensor(state, dtype=torch.float32).to(device))
            action = torch.argmax(action_logits).item()
            hemoglobin += drug_effects[action][0] * (weight / 70)
            glucose += drug_effects[action][1] * (weight / 70)
            side_effect = drug_effects[action][2] * (age / 50)
            reward = -(max(0, 12 - hemoglobin) + max(0, hemoglobin - 16) +
                       max(0, 70 - glucose) + max(0, glucose - 110)) - side_effect
            state = np.array([hemoglobin, glucose, age, weight], dtype=np.float32)
            history.append((hemoglobin, glucose))
            rewards.append(reward)
            actions.append(action)
            if reward > 0:
                break
    logger.info(f"Simulation completed: History={history}, Rewards={rewards}, Actions={actions}")
    return history, rewards, actions

def plot_trajectory(history, rewards):
    hgb, glu = zip(*history)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(hgb, label="Hemoglobin (g/dL)", color="red")
    ax1.plot(glu, label="Glucose (mg/dL)", color="blue")
    ax1.axhspan(12, 16, alpha=0.2, color='red', label="Hgb Target Range")
    ax1.axhspan(70, 110, alpha=0.2, color='blue', label="Glucose Target Range")
    ax1.set_xlabel("Treatment Step")
    ax1.set_ylabel("Concentration")
    ax1.legend()
    ax1.set_title("Patient Biomarker Trajectory")
    ax2.plot(rewards, label="Treatment Score", color="green")
    ax2.set_xlabel("Treatment Step")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.set_title("Treatment Effectiveness")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.savefig("static/trajectory.png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_drug_effects():
    try:
        hgb_effects = [effect[0] for effect in drug_effects]
        glu_effects = [effect[1] for effect in drug_effects]
        side_effects = [effect[2] for effect in drug_effects]
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(drug_list))
        ax.bar(x, hgb_effects, width=0.2, label="Hemoglobin Change (g/dL)", align='center')
        ax.bar([i + 0.2 for i in x], glu_effects, width=0.2, label="Glucose Change (mg/dL)", align='center')
        ax.bar([i + 0.4 for i in x], side_effects, width=0.2, label="Side Effect Intensity", align='center')
        ax.set_xticks(x)
        ax.set_xticklabels(drug_names, rotation=45, ha="right")
        ax.set_xlabel("Drug")
        ax.set_ylabel("Effect Magnitude")
        ax.legend()
        ax.set_title("Drug Effects on Biomarkers and Side Effects")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.savefig("static/drug_effects.png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error in plot_drug_effects: {str(e)}", exc_info=True)
        return None

def plot_drug_structure(smiles, drug_name):
    try:
        compounds = pcp.get_compounds(smiles, 'smiles', record_type='3d')
        if not compounds or not hasattr(compounds[0], 'coords'):
            logger.warning(f"No 3D structure available for {drug_name} ({smiles})")
            return None, drug_name
        
        compound = compounds[0]
        x, y, z = [], [], []
        for atom in compound.coords:
            x.append(atom['x'])
            y.append(atom['y'])
            z.append(atom['z'] if 'z' in atom else 0.0) 
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', marker='o', s=50)
        ax.set_xlabel("X Coordinate (Å)")
        ax.set_ylabel("Y Coordinate (Å)")
        ax.set_zlabel("Z Coordinate (Å)")
        ax.set_title(f"3D Structure of {drug_name}")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.savefig("static/drug_structure.png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8'), drug_name
    except Exception as e:
        logger.error(f"Error in plot_drug_structure: {str(e)}", exc_info=True)
        return None, drug_name

latest_state = None
latest_history = None
latest_rewards = None
latest_actions = None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global latest_state, latest_history, latest_rewards, latest_actions
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file and file.filename.endswith(('.txt', '.pdf')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            pipeline = DataPipeline()
            try:
                state = pipeline.extract_patient_data(file_path)
                latest_state = state
                history, rewards, actions = simulate_treatment(state)
                latest_history, latest_rewards, latest_actions = history, rewards, actions
                if not history or not rewards or not actions:
                    raise ValueError("Simulation returned empty results")
                plot_trajectory(history, rewards)
                final_action = actions[-1]
                return redirect(url_for('result', 
                                        action=final_action, 
                                        drug=drug_names[final_action], 
                                        smiles=drug_list[final_action]))
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}", exc_info=True)
                flash(f"Error: {str(e)}")
                return redirect(request.url)
        flash('Invalid file')
        return redirect(request.url)
    return render_template('upload.html')

@app.route('/visualization')
def visualization():
    drug_plot = plot_drug_effects()
    if drug_plot is None:
        flash("Error generating drug effects plot")
        return redirect(url_for('upload_file'))
    return render_template('visualization.html', drug_plot=drug_plot)

@app.route('/report')
def report():
    global latest_state, latest_history, latest_rewards, latest_actions
    if latest_state is None or latest_history is None or latest_rewards is None or latest_actions is None:
        flash("No simulation data available. Please upload a patient report first.")
        return redirect(url_for('upload_file'))
    
    trajectory_plot = plot_trajectory(latest_history, latest_rewards)
    drug_plot = plot_drug_effects()
    final_action = latest_actions[-1]
    drug_structure_plot, drug_name = plot_drug_structure(drug_list[final_action], drug_names[final_action])
    final_smiles = drug_list[final_action]
    
    return render_template('report.html', 
                          trajectory_plot=trajectory_plot, 
                          drug_plot=drug_plot, 
                          drug_structure_plot=drug_structure_plot, 
                          drug=drug_name, 
                          smiles=final_smiles,
                          history=latest_history,
                          rewards=latest_rewards)

@app.route('/result/<int:action>/<drug>/<smiles>')
def result(action, drug, smiles):
    return render_template('result.html', action=action, drug=drug, smiles=smiles)

@app.route('/download_trajectory')
def download_trajectory():
    return send_file(os.path.join(app.config['STATIC_FOLDER'], 'trajectory.png'), as_attachment=True, download_name='trajectory.png')

@app.route('/download_drug_effects')
def download_drug_effects():
    plot_drug_effects()
    return send_file(os.path.join(app.config['STATIC_FOLDER'], 'drug_effects.png'), as_attachment=True, download_name='drug_effects.png')

@app.route('/download_drug_structure')
def download_drug_structure():
    global latest_actions
    if latest_actions is None:
        flash("No drug structure available. Please upload a patient report first.")
        return redirect(url_for('upload_file'))
    final_action = latest_actions[-1]
    plot_drug_structure(drug_list[final_action], drug_names[final_action])
    return send_file(os.path.join(app.config['STATIC_FOLDER'], 'drug_structure.png'), as_attachment=True, download_name='drug_structure.png')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True, port=5002)