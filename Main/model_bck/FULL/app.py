import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import base64
import logging
import requests
import pubchempy as pcp
import matplotlib.animation as animation
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def decode_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_formula = rdMolDescriptors.CalcMolFormula(mol)
    compounds = pcp.get_compounds(smiles, 'smiles')
    drug_name = compounds[0].iupac_name if compounds else "Unknown"
    return drug_name, mol_formula



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
    logger.error(f"Required files missing: {MODEL_PATH}, {DRUG_LIST_PATH}, {DRUG_EFFECTS_PATH}, {DRUG_NAMES_PATH}.")
    raise FileNotFoundError("Model or data files missing.")

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

DRUG_INTERACTIONS = {
    "Aspirin": ["Warfarin", "Ibuprofen"],
    "Metformin": ["Citalopram"],
    "Warfarin": ["Aspirin", "Clopidogrel"]
}

class DataPipeline:
    def extract_patient_data(self, file_path):
        with open(file_path, 'r') as f:
            text = f.read()
        hemoglobin = float(re.search(r"Hemoglobin:\s*(\d+\.\d+)", text).group(1))
        glucose = float(re.search(r"Glucose:\s*(\d+\.\d+)", text).group(1))
        age = float(re.search(r"Age:\s*(\d+)", text).group(1))
        weight = float(re.search(r"Weight:\s*(\d+\.\d+)", text).group(1))
        state = np.array([hemoglobin, glucose, age, weight], dtype=np.float32)
        logger.info(f"Extracted patient data: Hemoglobin={hemoglobin}, Glucose={glucose}, Age={age}, Weight={weight}")
        return state

def simulate_treatment(state):
    hemoglobin, glucose, age, weight = state
    history = [(hemoglobin, glucose)]
    rewards = []
    actions = []
    with torch.no_grad():
        for step in range(5):
            action_logits = policy(torch.tensor(state, dtype=torch.float32).to(device))
            epsilon = 0.3
            if random.random() < epsilon:
                action = random.randint(0, num_drugs - 1)
            else:
                action = torch.argmax(action_logits).item()
            logger.info(f"Step {step}: State={state.tolist()}, Action={action} ({drug_names[action]}), Logits={action_logits.tolist()}")
            hgb_change = drug_effects[action][0] * (weight / 70)
            glu_change = drug_effects[action][1] * (weight / 70)
            hemoglobin += hgb_change
            glucose += glu_change
            side_effect = drug_effects[action][2] * (age / 50)

            hemoglobin = max(10.0, min(18.0, hemoglobin))
            glucose = max(50.0, min(200.0, glucose)) 

            reward = -(max(0, 12 - hemoglobin) + max(0, hemoglobin - 16) +
                       max(0, 70 - glucose) + max(0, glucose - 110)) - side_effect
            state = np.array([hemoglobin, glucose, age, weight], dtype=np.float32)
            history.append((hemoglobin, glucose))
            rewards.append(reward)
            actions.append(action)
            if reward > -1:
                break
    logger.info(f"Simulation completed: Actions={actions}, Drugs={[drug_names[a] for a in actions]}, History={history}")
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
    static_buf = BytesIO()
    plt.savefig(static_buf, format="png")
    static_buf.seek(0)
    static_plot = base64.b64encode(static_buf.getvalue()).decode('utf-8')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    line1, = ax1.plot([], [], label="Hemoglobin (g/dL)", color="red")
    line2, = ax1.plot([], [], label="Glucose (mg/dL)", color="blue")
    ax1.axhspan(12, 16, alpha=0.2, color='red', label="Hgb Target Range")
    ax1.axhspan(70, 110, alpha=0.2, color='blue', label="Glucose Target Range")
    ax1.set_xlim(0, len(history) - 1)
    ax1.set_ylim(min(min(hgb), 70), max(max(hgb), 110))
    ax1.set_xlabel("Treatment Step")
    ax1.set_ylabel("Concentration")
    ax1.legend()
    ax1.set_title("Patient Biomarker Trajectory")
    line3, = ax2.plot([], [], label="Treatment Score", color="green")
    ax2.set_xlim(0, len(rewards) - 1)
    ax2.set_ylim(min(rewards) - 1, max(rewards) + 1)
    ax2.set_xlabel("Treatment Step")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.set_title("Treatment Effectiveness")
    plt.tight_layout()

    def update(frame):
        line1.set_data(range(frame + 1), hgb[:frame + 1])
        line2.set_data(range(frame + 1), glu[:frame + 1])
        if frame < len(rewards):
            line3.set_data(range(frame + 1), rewards[:frame + 1])
        return line1, line2, line3

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=500, repeat=True)
    gif_path = os.path.join(STATIC_FOLDER, 'trajectory.gif')
    try:
        ani.save(gif_path, writer='pillow', fps=2)
    except Exception as e:
        logger.error(f"Failed to save GIF: {str(e)}")
    plt.close()
    with open(gif_path, 'rb') as f:
        trajectory_gif = base64.b64encode(f.read()).decode('utf-8')
    
    return trajectory_gif, static_plot

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
        plt.savefig(os.path.join(STATIC_FOLDER, "drug_effects.png"))
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error in plot_drug_effects: {str(e)}")
        return None

def plot_drug_structure(smiles, drug_name):
    try:
        compounds = pcp.get_compounds(smiles, 'smiles', record_type='3d')
        if not compounds or not hasattr(compounds[0], 'coords'):
            logger.warning(f"No 3D structure for {drug_name} ({smiles})")
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
        plt.savefig(os.path.join(STATIC_FOLDER, "drug_structure.png"))
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8'), drug_name
    except Exception as e:
        logger.error(f"Error in plot_drug_structure: {str(e)}")
        return None, drug_name

def predict_conditions(state):
    hemoglobin, glucose, _, _ = state
    conditions = []
    if hemoglobin < 12:
        conditions.append("Possible Anemia (Hemoglobin < 12 g/dL)")
    if glucose > 110:
        conditions.append("Possible Diabetes (Glucose > 110 mg/dL)")
    return conditions if conditions else ["No significant conditions detected"]

def generate_insights(drug, conditions):
    insights = [f"Recommended {drug} to optimize biomarkers."]
    if "Anemia" in " ".join(conditions):
        insights.append("Consider iron supplements alongside treatment.")
    if "Diabetes" in " ".join(conditions):
        insights.append("Monitor glucose levels closely post-treatment.")
    return insights

def symptom_checker(symptoms):
    symptom_map = {
        "fever": ["Possible Flu", "Possible Infection"],
        "fatigue": ["Possible Anemia", "Possible Chronic Fatigue"],
        "high blood sugar": ["Possible Diabetes"],
        "low energy": ["Possible Anemia"]
    }
    possible_conditions = []
    for symptom in symptoms.split(","):
        symptom = symptom.strip().lower()
        if symptom in symptom_map:
            possible_conditions.extend(symptom_map[symptom])
    return list(set(possible_conditions)) if possible_conditions else ["No conditions matched"]

def calculate_dosage(state, drug_index):
    _, _, age, weight = state
    base_dose = 10 
    dose = base_dose * (weight / 70) * (50 / age) 
    return f"Recommended {drug_names[drug_index]} dosage: {dose:.2f} mg (adjusted for age {age}, weight {weight} kg)"

def check_interactions(drug, current_meds):
    current_meds = [med.strip() for med in current_meds.split(",")]
    interactions = DRUG_INTERACTIONS.get(drug, [])
    alerts = [f"Warning: {drug} may interact with {med}" for med in current_meds if med in interactions]
    return alerts if alerts else ["No interactions detected"]

def export_timeline(history, rewards, drug):
    pdf_path = os.path.join(STATIC_FOLDER, 'timeline.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, f"Treatment Timeline for {drug}")
    c.drawString(100, 730, "Step | Hemoglobin (g/dL) | Glucose (mg/dL) | Reward")
    y = 710
    for i, (hgb, glu) in enumerate(history):
        reward = rewards[i] if i < len(rewards) else 0
        c.drawString(100, y, f"{i+1}    | {hgb:.1f}            | {glu:.1f}         | {reward:.2f}")
        y -= 20
    c.save()
    return pdf_path

latest_state = None
latest_history = None
latest_rewards = None
latest_actions = None
latest_symptoms = ""
latest_meds = ""

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
                trajectory_gif, _ = plot_trajectory(history, rewards)
                final_action = actions[-1]
                return redirect(url_for('result', 
                                        action=final_action, 
                                        drug=drug_names[final_action], 
                                        smiles=drug_list[final_action]))
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                flash(f"Error: {str(e)}")
                return redirect(request.url)
        flash('Invalid file')
        return redirect(request.url)
    return render_template('upload.html')

@app.route('/visualization')
def visualization():
    global latest_actions
    drug_plot = plot_drug_effects()
    if drug_plot is None:
        flash("Error generating drug effects plot")
        return redirect(url_for('upload_file'))
    if latest_actions:
        final_action = latest_actions[-1]
        drug_structure_plot, drug_name = plot_drug_structure(drug_list[final_action], drug_names[final_action])
    else:
        drug_structure_plot, drug_name = None, "No drug selected"
    return render_template('visualization.html', drug_plot=drug_plot, drug_structure_plot=drug_structure_plot, drug_name=drug_name)

@app.route('/report', methods=['GET', 'POST'])
def report():
    global latest_state, latest_history, latest_rewards, latest_actions, latest_symptoms, latest_meds
    if latest_state is None or latest_history is None or latest_rewards is None or latest_actions is None:
        flash("No simulation data available. Please upload a patient report first.")
        return redirect(url_for('upload_file'))
    
    trajectory_gif, static_plot = plot_trajectory(latest_history, latest_rewards)
    drug_plot = plot_drug_effects()
    final_action = latest_actions[-1]
    drug_structure_plot, drug_name = plot_drug_structure(drug_list[final_action], drug_names[final_action])
    final_smiles = drug_list[final_action]
    conditions = predict_conditions(latest_state)
    interactions = DRUG_INTERACTIONS.get(drug_name, [])
    insights = generate_insights(drug_name, conditions)

    symptom_results = []
    dosage = ""
    interaction_alerts = []
    if request.method == 'POST':
        if 'symptoms' in request.form:
            latest_symptoms = request.form['symptoms']
            symptom_results = symptom_checker(latest_symptoms)
        if 'calculate_dosage' in request.form:
            dosage = calculate_dosage(latest_state, final_action)
        if 'current_meds' in request.form:
            latest_meds = request.form['current_meds']
            interaction_alerts = check_interactions(drug_name, latest_meds)
    
    return render_template('report.html', 
                          trajectory_gif=trajectory_gif,
                          static_plot=static_plot,
                          drug_plot=drug_plot, 
                          drug_structure_plot=drug_structure_plot, 
                          drug=drug_name, 
                          smiles=final_smiles,
                          history=latest_history,
                          rewards=latest_rewards,
                          conditions=conditions,
                          interactions=interactions,
                          insights=insights,
                          symptom_results=symptom_results,
                          dosage=dosage,
                          interaction_alerts=interaction_alerts,
                          symptoms=latest_symptoms,
                          current_meds=latest_meds)

@app.route('/result/<int:action>/<drug>/<smiles>')
#drug_name, mol_formula = decode_smiles(recommended_drug)
def result(action, drug, smiles):
    global latest_history, latest_rewards
    _, static_plot = plot_trajectory(latest_history, latest_rewards)
    return render_template('result.html', action=action, drug=drug, smiles=smiles,_drug_ = decode_smiles(smiles)[0], static_plot=static_plot)

@app.route('/download_trajectory')
def download_trajectory():
    return send_file(os.path.join(STATIC_FOLDER, 'trajectory.gif'), as_attachment=True, download_name='trajectory.gif')

@app.route('/download_drug_effects')
def download_drug_effects():
    plot_drug_effects()
    return send_file(os.path.join(STATIC_FOLDER, 'drug_effects.png'), as_attachment=True, download_name='drug_effects.png')

@app.route('/download_drug_structure')
def download_drug_structure():
    global latest_actions
    if latest_actions is None:
        flash("No drug structure available.")
        return redirect(url_for('upload_file'))
    final_action = latest_actions[-1]
    plot_drug_structure(drug_list[final_action], drug_names[final_action])
    return send_file(os.path.join(STATIC_FOLDER, 'drug_structure.png'), as_attachment=True, download_name='drug_structure.png')

@app.route('/download_timeline')
def download_timeline():
    global latest_history, latest_rewards, latest_actions
    final_action = latest_actions[-1]
    pdf_path = export_timeline(latest_history, latest_rewards, drug_names[final_action])
    return send_file(pdf_path, as_attachment=True, download_name='treatment_timeline.pdf')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True, port=5002)