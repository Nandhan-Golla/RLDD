import gym
from gym import spaces
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import re
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tdc.single_pred import ADME
import os


class SimulatedPatientData:
    def __init__(self, num_patients=10):
        self.num_patients = num_patients
        self.blood_reports = [f"Patient {i}: Hemoglobin: {np.random.uniform(10, 15):.1f} g/dL, Glucose: {np.random.uniform(70, 120):.1f} mg/dL" for i in range(num_patients)]
        self.protein_scans = [f"protein_scan_{i}.jpg" for i in range(num_patients)] 
        self.drug_responses = np.random.rand(num_patients, 10) 

    def get_patient_data(self, idx):
        return self.blood_reports[idx], self.protein_scans[idx], self.drug_responses[idx]

class DataPipeline:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.eval()
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process_blood_report(self, report):
        inputs = self.tokenizer(report, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        bert_features = outputs.pooler_output.squeeze().numpy()
        hemoglobin = float(re.search(r"Hemoglobin:\s*(\d+\.\d+)", report).group(1))
        glucose = float(re.search(r"Glucose:\s*(\d+\.\d+)", report).group(1))
        return np.concatenate([bert_features, [hemoglobin, glucose]])

    def process_protein_scan(self, image_path):
        if not os.path.exists(image_path):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        else:
            img = Image.open(image_path).convert('RGB')
        input_tensor = self.image_transform(img).unsqueeze(0)
        with torch.no_grad():
            features = self.resnet(input_tensor)
        return features.squeeze().numpy()

class PatientDrugEnv(gym.Env):
    def __init__(self, patient_data):
        super(PatientDrugEnv, self).__init__()
        self.patient_data = patient_data
        self.pipeline = DataPipeline()
        self.num_drugs = 10
        self.state_dim = 770 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_drugs)
        self.current_patient_idx = None

    def reset(self):
        self.current_patient_idx = np.random.randint(self.patient_data.num_patients)
        blood_report, protein_scan, _ = self.patient_data.get_patient_data(self.current_patient_idx)
        blood_features = self.pipeline.process_blood_report(blood_report)
        protein_features = self.pipeline.process_protein_scan(protein_scan)
        self.state = np.concatenate([blood_features, protein_features[-2:]])
        return self.state

    def step(self, action):
        _, _, drug_responses = self.patient_data.get_patient_data(self.current_patient_idx)
        reward = drug_responses[action]
        next_state = self.state
        done = True
        info = {}
        return next_state, reward, done, info

    def render(self, mode='human'):
        print(f"Patient {self.current_patient_idx}: State {self.state[:5]}...")

def load_tdc_data():
    data = ADME(name='Caco2_Wang')
    drug_smiles = data.get_data()['Drug'].tolist()
    return drug_smiles[:10]

if __name__ == "__main__":
    patient_data = SimulatedPatientData(num_patients=10)
    drug_list = load_tdc_data()
    print(f"Loaded {len(drug_list)} drugs from TDC: {drug_list[:2]}...")

    env = PatientDrugEnv(patient_data)
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)
    print("Training RL model...")
    model.learn(total_timesteps=10000)
    model.save("ppo_drug_discovery")
    new_blood_report = "Patient Test: Hemoglobin: 14.2 g/dL, Glucose: 95.3 mg/dL"
    new_protein_scan = "test_protein_scan.jpg" 
    pipeline = DataPipeline()
    blood_features = pipeline.process_blood_report(new_blood_report)
    protein_features = pipeline.process_protein_scan(new_protein_scan)
    new_state = np.concatenate([blood_features, protein_features[-2:]])


    model = PPO.load("ppo_drug_discovery")
    action, _ = model.predict(new_state, deterministic=True)
    recommended_drug = drug_list[action]
    print(f"Recommended drug for new patient: {recommended_drug}")

    total_reward = 0
    for _ in range(5):
        obs = env.reset()
        action, _ = model.predict(obs)
        _, reward, _, _ = env.step(action)
        total_reward += reward
    print(f"Average reward over 5 test episodes: {total_reward / 5:.2f}")