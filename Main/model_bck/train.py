import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import re
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tdc.single_pred import ADME
import os
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimulatedPatientData:
    def __init__(self, num_patients=100):
        self.num_patients = num_patients
        self.blood_reports = [f"Patient {i}: Hemoglobin: {np.random.uniform(8, 18):.1f} g/dL, Glucose: {np.random.uniform(60, 140):.1f} mg/dL, Age: {np.random.randint(20, 80)}, Weight: {np.random.uniform(50, 100):.1f} kg" for i in range(num_patients)]
        self.protein_scans = [f"protein_scan_{i}.jpg" for i in range(num_patients)]
        self.drug_effects = np.array([
            [-0.5, -5.0, 0.1], [0.5, -3.0, 0.2], [-1.0, 2.0, 0.3], [1.0, 0.0, 0.1],
            [0.0, -10.0, 0.4], [-0.2, 5.0, 0.2], [0.8, -2.0, 0.1], [-0.8, -1.0, 0.3],
            [0.3, 3.0, 0.2], [0.0, 0.0, 0.0], [1.2, -4.0, 0.5], [-0.7, 6.0, 0.3],
            [0.4, -1.5, 0.1], [-1.5, 3.0, 0.4], [0.6, -2.5, 0.2], [-0.3, 4.0, 0.3],
            [0.9, -3.5, 0.1], [-0.9, 1.0, 0.2], [0.2, 2.0, 0.3], [-0.4, -6.0, 0.4]
        ])

    def get_patient_data(self, idx):
        return self.blood_reports[idx], self.protein_scans[idx], self.drug_effects

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=256)
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.bert.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.resnet = models.resnet18(pretrained=True).to(device)
        self.resnet.fc = nn.Identity()
        self.resnet.eval()
        self.image_transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=770, nhead=5, dim_feedforward=512), num_layers=2
        ).to(device)
        self.fc = nn.Linear(770, 256).to(device)

    def forward(self, observations):
        batch_size = observations.shape[0]
        bert_input = observations[:, :-4].reshape(batch_size, 1, 768)
        extra_features = observations[:, -4:].to(device)
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_tensor = self.image_transform(img).unsqueeze(0).to(device).repeat(batch_size, 1, 1, 1)
        with torch.no_grad():
            img_features = self.resnet(img_tensor).reshape(batch_size, 1, -1)[:, :, -2:]
        combined = torch.cat((bert_input, extra_features.unsqueeze(1), img_features), dim=2)
        with torch.no_grad():
            transformer_out = self.transformer(combined.transpose(0, 1)).transpose(0, 1)
        return self.fc(transformer_out.squeeze(1))

class PatientDrugEnv(gym.Env):
    def __init__(self, patient_data):
        super(PatientDrugEnv, self).__init__()
        self.patient_data = patient_data
        self.pipeline = DataPipeline()
        self.num_drugs = 20
        self.state_dim = 772
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_drugs)
        self.current_patient_idx = None
        self.current_hemoglobin = None
        self.current_glucose = None
        self.current_age = None
        self.current_weight = None
        self.step_count = 0
        self.max_steps = 5
        self.history = deque(maxlen=5)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_patient_idx = np.random.randint(self.patient_data.num_patients)
        blood_report, protein_scan, _ = self.patient_data.get_patient_data(self.current_patient_idx)
        blood_features = self.pipeline.process_blood_report(blood_report)
        self.current_hemoglobin = float(re.search(r"Hemoglobin:\s*(\d+\.\d+)", blood_report).group(1))
        self.current_glucose = float(re.search(r"Glucose:\s*(\d+\.\d+)", blood_report).group(1))
        self.current_age = float(re.search(r"Age:\s*(\d+)", blood_report).group(1))
        self.current_weight = float(re.search(r"Weight:\s*(\d+\.\d+)", blood_report).group(1))
        self.state = np.concatenate([blood_features, [self.current_hemoglobin, self.current_glucose, self.current_age, self.current_weight]]).astype(np.float32)
        self.step_count = 0
        self.history.clear()
        self.history.append((self.current_hemoglobin, self.current_glucose))
        return self.state, {"patient_idx": self.current_patient_idx}

    def step(self, action):
        _, _, drug_effects = self.patient_data.get_patient_data(self.current_patient_idx)
        self.current_hemoglobin += drug_effects[action][0] * (self.current_weight / 70)
        self.current_glucose += drug_effects[action][1] * (self.current_weight / 70)
        side_effect = drug_effects[action][2] * (self.current_age / 50)
        
        self.state[-4:] = [self.current_hemoglobin, self.current_glucose, self.current_age, self.current_weight]
        self.history.append((self.current_hemoglobin, self.current_glucose))
        self.step_count += 1
        
        hemoglobin_error = max(0, 12 - self.current_hemoglobin) + max(0, self.current_hemoglobin - 16)
        glucose_error = max(0, 70 - self.current_glucose) + max(0, self.current_glucose - 110)
        reward = -(hemoglobin_error + glucose_error) - side_effect
        if reward > -0.5:
            reward += 1.0
        
        terminated = (reward > 0 or self.step_count >= self.max_steps)
        truncated = False
        info = {"new_hemoglobin": self.current_hemoglobin, "new_glucose": self.current_glucose, "side_effect": side_effect}
        return self.state, reward, terminated, truncated, info

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
def load_tdc_data():
    data = ADME(name='Caco2_Wang')
    return data.get_data()['Drug'].tolist()[:20]


if __name__ == "__main__":

    patient_data = SimulatedPatientData(num_patients=100)
    drug_list = load_tdc_data()
    print(f"Loaded {len(drug_list)} drugs from TDC: {drug_list[:2]}...")

    env = PatientDrugEnv(patient_data)
    check_env(env)
    policy_kwargs = dict(features_extractor_class=CustomFeatureExtractor, features_extractor_kwargs=dict())
    model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=0.0001, n_steps=2048, batch_size=64, n_epochs=10, device=device, policy_kwargs=policy_kwargs)
    print("Training LUMINARIX model on GPU...")
    model.learn(total_timesteps=100000)
    model.save("atherix_advanced")
    print("Model saved as 'atherix_advanced.zip'")