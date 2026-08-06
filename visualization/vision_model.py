import os
import json
import torch
import torch.nn as nn


# Define the ConvBiLSTM Architecture
class RobustConvBiLSTM(nn.Module):
    def __init__(self, input_dim=516, hidden_dim=128, num_classes=100):
        super(RobustConvBiLSTM, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv).transpose(1, 2)
        lstm_out, _ = self.lstm(x_conv)
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool = torch.max(lstm_out, dim=1)[0]
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.fc(pooled)


class SignVisionPredictor:
    def __init__(self, model_path, mapping_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load Gloss Mapping
        with open(mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        self.idx_to_gloss = {v: k for k, v in self.class_mapping.items()}

        # Load Model Weights
        self.model = RobustConvBiLSTM(num_classes=len(self.class_mapping)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict_top_k(self, padded_sequence, top_k=10):
        """
        Takes a processed numpy sequence (40, 516), converts to tensor,
        and returns top-K gloss names and confidence percentages.
        """
        input_tensor = torch.tensor(padded_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            topk_probs, topk_indices = torch.topk(probabilities, top_k, dim=1)

        predictions = []
        for i in range(top_k):
            idx = topk_indices[0][i].item()
            prob = topk_probs[0][i].item() * 100
            gloss = self.idx_to_gloss.get(idx, f"Class_{idx}")
            predictions.append((gloss, prob))

        return predictions