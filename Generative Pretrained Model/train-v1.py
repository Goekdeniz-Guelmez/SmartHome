from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import json

# Load the dataset
with open('/Users/gokdenizgulmez/Desktop/gpt/SmartHome/Generative Pretrained Model/dataset-automated.json', 'r') as f:
    data = json.load(f)

# Prepare the texts and labels
texts = [json.dumps({k: v for k, v in item.items() if k != 'system_action' and k != 'system_response'}, separators=(',', ':')) for item in data]
labels = [json.dumps(item['system_action'], separators=(',', ':')) for item in data]

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a new padding token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Create a PyTorch Dataset
class SmartHomeDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = tokenizer.encode_plus(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        targets = tokenizer.encode_plus(label, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        
        # Check if the encoded inputs are empty
        if inputs['input_ids'].size(1) == 0 or targets['input_ids'].size(1) == 0:
            return None
        
        inputs['labels'] = targets['input_ids']
        return inputs

# Create a DataLoader
dataset = SmartHomeDataset(texts, labels)

# Filter out None values from the dataset
dataset = [data for data in dataset if data is not None]

dataloader = DataLoader(dataset, batch_size=8)

# Prepare for training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 3)  # 3 epochs

# Train the model
for epoch in range(3):
    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

# Save the trained model
model.save_pretrained('smart_home_model')
