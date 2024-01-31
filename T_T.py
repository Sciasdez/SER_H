import time
import logging
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
from transformers import Trainer
from transformers import TrainingArguments
from tqdm import tqdm
import torch
import os
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor
from datasets import Dataset
import librosa
from transformers import HubertForSequenceClassification
from transformers import AutoModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pandas.core.frame import DataFrame
import seaborn as sns


cuda_id = 2
device = torch.device("cuda:%s" % cuda_id if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(cuda_id) if torch.cuda.is_available() else "cpu"
print("We are using the device %s - %s" % (device, device_name))

TES = '/home/siavosh.sepanta/TIM-Net_SER/Code/SER_WAV_DATA/TESS/ALL'
dir_list = os.listdir(TES)
dir_list.sort()


gender = []
emotion = []
labels = []
source = []
path = []

# Mapping emotions to labels
emotion_label_mapping_tess = {
    'neutral': {'emotion': 'neutral', 'label': '01'},
    'happy': {'emotion': 'happy', 'label': '03'},
    'sad': {'emotion': 'sad', 'label': '04'},
    'angry': {'emotion': 'angry', 'label': '05'},
    'fear': {'emotion': 'fear', 'label': '06'},
    'disgust': {'emotion': 'disgust', 'label': '07'},
    'ps': {'emotion': 'surprise', 'label': '08'}
}

# Iterate through files in the directory
for file in os.listdir(TES):
    if file.endswith('.wav'):
        # Extracting the parts of the filename
        parts = file.split('_')
        emotion_label_tess = parts[2].split('.')[0]  # Extracting the third part (C) of the filename without extension

        # Assign values based on the filename format
        if emotion_label_tess in emotion_label_mapping_tess:
            emotion_info_tess = emotion_label_mapping_tess[emotion_label_tess]
            emotion.append(emotion_info_tess['emotion'])
            labels.append(emotion_info_tess['label'])
        gender.append('female')
        source.append('TESS')
        path.append(os.path.abspath(os.path.join(TES, file)))  # Constructing the path column accurately

# Creating the DataFrame
tess_data = {
    'gender': gender,
    'emotion': emotion,
    'labels': labels,
    'source': source,
    'path': path
}

TES_df = pd.DataFrame(tess_data)
print(TES_df)
        

#In this demonstration, we only choose 4 emotions, neutral, happy, sad and angry.
TES_df = TES_df[(TES_df["emotion"]=="neutral") | (TES_df["emotion"]=="happy") | (TES_df["emotion"]=="sad") | (TES_df["emotion"]=="angry")]
train_df = TES_df.sample(frac=0.8)
test_df = TES_df.drop(train_df.index)
#Let's have a look at the training data.
print(test_df)
print(train_df["emotion"].value_counts())


# Display the first data.
id = 1
print(train_df.iloc[id])
ipd.Audio(data=test_df.iloc[id]["path"], autoplay=True, rate=16000)


def map_to_array(example):
    speech, _ = librosa.load(example["path"], sr=16000, mono=True)
    example["speech"] = speech
    return example

train_data = Dataset.from_pandas(train_df).map(map_to_array)
test_data = Dataset.from_pandas(test_df).map(map_to_array)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")

train_encodings = feature_extractor(list(train_data["speech"]), sampling_rate=16000, padding=True, return_tensors="pt")
test_encodings = feature_extractor(list(test_data["speech"]), sampling_rate=16000, padding=True, return_tensors="pt")

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        pattern = {1:0, 2:0, 3:1, 4:3, 5:2}
        self.labels = [pattern[int(x)] for x in labels]

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, list(train_data["labels"]))
test_dataset = EmotionDataset(test_encodings, list(test_data["labels"]))


# Loading the model
model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
model.to(device)

# Loading the optimizer
optim = AdamW(model.parameters(), lr=1e-5)
# Prediction function
def predict(outputs):
    probabilities = torch.softmax(outputs["logits"], dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    return predictions

# Set the number of epoch
epoch = 3
# Start training
model.train()

train_loss = list()
train_accuracies = list()
for epoch_i in range(epoch):
    print('Epoch %s/%s' % (epoch_i + 1, epoch))
    time.sleep(0.3)

    # Get training data by DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    correct = 0
    count = 0
    epoch_loss = list()
    
    pbar = tqdm(train_loader)
    for batch in pbar:
        optim.zero_grad()
        input_ids = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        loss.backward()
        optim.step()
        
        # make predictions
        predictions = predict(outputs)

        # count accuracy
        correct += predictions.eq(labels).sum().item()
        count += len(labels)
        accuracy = correct * 1.0 / count

        # show progress along with metrics
        pbar.set_postfix({
            'Loss': '{:.3f}'.format(loss.item()),
            'Accuracy': '{:.3f}'.format(accuracy)
        })
        
        # record the loss for each batch
        epoch_loss.append(loss.item())
        
    pbar.close()
    
    # record the loss and accuracy for each epoch
    train_loss += epoch_loss
    train_accuracies.append(accuracy)
model.save_pretrained("saved_Models/TESS/")

# Testing
from torch.utils.data import DataLoader
# Get test data by DataLoader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Start testing
model.eval()

with torch.no_grad():
    
    correct = 0
    count = 0
    record = {"labels":list(), "predictions":list()}
    
    pbar = tqdm(test_loader)
    for batch in pbar:
        input_ids = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        # make predictions
        predictions = predict(outputs)

        # count accuracy
        correct += predictions.eq(labels).sum().item()
        count += len(labels)
        accuracy = correct * 1.0 / count

        # show progress along with metrics
        pbar.set_postfix({
            'loss': '{:.3f}'.format(loss.item()),
            'accuracy': '{:.3f}'.format(accuracy)
        })
    
        # record the results
        record["labels"] += labels.cpu().numpy().tolist()
        record["predictions"] += predictions.cpu().numpy().tolist()
        
    pbar.close()
    
time.sleep(0.3)
print("The final accuracy on the test dataset: %s%%" % round(accuracy*100,4))

# Convert test record to a pandas DataFrame object
from pandas.core.frame import DataFrame
df_record = DataFrame(record)
df_record.columns = ["Ground Truth","Model Prediction"]

def get_emotion(label_id):
    return model.config.id2label[label_id]
    
df_record["Ground Truth"] = df_record.apply(lambda x: get_emotion(x["Ground Truth"]), axis=1)
df_record["Model Prediction"] = df_record.apply(lambda x: get_emotion(x["Model Prediction"]), axis=1)

# Concat test texts and test records
df = pd.concat([test_df.reset_index(), df_record["Model Prediction"]], axis=1)
df["emotion"] = df.apply(lambda x: x["emotion"][:3], axis=1)

# Show incorrect predictions 
df[df["emotion"]!=df["Model Prediction"]]

# Display the Confusion Matrix
import seaborn as sns
crosstab = pd.crosstab(df_record["Ground Truth"],df_record["Model Prediction"])
sns.heatmap(crosstab, cmap='Oranges', annot=True, fmt='g', linewidths=5)
accuracy = df_record["Ground Truth"].eq(df_record["Model Prediction"]).sum() / len(df_record["Ground Truth"])
plt.title("Confusion Matrix (Accuracy: %s%%)" % round(accuracy*100,2))
plt.savefig('./images/cofusion_matrix_T_T.png')
plt.show()
