import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# NLTK indirme işlemleri (Eğer daha önce indirilmemişse)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Türkçe stopwordsleri ve lemmatizer'ı ayarlayın
stop_words = set(stopwords.words('turkish'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Metni küçük harfe çevir
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]  # Noktalama işaretlerini kaldır
    tokens = [word for word in tokens if word not in stop_words]  # Stopwordsleri kaldır
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize et
    return ' '.join(tokens)

# Veriyi yükleyin
df = pd.read_csv(r'C:\vssssscode\balanced23_data.csv')  # Verinin yolunu değiştirin

# Veriyi ön işleme tabi tutun
df['text'] = df['Review'].apply(preprocess_text)

# Eğitim ve test verisini ayırın
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Tokenizer ve model yükleyin
tokenizer = DistilBertTokenizer.from_pretrained('dbmdz/distilbert-base-turkish-cased')
model = DistilBertForSequenceClassification.from_pretrained('dbmdz/distilbert-base-turkish-cased', num_labels=5)

# Dataset ve DataLoader oluşturun
train_dataset = SentimentDataset(train_df['text'].to_numpy(), train_df['Rating (Star)'].to_numpy(), tokenizer, max_len=128)
val_dataset = SentimentDataset(val_df['text'].to_numpy(), val_df['Rating (Star)'].to_numpy(), tokenizer, max_len=128)

# Eğitim argümanlarını tanımlayın
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer'ı oluşturun ve modeli eğitin
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Modeli kaydedin
model.save_pretrained('gagazz1')
tokenizer.save_pretrained('gagazz1')
