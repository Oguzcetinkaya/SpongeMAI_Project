import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Model ve tokenizer'ı yükleyin
tokenizer = DistilBertTokenizer.from_pretrained('SpongeModel')
model = DistilBertForSequenceClassification.from_pretrained('SpongeModel')

# Modeli değerlendirme moduna alın
model.eval()

# Türkçe stopwords'leri ve lemmatizer'ı ayarlayın
stop_words = set(stopwords.words('turkish'))
lemmatizer = WordNetLemmatizer()

# Metin ön işleme fonksiyonu
def preprocess_text(text):
    text = text.lower()  # Küçük harfe çevir
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha()]  # Sadece kelimeleri al
    tokens = [token for token in tokens if token not in stop_words]  # Stopwords kaldır
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatization
    return ' '.join(tokens)

def predict_star_rating(text):
    # Metni ön işleme tabi tutun
    text = preprocess_text(text)
    
    # Tokenize edin
    inputs = tokenizer(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Tahmin yapın
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Her sınıfın olasılıklarını al
    scores = predictions[0].tolist()
    
    return scores

# Test cümleleri
test_sentences = [

   "daha iyi bir ambalaj beklerdim."
]
for sentence in test_sentences:
    scores = predict_star_rating(sentence)
    print(f"Cümle: {sentence}")
    for i in range(5):
        print(f"Yıldız Puanı {i+1}: {scores[i] * 100:.2f}%")
    print()
