from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import stanza
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = FastAPI()

# Stanza Türkçe modelini yükleyin
stanza.download('tr')
nlp = stanza.Pipeline(lang='tr', processors='tokenize,mwt,pos,ner')

# Modeli ve tokenizer'ı yükleyin
tokenizer = DistilBertTokenizer.from_pretrained('SpongeModel')
model = DistilBertForSequenceClassification.from_pretrained('SpongeModel')

class TextRequest(BaseModel):
    text: str

def load_conjunctions(file_path):
    """Dosyadan bağlaçları oku ve bir liste döndür."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            conjunctions = [line.strip().lower() for line in file]
        return conjunctions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading conjunctions file: {e}")

def split_by_conjunctions(text, conjunctions):
    """Metni bağlaçlara göre parçalara ayır."""
    pattern = '|'.join([rf'\b{re.escape(conj)}\b' for conj in conjunctions])
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]

def split_by_punctuation(text):
    """Metni ünlem, nokta ve soru işaretlerine göre parçalara ayır."""
    punctuation_pattern = r'(?<=[.,!?;:])\s+'
    return re.split(punctuation_pattern, text)

def extract_entities(text):
    """Metinden entity'leri çıkar."""
    doc = nlp(text)
    entities = []
    
    for sent in doc.sentences:
        current_entity = []
        for word in sent.words:
            if word.upos in ['NOUN', 'PROPN']:
                if current_entity:
                    current_entity.append(word.text)
                else:
                    current_entity.append(word.text)
            else:
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
        
        if current_entity:
            entities.append(' '.join(current_entity))
    
    return entities

def split_by_entities(text):
    """Metni entity'ler arasında ayır."""
    entities = extract_entities(text)
    
    if not entities:
        return [text], [None]
    
    sentences = []
    sentence_entities = []
    start = 0
    
    def find_entity_index(entity, start):
        return text.lower().find(entity.lower(), start)
    
    for i, entity in enumerate(entities):
        entity_lower = entity.lower()
        index = find_entity_index(entity_lower, start)
        
        if index != -1:
            if i == 0:
                end_index = find_entity_index(entities[i+1].lower(), index) if i + 1 < len(entities) else len(text)
                sentences.append(text[start:end_index].strip())
                sentence_entities.append([entity])
            else:
                end_index = find_entity_index(entities[i+1].lower(), index) if i + 1 < len(entities) else len(text)
                sentences.append(text[start:end_index].strip())
                sentence_entities.append([entity])
            
            start = end_index

    if start < len(text):
        sentences.append(text[start:].strip())
        sentence_entities.append([e for e in entities if find_entity_index(e.lower(), start) != -1])
    
    return sentences, sentence_entities

def remove_punctuation(text):
    """Metinden noktalama işaretlerini çıkar."""
    return re.sub(r'[.,!?;:]', '', text).strip()

def predict_star_rating(text):
    """Metnin yıldız puanını tahmin et."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    return predicted_class + 1  # Model 0-4 arası bir değer döndürüyor, 1-5 arası dönüştürülüyor

def classify_star_rating(rating):
    """Yıldız puanını sınıflandır."""
    if rating in [1, 2]:
        return 'olumsuz'
    elif rating == 3:
        return 'nötr'
    elif rating in [4, 5]:
        return 'olumlu'

@app.post("/predict")
async def process_text(request: TextRequest):
    text = request.text
    conjunctions_file = 'baglaclar.txt'
    
    # Bağlaçları yükle
    conjunctions = load_conjunctions(conjunctions_file)
    
    # Metni küçük harfe dönüştür
    text = text.lower()
    
    # Metni bağlaçlara göre ayırın
    conjunction_parts = split_by_conjunctions(text, conjunctions)
    
    all_sentences = []
    all_sentence_entities = []
    
    # Her bağlaç parçasını noktalama işaretlerine göre ayırın
    for part in conjunction_parts:
        if part:
            punctuation_parts = split_by_punctuation(part)
            for sub_part in punctuation_parts:
                cleaned_sentence = remove_punctuation(sub_part)
                if cleaned_sentence:  # Boş olmayan cümleleri işle
                    # Bağlaçları kontrol et ve gerekirse çıkart
                    words = cleaned_sentence.split()
                    filtered_words = [word for word in words if word.lower() not in conjunctions]
                    filtered_sentence = ' '.join(filtered_words)
                    
                    entities, sentence_entities = split_by_entities(filtered_sentence)
                    
                    if filtered_sentence:  # Boş olmayan cümleleri ekle
                        all_sentences.append(filtered_sentence)
                    
                    # Her cümle için entity'leri belirle
                    if sentence_entities[0] is None:
                        sentence_entities = [all_sentence_entities[0] if all_sentence_entities else None]
                    
                    relevant_entities = []
                    for e in sentence_entities:
                        if e and any(entity.lower() in filtered_sentence.lower() for entity in e):
                            relevant_entities.extend(e)
                    all_sentence_entities.append(relevant_entities)
    
    # Hesaplamalar
    results = []
    entity_set = set()  # Entity'leri sıralı şekilde tutacak set
    for i, part in enumerate(all_sentences):
        star_rating = predict_star_rating(part)
        rating_class = classify_star_rating(star_rating)
        entities = all_sentence_entities[i]
        
        # Her entity için duygu durumunu cümleye göre belirle
        for entity in entities:
            if entity not in entity_set:
                results.append({
                    "entity": entity,
                    "sentiment": rating_class  # Entity'nin sentimenti, cümlenin sentimenti olur
                })
                entity_set.add(entity)
    #uvicorn SpongeMAIWebService:app  --host 0.0.0.0 --reload
    # http://127.0.0.1:8000/docs
    # Entity listesine sıralı şekilde entity'leri ekleyin
    entity_list = [e['entity'] for e in results]

    return {
        "entity_list": entity_list,
        "results": results
    }
 