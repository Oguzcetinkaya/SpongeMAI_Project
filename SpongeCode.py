import stanza
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np

# Stanza Türkçe modelini yükleyin
stanza.download('tr')
nlp = stanza.Pipeline(lang='tr', processors='tokenize,mwt,pos,ner')

# Modeli ve tokenizer'ı yükleyin
tokenizer = DistilBertTokenizer.from_pretrained('SpongeModel')
model = DistilBertForSequenceClassification.from_pretrained('SpongeModel')

def load_conjunctions(file_path):
    """Dosyadan bağlaçları oku ve bir liste döndür."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            conjunctions = [line.strip().lower() for line in file]
        return conjunctions
    except Exception as e:
        print(f"Error loading conjunctions file: {e}")
        return []

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
        return 'Negatif'
    elif rating == 3:
        return 'Nötr'
    elif rating in [4, 5]:
        return 'Pozitif'

def round_to_nearest_half(number):
    """Sayiyi en yakın yarım tam sayıya yuvarla."""
    return round(number * 2) / 2

def sponge_mai_rating(sentences):
    """SPONGE MAİ yıldız hesaplama fonksiyonu."""
    positive_count = sum(1 for sentence in sentences if classify_star_rating(predict_star_rating(sentence)) == 'Pozitif')
    negative_count = sum(1 for sentence in sentences if classify_star_rating(predict_star_rating(sentence)) == 'Negatif')
    
    if positive_count > 0 and negative_count == 0:
        return 5
    elif positive_count > negative_count and negative_count > 0:
        return 4
    elif positive_count == negative_count:
        return 3
    elif negative_count > positive_count and positive_count > 0:
        return 2
    elif negative_count > positive_count and positive_count == 0:
        return 1

def process_text(text, conjunctions_file):
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
    
    # SPONGE MAİ yıldız puanını hesapla
    sponge_mai_rating_value = sponge_mai_rating(all_sentences)
    
    # Ayrılan cümlelerin yıldız puanlarını tahmin et ve ortalama puanı hesapla
    star_ratings = [predict_star_rating(sentence) for sentence in all_sentences]
    average_star_rating = np.mean(star_ratings)
    
    # Ortalama puanı en yakın yarım tam sayıya yuvarla
    
    
    # Genel yıldız puanını tahmin et ve yazdır
    overall_star_rating = predict_star_rating(text)
    overall_rating_class = classify_star_rating(overall_star_rating)
    
    # Hesaplamalar
    total_sum = average_star_rating + sponge_mai_rating_value + overall_star_rating
    rating_divided = total_sum / 3
    rounded_rating_divided = round_to_nearest_half(rating_divided)
    
    # Sonuçları yazdır
    print("Ayırılmış Parçalar ve yıldız puanları:")
    if all_sentences:
        # İlk ayrılan cümlenin entity'lerini yazdır
        if all_sentence_entities[0]:
            first_sentence_entities = all_sentence_entities[0]
            print(f"İlk Ayrılan Cümlenin Entity'leri: {', '.join(first_sentence_entities)}")
        
    for i, part in enumerate(all_sentences):
        star_rating = predict_star_rating(part)
        entities = all_sentence_entities[i]
        rating_class = classify_star_rating(star_rating)
        
        print(f"- Parça: {part}")
        if entities:
            print(f"  Entity'ler: {', '.join(entities)}")
        else:
            # İlk ayrılan cümlenin entity'sini yazdır
            first_entity = all_sentence_entities[0] if all_sentence_entities else 'Bulunamadı'
            print(f"  Entity'ler: {first_entity}")
        print(f"  Entity Sentiment: {rating_class}")
        print(f"  Entity'nin Yıldız Puanı: {star_rating}")
        print()
    
    # Genel yıldız puanını yazdır
    print(f"Sponge MAI S (Girilen Textin Yıldız Puanı) : {overall_star_rating} ({overall_rating_class})")
    
    # SPONGE MAİ puanını yazdır
    print(f"SPONGE MAİ X (Olumluluk ve olumsuzluk sayısına göre puanlama): {sponge_mai_rating_value}")

    # Ortalama puanı yazdır
    print(f"Entity'lerin Puanlarının Ortalaması: {average_star_rating}")


    
    # Yeni hesaplanan puanı yazdır
    print(f"Yorumun Yıldız Puanı: {rounded_rating_divided}")

# Örnek metin ve bağlaçlar dosya yolu
text = "türkcell böyle olmaz"
conjunctions_file = 'baglaclar.txt'  

# Metni işleyin
process_text(text, conjunctions_file)
