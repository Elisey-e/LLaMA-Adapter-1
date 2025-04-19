import cv2
import llama
import torch
from PIL import Image
import xml.etree.ElementTree as ET
from sklearn.metrics import f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
import os

# Загрузка необходимых ресурсов NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')

# Инициализация модели
device = "cuda" if torch.cuda.is_available() else "cpu"
llama_dir = "LLaMA-7B"
model, preprocess = llama.load("CAPTION-7B", llama_dir, device)
model.eval()

# Функция для предварительной обработки текста
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    
    # Токенизация
    try:
        tokens = word_tokenize(text.lower())
    except:
        tokens = text.lower().split()
    
    # Удаление стоп-слов и пунктуации
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return tokens

# Функция для вычисления F1-score между двумя текстами
def calculate_f1(reference, candidate):
    ref_tokens = set(preprocess_text(reference))
    cand_tokens = set(preprocess_text(candidate))
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    # Вычисление precision и recall
    common_tokens = ref_tokens & cand_tokens
    precision = len(common_tokens) / len(cand_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    
    # Вычисление F1-score
    if (precision + recall) == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Парсинг XML-файла с разметкой CUTE80
def parse_cute80_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    image_data = []
    for image in root.findall('Image'):
        image_name = image.find('ImageName').text
        polygons = []
        for polygon in image.findall('PolygonPoints'):
            text = polygon.text.strip() if polygon.text else ""
            if text:  # Добавляем только непустые описания
                polygons.append(text)
        if polygons:  # Добавляем только изображения с описаниями
            image_data.append({'image_name': image_name, 'labels': polygons})
    
    return image_data

# Основная функция оценки
def evaluate_on_cute80(xml_path, images_dir):
    # Парсинг XML
    image_data = parse_cute80_xml(xml_path)
    
    f1_scores = []
    
    for item in image_data:
        image_path = os.path.join(images_dir, item['image_name'])
        
        try:
            # Проверка существования файла
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
                
            # Загрузка и предобработка изображения
            img_array = cv2.imread(image_path)
            if img_array is None:
                print(f"Could not read image: {image_path}")
                continue
                
            img = Image.fromarray(img_array)
            img = preprocess(img).unsqueeze(0).to(device)
            
            # Генерация описания
            prompt = llama.format_prompt('Please describe this image in detail.')
            generated_description = model.generate(img, [prompt])[0]
            
            # Сравнение с каждым эталонным описанием
            current_scores = []
            for label in item['labels']:
                f1 = calculate_f1(label, generated_description)
                current_scores.append(f1)
            
            if current_scores:
                max_f1 = max(current_scores)
                f1_scores.append(max_f1)
                
                print(f"Image: {item['image_name']}")
                print(f"Generated: {generated_description}")
                print(f"Labels: {item['labels']}")
                print(f"F1-scores: {current_scores} (max: {max_f1:.4f})")
                print("---------------")
            
        except Exception as e:
            print(f"Error processing {item['image_name']}: {str(e)}")
            continue
    
    # Вычисление среднего F1-score
    if f1_scores:
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        print(f"\nEvaluation results:")
        print(f"Processed {len(f1_scores)}/{len(image_data)} images")
        print(f"Mean F1-score: {mean_f1:.4f}")
        print(f"Std F1-score: {std_f1:.4f}")
        print(f"Min F1-score: {min(f1_scores):.4f}")
        print(f"Max F1-score: {max(f1_scores):.4f}")
        return mean_f1
    else:
        print("No valid F1-scores were calculated.")
        return 0.0

# Пример использования
if __name__ == "__main__":
    # Укажите пути к вашим файлам
    xml_path = "datasets/cute80/Groundtruth/GroundTruth.xml"  # Путь к XML-файлу с разметкой
    images_dir = "datasets/cute80/CUTE80"  
    
    # Проверка существования путей
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Запуск оценки
    evaluate_on_cute80(xml_path, images_dir)


