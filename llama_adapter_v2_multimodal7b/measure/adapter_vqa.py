import sys
from pathlib import Path

# Получаем путь к родительской директории текущего файла
parent_dir = str(Path(__file__).parent.parent)  # На два уровня выше: .parent.parent

# Добавляем путь в sys.path
sys.path.append(parent_dir)


import sys
from pathlib import Path
import os
import cv2
import llama
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import re

# Получаем путь к родительской директории текущего файла
parent_dir = str(Path(__file__).parent.parent)  # На два уровня выше: .parent.parent
sys.path.append(parent_dir)

def load_vqa_data(json_path):
    """Загрузка данных VQA из JSON файла"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    dataset = []
    processed_images = set()
    
    for item in data['data']:
        image_id = item['image_id']
        
        # Берем только одно описание для каждого изображения
        if image_id in processed_images:
            continue
            
        processed_images.add(image_id)
        
        # Берем список классов как ground truth
        ground_truth = ' '.join(item['image_classes'])
        
        entry = {
            'image_id': image_id,
            'ground_truth': ground_truth,
            'image_path': f"../datasets/VQA/test_images/{image_id}.jpg"
        }
        dataset.append(entry)
    
    return dataset

def run_inference_on_vqa(dataset_dir, json_path, model, preprocess, device, num_samples=None):
    """Инференс на данных VQA с использованием ImageBind"""
    test_data = load_vqa_data(json_path)
    
    if num_samples is not None:
        test_data = test_data[:num_samples]
    
    results = []
    
    for sample in tqdm(test_data, desc="Running inference"):
        img_path = sample['image_path']
        
        try:
            # Чтение и предобработка изображения
            img = Image.fromarray(cv2.imread(img_path))
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # Создаем промпт для модели
            prompt = llama.format_prompt('Identify ONLY the two most prominent objects in this image. '
                                      'List them separated by a single space, nothing else. '
                                      'Examples:\n'
                                      '- For laptop and phone: "laptop phone"\n'
                                      '- For car and tree: "car tree"\n'
                                      'Now analyze this image and provide JUST the objects.')
            
            # Генерация предсказания
            prediction = model.generate(img_tensor, [prompt])[0]
            
            results.append({
                'image_path': img_path,
                'ground_truth': sample['ground_truth'],
                'prediction': prediction
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return results

def clean_prediction(prediction):
    """Улучшенная очистка предсказания"""
    # Приводим к нижнему регистру
    prediction = prediction.lower().strip()
    
    # Удаляем все кроме букв и пробелов
    prediction = re.sub(r'[^a-z\s]', '', prediction)
    
    # Удаляем стоп-слова
    stop_words = {'a', 'an', 'the', 'and', 'is', 'are', 'in', 'this', 'of', 'with'}
    words = [word for word in prediction.split() if word not in stop_words]
    
    # Удаляем дубликаты и сортируем для единообразия
    words = sorted(list(set(words)))
    
    return ' '.join(words)

def calculate_accuracy(results):
    total_samples = len(results)
    correct_predictions = 0
    partial_predictions = 0  # Для частично правильных ответов
    
    for result in results:
        gt = set(result['ground_truth'].lower().split())
        pred = set(clean_prediction(result['prediction']).split())
        
        # Полное совпадение
        if gt == pred:
            correct_predictions += 1
        # Частичное совпадение (хотя бы один объект угадан)
        elif not gt.isdisjoint(pred):
            partial_predictions += 1
    
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    partial_accuracy = (correct_predictions + partial_predictions) / total_samples if total_samples > 0 else 0
    
    return {
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        'partial_predictions': partial_predictions,
        'accuracy': accuracy,
        'partial_accuracy': partial_accuracy
    }

def evaluate_results(results):
    metrics = calculate_accuracy(results)
    return metrics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset_dir = "../datasets/VQA"
    llama_dir = "../LLaMA-7B"
    json_path = os.path.join(dataset_dir, "ques.json")
    model_name = "CAPTION-7B"  # choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    res_json = os.path.join(dataset_dir, "imagebind_vqa_results.json")
    stat_json = os.path.join(dataset_dir, "imagebind_vqa_statistics.json")
    
    print(f"Loading model: {model_name}...")
    model, preprocess = llama.load(model_name, llama_dir, device)
    model.eval()
    
    print("Running inference on VQA dataset...")
    results = run_inference_on_vqa(dataset_dir, json_path, model, preprocess, device)
    metrics = evaluate_results(results)

    print("\nResults Summary:")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Partial predictions: {metrics['partial_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Partial accuracy: {metrics['partial_accuracy']:.4f}")
    
    output_results = [{
        'image_path': r['image_path'],
        'ground_truth': r['ground_truth'],
        'prediction': r['prediction'],
        'cleaned_prediction': clean_prediction(r['prediction'])
    } for r in results]
    
    with open(stat_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(res_json, 'w') as f:
        json.dump({
            'results': output_results
        }, f, indent=2)

if __name__ == "__main__":
    main()
