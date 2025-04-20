import sys
from pathlib import Path
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import re
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Установка пользовательской директории для кеша Hugging Face
cache_dir = '/beta/projects/hyperflex/code/zhdanov/llm_test/LLaMA-Adapter/llama_adapter_v2_multimodal7b/measure/huggingface_cache'
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')

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

def run_inference_on_vqa(dataset_dir, json_path, processor, model, device, num_samples=None):
    """Инференс на данных VQA"""
    test_data = load_vqa_data(json_path)
    
    if num_samples is not None:
        test_data = test_data[:num_samples]
    
    results = []
    
    for sample in tqdm(test_data, desc="Running inference"):
        img_path = sample['image_path']
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Более строгий и конкретный промпт
            prompt = "Identify ONLY the two most prominent objects in this image. List them separated by a single space, nothing else. Example: 'computer phone'"
            
            inputs = processor(image, text=prompt, return_tensors="pt").to(device)
            
            # Генерация с более строгими параметрами
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    num_beams=5,
                    temperature=0.01,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
                
            prediction = processor.decode(output[0], skip_special_tokens=True)
            
            # Удаляем весь текст до последнего двоеточия (если модель повторяет часть промпта)
            if ':' in prediction:
                prediction = prediction.split(':')[-1].strip()
            
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
        'predictions': partial_predictions,
        'accuracy': partial_accuracy
    }

def evaluate_results(results):
    metrics = calculate_accuracy(results)
    
    return metrics

def main():
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, 'transformers'), exist_ok=True)
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset_dir = "../datasets/VQA"
    json_path = os.path.join(dataset_dir, "ques.json")
    res_json = os.path.join(dataset_dir, "blip_vqa_results_improved.json")
    stat_json = os.path.join(dataset_dir, "blip_vqa_statistics_improved.json")
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("Loading BLIP model...")
    model_name = "Salesforce/blip2-opt-2.7b"
    
    processor = Blip2Processor.from_pretrained(model_name, cache_dir=cache_dir)
    
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=cache_dir
    ).to(device)
    
    print("Running inference on VQA dataset...")
    results = run_inference_on_vqa(dataset_dir, json_path, processor, model, device)
    
    metrics = evaluate_results(results)

    print("\nResults Summary:")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Correct predictions: {metrics['predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")\
    
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