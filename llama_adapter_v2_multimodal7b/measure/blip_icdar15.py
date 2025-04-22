import sys
from pathlib import Path
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import glob

# Установка пользовательской директории для кеша Hugging Face
cache_dir = 'zhdanov/llm_test/LLaMA-Adapter/llama_adapter_v2_multimodal7b/measure/huggingface_cache'
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')


def load_icdar2015_test_data(gt_dir, image_dir):
    """
    Load the ICDAR2015 test data - words only
    Ground truth format (per line): x1,y1,x2,y2,x3,y3,x4,y4,text
    """
    dataset = []
    word_count = 0
    
    # Get all ground truth files
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "gt_img_*.txt")))
    
    for gt_file in gt_files:
        # Extract image ID from ground truth filename
        img_id = os.path.basename(gt_file).replace("gt_img_", "").replace(".txt", "")
        img_name = f"img_{img_id}.jpg"
        img_path = os.path.join(image_dir, img_name)
        
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} not found")
            continue
        
        # Read ground truth file
        words = []
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Parse line: x1,y1,x2,y2,x3,y3,x4,y4,text
            parts = line.split(',')
            if len(parts) < 9:  # Need at least 8 coordinates + text
                print(f"Warning: Skipping malformed line: {line}")
                continue
                
            # Extract text (everything after the 8 coordinates, joined by commas if text contains commas)
            text = ','.join(parts[8:])
            words.append(text)
            word_count += 1
        
        entry = {
            'img_name': img_name,
            'img_path': img_path,
            'ground_truth': words  # All words in this image
        }
        dataset.append(entry)
    
    print(f"Loaded {len(dataset)} images with {word_count} total words")
    return dataset

def run_inference_on_icdar2015(dataset_dir, processor, model, device, num_samples=None):
    """Инференс на данных ICDAR2013 с модификациями для OCR"""
    gt_dir = os.path.join(dataset_dir, "gt")
    image_dir = os.path.join(dataset_dir, "images")
    test_data = load_icdar2015_test_data(gt_dir, image_dir)
    
    if num_samples is not None:
        test_data = test_data[:num_samples]
    
    results = []
    
    for sample in tqdm(test_data, desc="Running inference"):
        img_path = os.path.join(image_dir, sample['img_name'])
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Модифицируем промпт для получения только текста
            prompt = "Question: What is the written word. Answer:"
            
            inputs = processor(image, text=prompt, return_tensors="pt").to(device)
            
            # Генерация с ограничением длины и температурой
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=20,  # Ограничиваем длину вывода
                    num_beams=3,         # Используем beam search
                    temperature=0.1,     # Понижаем температуру для детерминированности
                    early_stopping=True
                )
                
            prediction = processor.decode(output[0], skip_special_tokens=True)
            
            # Удаляем промпт из ответа
            prediction = prediction.replace(prompt, "").strip()
            
            results.append({
                'img_path': img_path,
                'ground_truth': sample['ground_truth'],
                'prediction': prediction
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return results

def clean_prediction(prediction):
    """Очистка предсказания для сравнения с GT"""
    # Удаляем пунктуацию и лишние пробелы
    prediction = prediction.strip().lower()
    for punc in ',.?!:;"\'()[]{}':
        prediction = prediction.replace(punc, '')
    return ' '.join(prediction.split())

def calculate_word_accuracy(results):
    total_words = len(results)
    correct_words = 0
    
    for result in results:
        gt = clean_prediction(' '.join(result['ground_truth']))
        pred = clean_prediction(result['prediction'])
        
        if gt.lower() == pred.lower():
            correct_words += 1
    
    word_accuracy = correct_words / total_words if total_words > 0 else 0
    
    return {
        'total_words': total_words,
        'correct_words': correct_words,
        'word_accuracy': word_accuracy
    }

def evaluate_results(results):
    total = len(results)
    word_acc_metrics = calculate_word_accuracy(results)
    
    metrics = {
        'total_samples': total,
        'word_accuracy_metrics': word_acc_metrics
    }
    
    return metrics

def main():
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, 'transformers'), exist_ok=True)
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset_dir = "../datasets/icdar2015"
    res_json = os.path.join(dataset_dir, "blip_results_advanced.json")
    stat_json = os.path.join(dataset_dir, "blip_statistics_advanced.json")
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("Loading BLIP model...")
    model_name = "Salesforce/blip2-opt-2.7b"
    
    processor = Blip2Processor.from_pretrained(model_name, cache_dir=cache_dir)
    
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=cache_dir
    ).to(device)
    
    print("Running inference on WordArt dataset...")
    results = run_inference_on_icdar2015(dataset_dir, processor, model, device)
    
    metrics = evaluate_results(results)
    word_acc = metrics['word_accuracy_metrics']

    print("\nResults Summary:")
    print(f"Total samples: {metrics['total_samples']}")
    print("\nWord Accuracy Metrics:")
    print(f"Total words: {word_acc['total_words']}")
    print(f"Correct words: {word_acc['correct_words']}")
    print(f"Word accuracy: {word_acc['word_accuracy']:.4f}")
    
    output_results = [{
        'img_path': r['img_path'],
        'ground_truth': r['ground_truth'],
        'prediction': r['prediction'],
        'cleaned_prediction': clean_prediction(r['prediction'])
    } for r in results]
    
    with open(stat_json, 'w') as f:
        json.dump({
            'metrics': {
                'total_samples': metrics['total_samples'],
                'word_accuracy': word_acc['word_accuracy']
            }
        }, f, indent=2)
    
    with open(res_json, 'w') as f:
        json.dump({
            'results': output_results
        }, f, indent=2)

if __name__ == "__main__":
    main()
