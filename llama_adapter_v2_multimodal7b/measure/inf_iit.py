import sys
from pathlib import Path

# Получаем путь к родительской директории текущего файла
parent_dir = str(Path(__file__).parent.parent)  # На два уровня выше: .parent.parent

# Добавляем путь в sys.path
sys.path.append(parent_dir)


import os
import cv2
import llama
import torch
import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

def load_iiit5k_test_data(mat_file_path):
    """
    Load the IIIT5K test data from the .mat file
    """
    test_data = sio.loadmat(mat_file_path)['testdata'][0]
    dataset = []
    
    for i in range(len(test_data)):
        entry = {
            'img_name': str(test_data[i][0][0]),
            'ground_truth': str(test_data[i][1][0])
        }
        dataset.append(entry)
    
    return dataset

def run_inference_on_dataset(dataset_dir, model, preprocess, device, num_samples=None):

    test_data = load_iiit5k_test_data(os.path.join(dataset_dir, 'testdata.mat'))
    if num_samples is not None:
        test_data = test_data[:num_samples]
    results = []
    
    for sample in tqdm(test_data, desc="Running inference"):
        img_path = os.path.join(dataset_dir, sample['img_name'])
        
        try:
            img = Image.fromarray(cv2.imread(img_path))
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # prompt = llama.format_prompt('Transcribe all and only text in this image. Write only one word: nothing else')

            prompt = llama.format_prompt('Extract and transcribe all visible text from this image exactly as it appears.  \
- Include all words, numbers, and symbols in their original form.  \
- Preserve line breaks and spacing where relevant.  \
- Skip any non-text elements, artifacts, or image noise. - Output only the extracted text, without additional comments or formatting.  ')

            prediction = model.generate(img_tensor, [prompt])[0]
            
            results.append({
                'img_path': img_path,
                'ground_truth': sample['ground_truth'],
                'prediction': prediction
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return results

def clean_prediction(prediction):
    """
    Extract the actual word from the model's prediction
    """
    # First try to find the word directly
    pred_words = prediction.strip().split()
    if not pred_words:
        return ""
    
    candidate_words = [word.strip(',.?!:;"\'()[]{}') for word in pred_words]
    candidate_words = [word for word in candidate_words if word]  # Remove empty strings
    
    
    return ' '.join(candidate_words)

def calculate_word_accuracy(results):
    # total_words = 0
    # correct_words = 0
    
    # for result in results:
    #     gt = result['ground_truth'].strip()
    #     pred = clean_prediction(result['prediction'])
        
    #     print(gt.lower(), pred.lower(), sep='//')
    #     pred = pred.lower().split()
    #     total_words += len(gt.lower().split())
    #     for i in gt.lower().split():
    #         if i in pred:
    #             correct_words += 1
    #             print(True)

    total_words = len(results)
    correct_words = 0
    
    for result in results:
        gt = result['ground_truth'].strip()
        pred = clean_prediction(result['prediction'])
        
        print(gt.lower(), pred.lower(), sep='//')
        if gt.lower() == pred.lower():
            correct_words += 1
    
    word_accuracy = correct_words / total_words
    
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset_dir = "../datasets/IIIT5K"
    llama_dir = "../LLaMA-7B"
    model_name = "CAPTION-7B"  # choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    res_json = "../datasets/IIIT5K/generate_markup_advanced.json"
    stat_json = "../datasets/IIIT5K/statistic_advanced.json"
    
    print(f"Loading model: {model_name}...")
    model, preprocess = llama.load(model_name, llama_dir, device)
    model.eval()
    
    print("Running inference on IIIT5K dataset...")
    results = run_inference_on_dataset(dataset_dir, model, preprocess, device)
    metrics = evaluate_results(results)
    word_acc = metrics['word_accuracy_metrics']

    print("\nBasic Results:")
    print(f"Total samples: {metrics['total_samples']}")
    print("\nWord Accuracy Metrics:")
    print(f"Total words: {word_acc['total_words']}")
    print(f"Word accuracy: {word_acc['word_accuracy']:.4f}")
    
    import json
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