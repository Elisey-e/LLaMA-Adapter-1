import sys
from pathlib import Path

# Get parent directory path
parent_dir = str(Path(__file__).parent.parent)  # Two levels up: .parent.parent
sys.path.append(parent_dir)

import os
import cv2
import llama
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

def load_icdar2013_test_data(gt_file_path):
    """
    Load ICDAR2013 test data from annotation file.
    
    Args:
        gt_file_path: Path to ground truth file
        
    Returns:
        List of dicts containing image names and ground truth text
    """
    dataset = []
    
    with open(gt_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        parts = line.split(' ')
        if len(parts) != 2:
            print(f"Warning: Skipping malformed line: {line}")
            continue
            
        img_name = parts[0].strip()
        ground_truth = parts[1].strip()
        
        entry = {
            'img_name': img_name,
            'ground_truth': ground_truth
        }
        dataset.append(entry)
    
    return dataset

def run_inference_on_icdar2013(dataset_dir, gt_file_path, image_dir, model, preprocess, device, num_samples=None):
    """
    Run inference on ICDAR2013 dataset using LLaMA model.
    
    Args:
        dataset_dir: Dataset directory path
        gt_file_path: Ground truth file path
        image_dir: Directory containing images
        model: LLaMA model
        preprocess: Image preprocessing function
        device: Device for inference (cuda/cpu)
        num_samples: Number of samples to process (None for all)
        
    Returns:
        List of inference results
    """
    test_data = load_icdar2013_test_data(gt_file_path)
    
    if num_samples is not None:
        test_data = test_data[:num_samples]
    
    results = []
    
    for sample in tqdm(test_data, desc="Running inference"):
        img_path = os.path.join(image_dir, sample['img_name'])
        
        try:
            img = Image.fromarray(cv2.imread(img_path))
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            prompt = llama.format_prompt(
                'Extract and transcribe all visible text from this image exactly as it appears. '
                '- Include all words, numbers, and symbols in their original form. '
                '- Preserve line breaks and spacing where relevant. '
                '- Skip any non-text elements, artifacts, or image noise. '
                '- Output only the extracted text, without additional comments or formatting.'
            )

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
    Clean model prediction for comparison with ground truth.
    
    Args:
        prediction: Raw model output
        
    Returns:
        Cleaned text string
    """
    pred_words = prediction.strip().split()
    if not pred_words:
        return ""
    
    candidate_words = [word.strip(',.?!:;"\'()[]{}') for word in pred_words]
    candidate_words = [word for word in candidate_words if word]  # Remove empty strings
    
    return ' '.join(candidate_words)

def calculate_word_accuracy(results):
    """
    Calculate word-level accuracy metrics.
    
    Args:
        results: List of inference results
        
    Returns:
        Dict containing accuracy metrics
    """
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
    """
    Evaluate and aggregate inference results.
    
    Args:
        results: List of inference results
        
    Returns:
        Dict containing evaluation metrics
    """
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
    
    dataset_dir = "../datasets/WordArt"
    llama_dir = "../LLaMA-7B"
    gt_file_path = os.path.join(dataset_dir, "test_label.txt")
    image_dir = os.path.join(dataset_dir, "images")
    model_name = "CAPTION-7B"  # Options: BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    res_json = "../datasets/WordArt/generate_markup.json"
    stat_json = "../datasets/WordArt/statistic.json"
    
    print(f"Loading model: {model_name}...")
    model, preprocess = llama.load(model_name, llama_dir, device)
    model.eval()
    
    print("Running inference on IIIT5K dataset...")
    results = run_inference_on_icdar2013(dataset_dir, gt_file_path, image_dir, model, preprocess, device)
    metrics = evaluate_results(results)
    word_acc = metrics['word_accuracy_metrics']

    print("\nBasic Results:")
    print(f"Total samples: {metrics['total_samples']}")
    print("\nWord Accuracy Metrics:")
    print(f"Total words: {word_acc['total_words']}")
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