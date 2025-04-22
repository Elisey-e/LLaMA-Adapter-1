import sys
from pathlib import Path
import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from pandagpt import PandaGPT

def load_icdar2013_test_data(gt_file_path):
    """
    Load the ICDAR2013 test data from the GT text file
    Format: word_1.png, "Tiredness"
           word_2.png, "kills"
           ...
    """
    dataset = []
    
    with open(gt_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Each line has format: image_name, "ground_truth"
        parts = line.split(', "')
        if len(parts) != 2:
            print(f"Warning: Skipping malformed line: {line}")
            continue
            
        img_name = parts[0].strip()
        ground_truth = parts[1].strip('"').strip()
        
        entry = {
            'img_name': img_name,
            'ground_truth': ground_truth
        }
        dataset.append(entry)
    
    return dataset

def run_inference_on_icdar2013(dataset_dir, gt_file_path, image_dir, model, device, num_samples=None):
    """
    Run inference on the ICDAR2013 dataset using PandaGPT
    
    Args:
        dataset_dir (str): Directory containing the dataset
        gt_file_path (str): Path to the ground truth file
        image_dir (str): Directory containing images
        model: The PandaGPT model
        device: Device to run inference on
        num_samples (int, optional): Number of samples to process. If None, process all.
    
    Returns:
        list: Results containing image paths, ground truths, and model predictions
    """
    # Load test data
    test_data = load_icdar2013_test_data(gt_file_path)
    
    if num_samples is not None:
        test_data = test_data[:num_samples]
    
    results = []
    
    for sample in tqdm(test_data, desc="Running inference"):
        img_path = os.path.join(image_dir, sample['img_name'])
        
        try:
            # Read the image
            img = Image.open(img_path).convert('RGB')
            
            # Create prompt for the model
            prompt = "Extract and transcribe all visible text from this image exactly as it appears. " \
                    "- Include all words, numbers, and symbols in their original form. " \
                    "- Preserve line breaks and spacing where relevant. " \
                    "- Skip any non-text elements, artifacts, or image noise. " \
                    "- Output only the extracted text, without additional comments or formatting."
            
            # Generate prediction
            prediction = model.generate(img, prompt)
            
            # Store results
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
    Clean the model's prediction by removing extra formatting and keeping only the text content
    """
    # Remove common artifacts and formatting
    prediction = prediction.strip()
    
    # Remove quotation marks if present
    if prediction.startswith('"') and prediction.endswith('"'):
        prediction = prediction[1:-1]
    
    # Remove any remaining special formatting tokens
    for token in ['<s>', '</s>', '[IMG]', '[/IMG]']:
        prediction = prediction.replace(token, '')
    
    # Normalize whitespace
    prediction = ' '.join(prediction.split())
    
    return prediction

def calculate_word_accuracy(results):
    total_words = len(results)
    correct_words = 0
    
    for result in results:
        gt = result['ground_truth'].strip()
        pred = clean_prediction(result['prediction'])
        
        print(f"GT: {gt.lower()} || PRED: {pred.lower()}")
        if gt.lower() == pred.lower():
            correct_words += 1
    
    word_accuracy = correct_words / total_words if total_words > 0 else 0.0
    
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
    
    # Configuration paths
    dataset_dir = "../datasets/icdar2013"
    gt_file_path = os.path.join(dataset_dir, "Challenge2_Test_Task3_GT.txt")
    image_dir = os.path.join(dataset_dir, "images")
    res_json = "../datasets/icdar2013/pandagpt_results.json"
    stat_json = "../datasets/icdar2013/pandagpt_statistics.json"
    
    # Initialize PandaGPT model
    print("Loading PandaGPT model...")
    model = PandaGPT(device=device)
    
    print("Running inference on ICDAR2013 dataset...")
    results = run_inference_on_icdar2013(dataset_dir, gt_file_path, image_dir, model, device)
    metrics = evaluate_results(results)
    word_acc = metrics['word_accuracy_metrics']

    print("\nEvaluation Results:")
    print(f"Total samples: {metrics['total_samples']}")
    print("\nWord Accuracy Metrics:")
    print(f"Total words: {word_acc['total_words']}")
    print(f"Word accuracy: {word_acc['word_accuracy']:.4f}")
    
    # Save results
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