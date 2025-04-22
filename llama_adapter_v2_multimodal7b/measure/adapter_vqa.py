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

# Add parent directory to system path
parent_dir = str(Path(__file__).parent.parent)  # Two levels up: .parent.parent
sys.path.append(parent_dir)

def load_vqa_data(json_path):
    """
    Load VQA dataset from JSON file.
    
    Args:
        json_path: Path to JSON annotation file
        
    Returns:
        List of dictionaries containing image metadata and ground truth classes
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    dataset = []
    processed_images = set()
    
    for item in data['data']:
        image_id = item['image_id']
        
        # Use only one entry per image
        if image_id in processed_images:
            continue
            
        processed_images.add(image_id)
        
        # Use class list as ground truth
        ground_truth = ' '.join(item['image_classes'])
        
        entry = {
            'image_id': image_id,
            'ground_truth': ground_truth,
            'image_path': f"../datasets/VQA/test_images/{image_id}.jpg"
        }
        dataset.append(entry)
    
    return dataset

def run_inference_on_vqa(dataset_dir, json_path, model, preprocess, device, num_samples=None):
    """
    Run inference on VQA dataset using ImageBind.
    
    Args:
        dataset_dir: Path to dataset directory
        json_path: Path to JSON annotations
        model: ImageBind model
        preprocess: Image preprocessing function
        device: Device for inference
        num_samples: Number of samples to process (None for all)
        
    Returns:
        List of inference results
    """
    test_data = load_vqa_data(json_path)
    
    if num_samples is not None:
        test_data = test_data[:num_samples]
    
    results = []
    
    for sample in tqdm(test_data, desc="Running inference"):
        img_path = sample['image_path']
        
        try:
            # Read and preprocess image
            img = Image.fromarray(cv2.imread(img_path))
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # Create optimized prompt for object identification
            prompt = llama.format_prompt(
                'Identify ONLY the two most prominent objects in this image. '
                'List them separated by a single space, nothing else. '
                'Examples:\n'
                '- For laptop and phone: "laptop phone"\n'
                '- For car and tree: "car tree"\n'
                'Now analyze this image and provide JUST the objects.'
            )
            
            # Generate prediction
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
    """
    Clean and normalize prediction text.
    
    Args:
        prediction: Raw model output
        
    Returns:
        Normalized string of unique words
    """
    prediction = prediction.lower().strip()
    
    # Keep only letters and spaces
    prediction = re.sub(r'[^a-z\s]', '', prediction)
    
    # Remove common stop words
    stop_words = {'a', 'an', 'the', 'and', 'is', 'are', 'in', 'this', 'of', 'with'}
    words = [word for word in prediction.split() if word not in stop_words]
    
    # Deduplicate and sort for consistency
    words = sorted(list(set(words)))
    
    return ' '.join(words)

def calculate_accuracy(results):
    """
    Calculate accuracy metrics for VQA results.
    
    Args:
        results: List of inference results
        
    Returns:
        Dictionary containing accuracy metrics
    """
    total_samples = len(results)
    correct_predictions = 0
    partial_predictions = 0  # For partially correct answers
    
    for result in results:
        gt = set(result['ground_truth'].lower().split())
        pred = set(clean_prediction(result['prediction']).split())
        
        # Full match
        if gt == pred:
            correct_predictions += 1
        # Partial match (at least one correct object)
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
    """
    Evaluate and aggregate VQA results.
    
    Args:
        results: List of inference results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    return calculate_accuracy(results)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset_dir = "../datasets/VQA"
    llama_dir = "../LLaMA-7B"
    json_path = os.path.join(dataset_dir, "ques.json")
    model_name = "CAPTION-7B"  # Options: BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
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