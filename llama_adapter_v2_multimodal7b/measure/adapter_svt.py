import sys
from pathlib import Path
import os
import cv2
import llama
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import json

# Add parent directory to system path
parent_dir = str(Path(__file__).parent.parent)  # Two levels up: .parent.parent
sys.path.append(parent_dir)

def load_icdar2015_test_data(gt_dir, image_dir):
    """
    Load ICDAR2015 test data (words only).
    
    Args:
        gt_dir: Directory containing ground truth text files
        image_dir: Directory containing images
        
    Returns:
        List of dictionaries containing image data and ground truth words
    """
    dataset = []
    word_count = 0
    
    # Get all ground truth files
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    
    for gt_file in gt_files:
        # Extract image ID from ground truth filename
        img_id = os.path.basename(gt_file).replace(".txt", "")
        img_name = f"{img_id}.jpg"
        img_path = os.path.join(image_dir, img_name)
        
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} not found")
            continue
        
        # Read ground truth file
        words = []
        with open(gt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Parse line: x1,y1,x2,y2,x3,y3,x4,y4,text
            parts = line.split(',')
            if len(parts) < 9:  # Skip invalid lines
                continue
            text = ','.join(parts[8:])  # Join remaining parts in case text contains commas
            words.append(text)
            word_count += 1
        
        entry = {
            'img_name': img_name,
            'img_path': img_path,
            'words': words  # All words in this image
        }
        dataset.append(entry)
    
    print(f"Loaded {len(dataset)} images with {word_count} total words")
    return dataset

def run_inference_on_icdar2015(dataset_dir, model, preprocess, device, num_samples=None):
    """
    Run LLaMA inference on ICDAR2015 dataset (whole images).
    
    Args:
        dataset_dir: Directory containing the dataset
        model: LLaMA model
        preprocess: Image preprocessing function
        device: Device for inference
        num_samples: Number of samples to process (None for all)
        
    Returns:
        List of inference results
    """
    gt_dir = os.path.join(dataset_dir, "gt")
    image_dir = os.path.join(dataset_dir, "images")
    test_data = load_icdar2015_test_data(gt_dir, image_dir)
    
    if num_samples is not None:
        test_data = test_data[:num_samples]
    
    results = []
    
    for sample in tqdm(test_data, desc="Running inference"):
        try:
            # Read and preprocess the entire image
            img = Image.fromarray(cv2.imread(sample['img_path']))
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # Create optimized prompt for text extraction
            prompt = llama.format_prompt(
                'Extract and transcribe all visible text from this image exactly as it appears.\n'
                '- Include all words, numbers, and symbols in their original form.\n'
                '- Preserve line breaks and spacing where relevant.\n'
                '- Skip any non-text elements, artifacts, or image noise.\n'
                '- Output only the extracted text, without additional comments or formatting.'
            )

            # Generate prediction
            prediction = model.generate(img_tensor, [prompt])[0]
            
            results.append({
                'img_path': sample['img_path'],
                'ground_truth': sample['words'],
                'prediction': prediction
            })
            
        except Exception as e:
            print(f"Error processing {sample['img_path']}: {e}")
    
    return results

def clean_prediction(prediction):
    """
    Clean prediction text for comparison with ground truth.
    
    Args:
        prediction: Raw model output
        
    Returns:
        Normalized text string
    """
    prediction = prediction.strip().lower()
    for punc in ',.?!:;"\'()[]{}':
        prediction = prediction.replace(punc, '')
    return ' '.join(prediction.split())

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
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define paths
    dataset_dir = "../datasets/cute80"
    llama_dir = "../LLaMA-7B"
    model_name = "CAPTION-7B"  # Options: BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    res_json = "../datasets/cute80/generate_markup_adv.json"
    stat_json = "../datasets/cute80/statistic_adv.json"
    
    # Load model
    print(f"Loading model: {model_name}...")
    model, preprocess = llama.load(model_name, llama_dir, device)
    model.eval()
    
    # Run inference
    print("Running inference on ICDAR2015 dataset...")
    results = run_inference_on_icdar2015(dataset_dir, model, preprocess, device)
    metrics = evaluate_results(results)
    word_acc = metrics['word_accuracy_metrics']

    print("\nResults Summary:")
    print(f"Total samples: {metrics['total_samples']}")
    print("\nWord Accuracy Metrics:")
    print(f"Total words: {word_acc['total_words']}")
    print(f"Correct words: {word_acc['correct_words']}")
    print(f"Word accuracy: {word_acc['word_accuracy']:.4f}")
    
    # Prepare and save results
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