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
from PIL import Image
from tqdm import tqdm
import glob

def load_icdar2015_test_data(gt_dir, image_dir):
    """
    Load the ICDAR2015 test data - words only
    Ground truth format (per line): x1,y1,x2,y2,x3,y3,x4,y4,text
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
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Parse line: x1,y1,x2,y2,x3,y3,x4,y4,text
            parts = line
                
            # Extract text (everything after the 8 coordinates, joined by commas if text contains commas)
            text = ','.join(parts)
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
    Run inference on the ICDAR2015 dataset - whole images
    
    Args:
        dataset_dir (str): Directory containing the dataset
        model: The LLaMA model
        preprocess: Image preprocessing function
        device: Device to run inference on
        num_samples (int, optional): Number of samples to process. If None, process all.
    
    Returns:
        list: Results containing image paths, ground truths, and model predictions
    """
    # Load test data
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
            
            # Create prompt for the model
            prompt = llama.format_prompt('Extract and transcribe all visible text from this image exactly as it appears.  \
- Include all words, numbers, and symbols in their original form.  \
- Preserve line breaks and spacing where relevant.  \
- Skip any non-text elements, artifacts, or image noise. - Output only the extracted text, without additional comments or formatting.  ')

            # Generate prediction
            prediction = model.generate(img_tensor, [prompt])[0]
            
            # Store results
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

    total_words = len(results)
    correct_words = 0
    
    for result in results:
        gt = clean_prediction(' '.join(result['ground_truth']))
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
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define paths
    dataset_dir = "../datasets/cute80"
    llama_dir = "../LLaMA-7B"
    model_name = "CAPTION-7B"  # choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    res_json = "../datasets/cute80/generate_markup_adv.json"
    stat_json = "../datasets/cute80/statistic_adv.json"
    
    # Load model
    print(f"Loading model: {model_name}...")
    model, preprocess = llama.load(model_name, llama_dir, device)
    model.eval()
    
    # Run inference
    print("Running inference on IIIT5K dataset...")
    results = run_inference_on_icdar2015(dataset_dir, model, preprocess, device)
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