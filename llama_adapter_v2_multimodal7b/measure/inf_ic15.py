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
            prompt = llama.format_prompt('Transcribe the text in this image.')
            
            # Generate prediction
            prediction = model.generate(img_tensor, [prompt])[0]
            
            # Store results
            results.append({
                'img_path': sample['img_path'],
                'gt_words': sample['words'],
                'prediction': prediction
            })
            
        except Exception as e:
            print(f"Error processing {sample['img_path']}: {e}")
    
    return results

def calculate_word_accuracy(results):
    """
    Calculate word accuracy for OCR results
    
    For each image, check how many ground truth words are found in the prediction
    """
    total_words = 0
    found_words = 0
    
    for result in results:
        gt_words = result['gt_words']
        prediction = result['prediction'].lower()
        
        total_words += len(gt_words)
        
        for word in gt_words:
            # Check if the ground truth word appears in the prediction (case-insensitive)
            if word.lower() in prediction:
                found_words += 1
    
    word_accuracy = found_words / total_words if total_words > 0 else 0
    
    return {
        'total_words': total_words,
        'found_words': found_words,
        'word_accuracy': word_accuracy
    }

def calculate_strict_word_accuracy(results):
    """
    Calculate strict word accuracy for OCR results
    
    For each image, check how many ground truth words are found as separate words in the prediction
    """
    total_words = 0
    found_words = 0
    
    for result in results:
        gt_words = result['gt_words']
        prediction = result['prediction'].lower()
        
        # Extract words from prediction
        pred_words = set()
        for word in prediction.split():
            # Clean up punctuation
            cleaned_word = word.strip(',.?!:;"\'()[]{}')
            if cleaned_word:
                pred_words.add(cleaned_word.lower())
        
        total_words += len(gt_words)
        
        for word in gt_words:
            # Check if the ground truth word appears as a separate word in the prediction
            if word.lower() in pred_words:
                found_words += 1
    
    strict_word_accuracy = found_words / total_words if total_words > 0 else 0
    
    return {
        'total_words': total_words,
        'found_words': found_words,
        'strict_word_accuracy': strict_word_accuracy
    }

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define paths
    dataset_dir = "datasets/icdar2015"
    llama_dir = "LLaMA-7B"
    model_name = "CAPTION-7B"  # choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    
    # Load model
    print(f"Loading model: {model_name}...")
    model, preprocess = llama.load(model_name, llama_dir, device)
    model.eval()
    
    # Run inference
    print("Running inference on ICDAR 2015 dataset...")
    results = run_inference_on_icdar2015(dataset_dir, model, preprocess, device)
    
    # Calculate word accuracy
    word_acc = calculate_word_accuracy(results)
    print("\nWord Accuracy Metrics:")
    print(f"Total words: {word_acc['total_words']}")
    print(f"Found words: {word_acc['found_words']}")
    print(f"Word accuracy: {word_acc['word_accuracy']:.4f}")
    
    # Calculate strict word accuracy (as separate words)
    strict_word_acc = calculate_strict_word_accuracy(results)
    print("\nStrict Word Accuracy Metrics:")
    print(f"Total words: {strict_word_acc['total_words']}")
    print(f"Found words (as separate words): {strict_word_acc['found_words']}")
    print(f"Strict word accuracy: {strict_word_acc['strict_word_accuracy']:.4f}")
    
    # Save results
    import json
    output_results = [{
        'img_path': r['img_path'],
        'gt_words': r['gt_words'],
        'prediction': r['prediction']
    } for r in results]
    
    with open('llama_icdar2015_results.json', 'w') as f:
        json.dump({
            'results': output_results,
            'metrics': {
                'word_accuracy': word_acc['word_accuracy'],
                'strict_word_accuracy': strict_word_acc['strict_word_accuracy']
            }
        }, f, indent=2)
    
    print("Results saved to llama_icdar2015_results.json")

if __name__ == "__main__":
    main()