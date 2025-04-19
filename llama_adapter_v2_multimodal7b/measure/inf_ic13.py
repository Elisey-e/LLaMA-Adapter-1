import os
import cv2
import llama
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

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

def run_inference_on_icdar2013(dataset_dir, gt_file_path, image_dir, model, preprocess, device, num_samples=None):
    """
    Run inference on the ICDAR2013 dataset
    
    Args:
        dataset_dir (str): Directory containing the dataset
        gt_file_path (str): Path to the ground truth file
        image_dir (str): Directory containing images
        model: The LLaMA model
        preprocess: Image preprocessing function
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
            # Read and preprocess the image
            img = Image.fromarray(cv2.imread(img_path))
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # Create prompt for the model
            prompt = llama.format_prompt('Transcribe the text in this image.')
            
            # Generate prediction
            prediction = model.generate(img_tensor, [prompt])[0]
            
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
    Extract the actual word from the model's prediction
    """
    # First try to find the word directly
    pred_words = prediction.strip().split()
    if not pred_words:
        return ""
    
    # Look for the shortest word as it's likely the transcribed text
    # This is a heuristic and might need to be adjusted based on model output patterns
    candidate_words = [word.strip(',.?!:;"\'()[]{}') for word in pred_words]
    candidate_words = [word for word in candidate_words if word]  # Remove empty strings
    
    if not candidate_words:
        return ""
    
    # Return the shortest word that's not a common article or preposition
    stopwords = {"the", "a", "an", "in", "on", "at", "to", "is", "are", "of"}
    filtered_candidates = [w for w in candidate_words if w.lower() not in stopwords]
    
    if filtered_candidates:
        return min(filtered_candidates, key=len)
    else:
        return min(candidate_words, key=len)

def calculate_word_accuracy(results):
    """
    Calculate word accuracy for OCR results
    
    Word accuracy = (number of correctly recognized words) / (total number of words)
    """
    total_words = len(results)
    correct_words = 0
    
    for result in results:
        gt = result['ground_truth'].strip()
        pred = clean_prediction(result['prediction'])
        
        if gt.lower() == pred.lower():  # Case-insensitive comparison
            correct_words += 1
    
    word_accuracy = correct_words / total_words if total_words > 0 else 0
    
    return {
        'total_words': total_words,
        'correct_words': correct_words,
        'word_accuracy': word_accuracy
    }

def evaluate_results(results):
    """
    Calculate accuracy metrics
    """
    total = len(results)
    exact_matches = 0
    case_insensitive_matches = 0
    
    for result in results:
        gt = result['ground_truth'].strip()
        pred = clean_prediction(result['prediction'])
        
        if gt == pred:
            exact_matches += 1
        if gt.lower() == pred.lower():
            case_insensitive_matches += 1
    
    # Calculate word accuracy using the dedicated function
    word_acc_metrics = calculate_word_accuracy(results)
    
    metrics = {
        'total_samples': total,
        'exact_match_accuracy': exact_matches / total if total > 0 else 0,
        'case_insensitive_accuracy': case_insensitive_matches / total if total > 0 else 0,
        'word_accuracy_metrics': word_acc_metrics
    }
    
    return metrics

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define paths
    dataset_dir = "datasets/icdar2013"
    gt_file_path = os.path.join(dataset_dir, "Challenge2_Test_Task3_GT.txt")
    image_dir = os.path.join(dataset_dir, "images")
    llama_dir = "LLaMA-7B"
    model_name = "CAPTION-7B"  # choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    
    # Load model
    print(f"Loading model: {model_name}...")
    model, preprocess = llama.load(model_name, llama_dir, device)
    model.eval()
    
    # Run inference
    print("Running inference on ICDAR 2013 dataset...")
    results = run_inference_on_icdar2013(dataset_dir, gt_file_path, image_dir, model, preprocess, device)
    
    # Evaluate basic results
    metrics = evaluate_results(results)
    print("\nBasic Results:")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Exact match accuracy: {metrics['exact_match_accuracy']:.4f}")
    print(f"Case-insensitive accuracy: {metrics['case_insensitive_accuracy']:.4f}")
    
    # Print word accuracy metrics
    word_acc = metrics['word_accuracy_metrics']
    print("\nWord Accuracy Metrics:")
    print(f"Total words: {word_acc['total_words']}")
    print(f"Correctly recognized words: {word_acc['correct_words']}")
    print(f"Word accuracy: {word_acc['word_accuracy']:.4f}")
    
    # Save results
    import json
    output_results = [{
        'img_path': r['img_path'],
        'ground_truth': r['ground_truth'],
        'prediction': r['prediction'],
        'cleaned_prediction': clean_prediction(r['prediction'])
    } for r in results]
    
    with open('llama_icdar2013_results.json', 'w') as f:
        json.dump({
            'results': output_results,
            'metrics': {
                'basic': {
                    'total_samples': metrics['total_samples'],
                    'exact_match_accuracy': metrics['exact_match_accuracy'],
                    'case_insensitive_accuracy': metrics['case_insensitive_accuracy']
                },
                'word_accuracy': word_acc['word_accuracy']
            }
        }, f, indent=2)
    
    print("Results saved to llama_icdar2013_results.json")

if __name__ == "__main__":
    main()