import torch
import sys
import os
import argparse
import yaml
import json
from datetime import datetime
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.data import bb_zoom_dataset
from utils.model import Net

parser = argparse.ArgumentParser(description = "Testing the model on real dataset")

""" 
Example Command for testing known Images:
python Unkown_Classes_Test.py \
--model_path <MODEL_PATH> \
--img_root_dir <KNOWN_IMG_DIR> \
--labels_root_dir <KNOWN_LABEL_DIR>

Example Command for testing unkown Images:
python Unkown_Classes_Test.py \
--model_path <MODEL_PATH> \
--img_root_dir <UNKNOWN_IMG_DIR> \
--labels_root_dir <UNKNOWN_LABEL_DIR> \
--Test_unkown_data True
"""

parser.add_argument('--model_path', type=str, help='Path to the model')
parser.add_argument('--img_root_dir', type=str, help='Path to the image root directory')
parser.add_argument('--labels_root_dir', type=str, help='Path to the labels root directory')
parser.add_argument('--Test_unkown_data', type=bool, default=False, help='Test unkown data')

parser.add_argument('--threshold', type=float, default=0.8, help='Threshold for confidence score')
parser.add_argument('--Top_margin_threshold', type=float, default=0.01, help='Difference between top two')
parser.add_argument('--EnergyScore_threshold', type=float, default=0.9, help='Energy score threshold inverse relation w conf')

args = parser.parse_args()

img_root_dir = args.img_root_dir
labels_root_dir = args.labels_root_dir
threshold = args.threshold
model_path = args.model_path
test_unkown = args.Test_unkown_data
top_margin_threshold = args.Top_margin_threshold
energy_score_threshold = args.EnergyScore_threshold

print("Threshold: ", threshold)
print("Model path: ", model_path)
print("Image root dir: ", img_root_dir)
print("Labels root dir: ", labels_root_dir)
print("Test unkown data: ", test_unkown)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device);

# Load class names from data_config.yaml
config_path = '/mnt/data/talha/springhouse-svc-pytorch-classifier/data/training/data_config.yaml'
class_names = []
unknown_class_id = None

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        class_names = config.get('names', [])
    
    # Find unknown class ID
    for i, name in enumerate(class_names):
        if 'unknown' in name.lower() or name == 'Unkown':
            unknown_class_id = i
            break
    
    if unknown_class_id is not None:
        print(f"Found unknown class: ID={unknown_class_id}, Name='{class_names[unknown_class_id]}'")
    else:
        print("Warning: Unknown class not found in class names, defaulting to ID=35")
        unknown_class_id = 35
else:
    print(f"Warning: Config file not found at {config_path}, defaulting to unknown_class_id=35")
    unknown_class_id = 35

## settings for images
val_transforms = transforms.Compose([ 
    transforms.ToTensor()
])

## accessing the dataset
val_dataset = bb_zoom_dataset(
    img_root_dir=img_root_dir,
    labels_root_dir=labels_root_dir,
    img_size=224,
    data_transform=val_transforms,
    augmentation=False,
    crop=False
);

##loading in batches
val_loader = DataLoader(
    val_dataset,
    batch_size=16
)

model_data = torch.load(model_path, map_location=device);
if isinstance(model_data, dict):
    class_count = model_data.get('class_count', 36)
    if 'student_state_dict' in model_data:
        state_dict = model_data['student_state_dict']
    elif 'state_dict' in model_data:
        state_dict = model_data['state_dict']
    else:
        state_dict = model_data
    model = Net(class_count, load_weights=False)
    model.load_state_dict(state_dict, strict=False)
else:
    model = model_data
model.to(device)
model.eval()

def calculate_binary_metrics(results, unknown_class_id):
    """Calculate binary Known vs Unknown metrics"""
    true_known = 0
    true_unknown = 0
    pred_known = 0
    pred_unknown = 0
    true_known_pred_known = 0
    true_known_pred_unknown = 0
    true_unknown_pred_known = 0
    true_unknown_pred_unknown = 0
    
    for r in results:
        true_id = r['true_class_id']
        pred_id = r['pred_class_id']
        
        is_true_unknown = (true_id == unknown_class_id)
        is_pred_unknown = (pred_id == unknown_class_id)
        
        if is_true_unknown:
            true_unknown += 1
        else:
            true_known += 1
        
        if is_pred_unknown:
            pred_unknown += 1
        else:
            pred_known += 1
        
        if not is_true_unknown and not is_pred_unknown:
            true_known_pred_known += 1
        elif not is_true_unknown and is_pred_unknown:
            true_known_pred_unknown += 1
        elif is_true_unknown and not is_pred_unknown:
            true_unknown_pred_known += 1
        else:
            true_unknown_pred_unknown += 1
    
    # Calculate metrics
    known_precision = (true_known_pred_known / pred_known * 100) if pred_known > 0 else 0
    known_recall = (true_known_pred_known / true_known * 100) if true_known > 0 else 0
    known_f1 = (2 * known_precision * known_recall / (known_precision + known_recall)) if (known_precision + known_recall) > 0 else 0
    
    unknown_precision = (true_unknown_pred_unknown / pred_unknown * 100) if pred_unknown > 0 else 0
    unknown_recall = (true_unknown_pred_unknown / true_unknown * 100) if true_unknown > 0 else 0
    unknown_f1 = (2 * unknown_precision * unknown_recall / (unknown_precision + unknown_recall)) if (unknown_precision + unknown_recall) > 0 else 0
    
    overall_accuracy = ((true_known_pred_known + true_unknown_pred_unknown) / len(results) * 100) if len(results) > 0 else 0
    
    return {
        "overall_accuracy_percent": round(overall_accuracy, 2),
        "confusion_matrix": {
            "true_known_pred_known": true_known_pred_known,
            "true_known_pred_unknown": true_known_pred_unknown,
            "true_unknown_pred_known": true_unknown_pred_known,
            "true_unknown_pred_unknown": true_unknown_pred_unknown
        },
        "known": {
            "precision_percent": round(known_precision, 2),
            "recall_percent": round(known_recall, 2),
            "f1_percent": round(known_f1, 2)
        },
        "unknown": {
            "precision_percent": round(unknown_precision, 2),
            "recall_percent": round(unknown_recall, 2),
            "f1_percent": round(unknown_f1, 2)
        },
        "distribution": {
            "true_known": true_known,
            "true_unknown": true_unknown,
            "pred_known": pred_known,
            "pred_unknown": pred_unknown
        }
    }

correct = 0
incorrect = 0
total = 0
binary_correct = 0
threshold = threshold

# Track all results for JSON output
all_results = []
filtered_results = []  # For binary metrics calculation (matching console output)
class_counts_true = defaultdict(int)

for images, labels in val_loader:
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)

    probabilities = torch.softmax(outputs, dim=1)
    confidences, predicted = torch.max(probabilities, 1)
    top2_values, top2_indices = torch.topk(probabilities, 2)

    for i in range(len(predicted)):
        pred_id = predicted[i].item()
        actual_id = labels[i].item()
        pred_name = class_names[pred_id] if pred_id < len(class_names) else f"Class_{pred_id}"
        actual_name = class_names[actual_id] if actual_id < len(class_names) else f"Class_{actual_id}"
        print(f"Predicted class: {pred_id} ({pred_name}), Confidence: {confidences[i].item()*100:.2f}%")
        print(f"Actual class: {actual_id} ({actual_name})")

    if test_unkown:
        for i in range(len(predicted)):
            logits = outputs[i]
            energy_score = -torch.logsumexp(logits, dim=0)
            margin = top2_values[i][0] - top2_values[i][1]
            total += 1
            true_label = labels[i].item()
            pred_label = predicted[i].item()
            
            # If confidence < threshold OR margin <= threshold OR energy > threshold, treat as unknown
            if confidences[i] < threshold or margin <= top_margin_threshold or energy_score > energy_score_threshold:
                pred_label = unknown_class_id
            
            # Get class names
            pred_name = class_names[pred_label] if pred_label < len(class_names) else f"Class_{pred_label}"
            true_name = class_names[true_label] if true_label < len(class_names) else f"Class_{true_label}"
            confidence_pct = confidences[i].item() * 100
            
            # Track for JSON output
            result_entry = {
                'true_class_id': true_label,
                'pred_class_id': pred_label,
                'true_class_name': true_name,
                'pred_class_name': pred_name,
                'confidence_percent': round(confidence_pct, 2)
            }
            all_results.append(result_entry)
            filtered_results.append(result_entry)  # All samples are included in test_unkown mode
            class_counts_true[true_label] += 1
            
            # Multi-class accuracy
            if pred_label == unknown_class_id:
                correct += 1
            else:
                incorrect += 1
            
            # Binary accuracy (Known vs Unknown) - Only checking unknowns
            # Formula: Correctly predicted Unknown / Total samples
            ##is_true_unknown = (true_label == unknown_class_id)
            is_pred_unknown = (pred_label == unknown_class_id)
            if (is_pred_unknown):
                binary_correct += 1
    else:
        for i in range(len(predicted)):
            logits = outputs[i]
            energy_score = -torch.logsumexp(logits, dim=0)
            margin = top2_values[i][0] - top2_values[i][1]
            true_label = labels[i].item()
            pred_label = predicted[i].item()
            
            # Get class names
            pred_name = class_names[pred_label] if pred_label < len(class_names) else f"Class_{pred_label}"
            true_name = class_names[true_label] if true_label < len(class_names) else f"Class_{true_label}"
            confidence_pct = confidences[i].item() * 100
            
            # Track for JSON output (all samples)
            all_results.append({
                'true_class_id': true_label,
                'pred_class_id': pred_label,
                'true_class_name': true_name,
                'pred_class_name': pred_name,
                'confidence_percent': round(confidence_pct, 2)
            })
            class_counts_true[true_label] += 1
            
            if confidences[i] >= threshold and margin > top_margin_threshold and energy_score <= energy_score_threshold:
                total += 1
                
                # Track filtered results for binary metrics (matching console output)
                filtered_results.append({
                    'true_class_id': true_label,
                    'pred_class_id': pred_label
                })
                
                # Multi-class accuracy
                if pred_label == true_label:
                    correct += 1
                else:
                    incorrect += 1
                
                # Binary accuracy (Known vs Unknown)
                is_true_unknown = (true_label == unknown_class_id)
                is_pred_unknown = (pred_label == unknown_class_id)
                if (not is_true_unknown and not is_pred_unknown) or (is_true_unknown and is_pred_unknown):
                    binary_correct += 1

print("Total: ", total)
print("Correct: ", correct)
print("Incorrect: ", incorrect)
multi_class_accuracy = (100 * correct / total) if total > 0 else 0
print(f'Accuracy of the class based model: {multi_class_accuracy:.2f} %')
if total > 0:
    print(f'Binary accuracy (Known vs Unknown): {100 * binary_correct / total:.2f} %')

# Calculate binary metrics for JSON output (using filtered results to match console output)
binary_metrics = calculate_binary_metrics(filtered_results, unknown_class_id) if unknown_class_id is not None and len(filtered_results) > 0 else None

# Build class counts dictionaries
class_counts_by_id = {str(k): v for k, v in sorted(class_counts_true.items())}
class_counts_by_name = {class_names[k]: v for k, v in sorted(class_counts_true.items()) if k < len(class_names)}

# Determine model type
model_type = "PyTorch"
if model_path and (model_path.endswith('.onnx')):
    model_type = "ONNX"

# Create JSON output
stats = {
    "model_file_information": {
        "generated_at": datetime.now().isoformat(),
        "model_path": model_path if model_path else "",
        "model_type": model_type,
        "dataset_root": img_root_dir if img_root_dir else "",
        "num_classes": len(class_names) if class_names else class_count,
        "class_names": class_names if class_names else []
    },
    "results": {
        "evaluation_metrics": {
            "multi_class_accuracy_percent": round(multi_class_accuracy, 2),
            "multi_class_correct": correct,
            "multi_class_total": total,
            "binary_known_vs_unknown": binary_metrics
        },
        "splits": {
            "test": {
                "num_images": len(all_results),
                "num_labels": len(all_results),
                "class_counts_by_id": class_counts_by_id,
                "class_counts_by_name": class_counts_by_name,
                "label_format_counts": {
                    "classification_single_int": len(all_results),
                    "yolo_five_values": 0
                }
            }
        },
        "details": all_results
    }
}

# Save JSON file in same folder as script
script_dir = os.path.dirname(os.path.abspath(__file__))
if model_path:
    model_filename = os.path.basename(model_path)
    json_filename = f"{model_filename}.json"
else:
    json_filename = "test_results.json"

json_path = os.path.join(script_dir, json_filename)

with open(json_path, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\nResults saved to: {json_path}")