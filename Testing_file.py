import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from difflib import get_close_matches

# Load the TFLite model
model_path = 'google_foods.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the labels from the CSV file
file_path = 'aiy_food_V1_labelmap.csv'
labels_df = pd.read_csv(file_path, header=0)

# Extract the labels into a list, skipping the header row
model_labels = labels_df.iloc[:, 1].tolist()

# Ensure labels are correctly processed
model_labels = [label.strip().lower() for label in model_labels]

# Function to preprocess images
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(192, 192))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.uint8)  # Ensure the image is of type UINT8
    return img

# Use the current working directory as the base folder path
base_folder_path = os.getcwd()

# Create a mapping from labels to indices
label_to_int = {label: index for index, label in enumerate(model_labels)}
int_to_label = {index: label for index, label in enumerate(model_labels)}

# Function to find the closest match for a label
def find_closest_label(label, model_labels):
    closest_matches = get_close_matches(label, model_labels, n=1, cutoff=0.0)
    return closest_matches[0] if closest_matches else None

# Initialize statistics tracking
class_stats = {label: {'total': 0, 'correct': 0, 'confidence_sum': 0.0} for label in model_labels}

# Iterate through each folder in the base directory
for folder_name in os.listdir(base_folder_path):
    folder_path = os.path.join(base_folder_path, folder_name)
    if os.path.isdir(folder_path):
        true_labels = []
        image_paths = []
        
        for filename in os.listdir(folder_path):
            if filename.endswith((".png", ".jpeg", ".jpg")):
                image_paths.append(os.path.join(folder_path, filename))
                true_labels.append(folder_name.replace('_', ' ').lower())
        
        # Find the closest match for the folder name in model labels
        matched_label = find_closest_label(folder_name.replace('_', ' ').lower(), model_labels)
        if matched_label is None:
            print(f"Warning: No close match found for folder '{folder_name}' in model labels.")
            continue
        
        print(f"Testing images in folder: {folder_name}, Matched label: {matched_label}")
        
        matched_label_index = label_to_int[matched_label]
        
        # Run inference and collect predictions
        predictions = []
        recognized_labels = []
        for image_path in image_paths:
            img = preprocess_image(image_path)
            
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_label_index = np.argmax(output_data)
            confidence = np.max(output_data)
            
            recognized_label = int_to_label.get(predicted_label_index, 'unknown')
            recognized_labels.append(recognized_label)
            
            if recognized_label == matched_label:
                class_stats[matched_label]['correct'] += 1
                class_stats[matched_label]['confidence_sum'] += confidence
            
            class_stats[matched_label]['total'] += 1
            predictions.append(predicted_label_index)

        # Convert true labels to the matched label index
        int_true_labels = [matched_label_index] * len(true_labels)
        
        # Calculate accuracy for this class
        if len(int_true_labels) > 0:
            accuracy = accuracy_score(int_true_labels, predictions)
            print(f'Accuracy for class "{matched_label}": {accuracy:.2f}')
        
        # Print the recognized objects for manual verification
        for image_path, recognized_label in zip(image_paths, recognized_labels):
            print(f"Image: {os.path.basename(image_path)}, Recognized as: {recognized_label}")

# Calculate and print overall statistics
for label, stats in class_stats.items():
    if stats['total'] > 0:
        success_rate = stats['correct'] / stats['total'] * 100
        avg_confidence = (stats['confidence_sum'] / stats['correct']) * 100 if stats['correct'] > 0 else 0.0
        print(f'Class: {label}, Total: {stats["total"]}, Correct: {stats["correct"]}, Success Rate: {success_rate:.2f}%, Average Confidence: {avg_confidence:.2f}%')
