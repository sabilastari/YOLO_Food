import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

# Path to your YOLOv8 model.
# Make sure 'food_detector.pt' is in the same directory as this app.py file,
# or provide the full path to your model.
MODEL_PATH = "food_detector.pt"

# Load the YOLOv8 model
try:
    # Attempt to load the model. Ensure the model file exists.
    model = YOLO(MODEL_PATH)
    print(f"Model successfully loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Please ensure that 'food_detector.pt' is in the same directory as app.py or provide the correct full path.")
    # Exit or handle the error gracefully if the model cannot be loaded
    exit()

# --- Load Calorie Data from CSV ---
try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('dish_calories.csv')
    
    # Clean the data: drop rows where the key or value is missing to prevent errors
    df_cleaned = df.dropna(subset=['English Name of Dish', 'Calories'])
    
    # Create the dictionary using 'English Name of Dish' as key and 'Calories' as value
    CALORIE_DATA = pd.Series(df_cleaned.Calories.values, index=df_cleaned['English Name of Dish']).to_dict()
    print("Calorie data successfully loaded from dish_calories.csv")

except FileNotFoundError:
    print("Error: 'dish_calories.csv' not found. Please ensure the file is in the correct directory.")
    print("Using an empty calorie list as a fallback.")
    CALORIE_DATA = {} # Use an empty dict as a fallback

except KeyError:
    print("Error: Could not find the required columns ('English Name of Dish', 'Calories') in the CSV.")
    print("Using an empty calorie list as a fallback.")
    CALORIE_DATA = {} # Use an empty dict as a fallback

except Exception as e:
    print(f"An error occurred while loading calorie data: {e}")
    CALORIE_DATA = {} # Use an empty dict as a fallback


def predict_food_detection(input_image: Image.Image):
    """
    Function to perform food object detection and estimate total calories.
    Input: Image from Gradio (PIL.Image format)
    Output: Image with bounding boxes, labels, and total calorie estimation.
    """
    img_np = np.array(input_image)
    if img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    results = model(img_np) 
    annotated_img_np = img_np.copy()
    
    total_calories = 0
    detected_foods_info = [] # To store details of detected foods

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            confidence = float(box.conf[0])      
            class_id = int(box.cls[0])           
            class_name = model.names[class_id]   

            # Get calorie estimation for the detected food from the loaded data
            estimated_calorie = CALORIE_DATA.get(class_name, 0) # Get calorie, 0 if not found
            total_calories += estimated_calorie
            
            # Store detected food info
            detected_foods_info.append(f"{class_name} ({estimated_calorie} kalori)")

            color = (0, 255, 0) 
            label = f"{class_name} {confidence:.2f}"
            
            cv2.rectangle(annotated_img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img_np, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Add total calorie information to the image
    text_position = (10, 30) 
    text_color = (255, 255, 255) # White color for text
    bg_color = (0, 0, 0) # Black background for better visibility
    text_font_scale = 1.0
    text_thickness = 2
    
    # Create the total calorie text
    calorie_text = f"Total Estimasi Kalori: {int(total_calories)} kalori"
    
    # Create a background rectangle for the text for better readability
    (w, h), _ = cv2.getTextSize(calorie_text, cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, text_thickness)
    cv2.rectangle(annotated_img_np, (text_position[0] - 5, text_position[1] - h - 5), (text_position[0] + w + 5, text_position[1] + 5), bg_color, -1)

    # Put the total calorie text on the image
    cv2.putText(annotated_img_np, calorie_text, text_position,
                cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, text_color, text_thickness)
    
    # Create the text block with details of detected foods
    detected_details_text = "\n".join(detected_foods_info)
    
    # Convert back to PIL Image
    annotated_img_pil = Image.fromarray(annotated_img_np)

    # Return the annotated image and a text output for detected food details.
    return annotated_img_pil, detected_details_text

# Define the title and description for the Gradio interface.
title = "Deteksi Makanan & Estimasi Kalori dengan YOLOv8 🍎🍕🍔"
description = """
Unggah gambar untuk mendeteksi objek makanan dan mengestimasi total kalorinya.
Model ini akan mengidentifikasi dan memberi label pada berbagai jenis makanan yang terdeteksi,
lalu menjumlahkan estimasi kalori untuk setiap makanan yang ditemukan.
Estimasi kalori diambil dari file 'dish_calories.csv'.
"""

# Define example images for the Gradio interface.
example_images = []
# You can add paths to example images here if you have them.
# e.g., example_images.append("path/to/your/example_image.jpg")

# Create the Gradio interface.
iface = gr.Interface(
    fn=predict_food_detection,
    inputs=gr.Image(type="pil", label="Upload Gambar Makanan"),
    outputs=[
        gr.Image(type="pil", label="Hasil Deteksi Makanan & Kalori"),
        gr.Textbox(label="Detail Makanan Terdeteksi", lines=5)
    ],
    title=title,
    description=description,
    examples=example_images,
    allow_flagging="auto"
)

# Launch the Gradio app.
iface.launch(share=False)
