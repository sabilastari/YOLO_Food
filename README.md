# YOLO Food Detection

This project is a final assignment for a university course that focuses on detecting various types of food from images using the YOLO (You Only Look Once) object detection model. The main goal is to build, train, and deploy a model capable of accurately identifying and localizing food items in a given image.

## Project Rationale

The ability for a machine to recognize and identify food items has a wide range of applications, from calorie tracking and dietary management to automated checkout systems in restaurants and cafeterias. Traditional image classification models can identify the presence of food, but they cannot localize multiple items within a single image. Object detection models like YOLO are better suited for this task as they can identify multiple objects and provide bounding boxes for each. This project was undertaken to explore the practical application of the YOLO algorithm in the context of food detection, leveraging its real-time processing capabilities.

## Process

The project followed a structured machine learning workflow:

1.  **Dataset Collection & Preparation**: A custom dataset of food images was gathered. The images were then annotated using bounding boxes to label the specific food items in each image. The dataset was split into training, validation, and testing sets.
2.  **Model Selection**: The YOLOv8 model was chosen for its high accuracy and performance. YOLO is a state-of-the-art, real-time object detection system.
3.  **Model Training**: The YOLOv8 model was trained on the custom food dataset. The training process involved fine-tuning the model on the specific classes of food items.
4.  **Model Evaluation**: The trained model was evaluated on the validation set to measure its performance. The key metric used for evaluation was mean Average Precision (mAP).
5.  **Deployment**: The final trained model was deployed as an interactive web application using Hugging Face Spaces, allowing users to upload an image and see the food detection results in real-time.

## Dataset

The dataset used for this project is a custom collection of images containing various food items. The data was sourced from public image repositories and personal collections. Each image was manually annotated with bounding boxes corresponding to the food items present.


## Results and Validation

The model's performance was evaluated using the validation dataset. It achieved a satisfactory level of accuracy in detecting and classifying different food items.

-   **mAP50-95:** 96.4%
-   **Precision:** 93.4%
-   **Recall:** 92.7%

## Deployment

The trained model is deployed and publicly accessible via Hugging Face Spaces. You can try it out here:

-   **Hugging Face Space:** [YOLO_Food](https://huggingface.co/spaces/bil4jay/YOLO_Food)
