
# Vitamin Deficiency Detection using Deep Learning (AlexNet)

## 1\. Project Overview

This project presents a deep learning-based system designed to detect and classify vitamin deficiencies from microscopy tissue images. By leveraging the power of a custom-built AlexNet Convolutional Neural Network (CNN) and a complementary clustering algorithm, the system provides an efficient and more accurate alternative to traditional, time-consuming manual diagnoses. The solution is integrated into a user-friendly web interface for seamless image upload and instant results.

## 2\. Motivation

Vitamin deficiencies can lead to severe health complications if left undiagnosed. Traditional methods for identifying these deficiencies from tissue samples are often slow, require specialized expertise, and are susceptible to human error. This project aims to automate and enhance this process, providing a robust, data-driven tool to assist medical professionals in rapid and reliable diagnosis.

## 3\. Methodology

### 3.1. Data Acquisition & Preprocessing

The system utilizes microscopy tissue images captured in RGB format. These images undergo a series of preprocessing steps to prepare them for the neural network:

  * **Resizing**: Images are resized to a uniform `227x227` pixels, which is the required input size for the AlexNet architecture.
  * **Cropping & Segmentation**: The focus is narrowed to the region of interest, and background noise is removed to ensure the model learns from relevant features.
  * **Normalization**: Pixel values are normalized to improve contrast and stabilize model training.

### 3.2. Model Architecture

The core of the system is a custom Convolutional Neural Network based on the **AlexNet** architecture. It is a multiclass classification model designed to identify one of several possible deficiency types (Vitamin A, B, C, D, or E). The network is composed of:

  * **Five Convolutional Layers**: These layers are responsible for hierarchical feature extraction, capturing intricate details like edges, textures, color intensities, and abnormal tissue structures.
  * **ReLU Activation**: The rectified linear unit (ReLU) is used as the activation function to introduce non-linearity and accelerate the training process.
  * **Max-Pooling Layers**: These layers reduce the dimensionality of the feature maps, making the model more computationally efficient and robust to variations in image position.
  * **Three Fully Connected Layers**: These layers interpret the features extracted by the convolutional layers and map them to the final classification output.
  * **Softmax Layer**: The final layer assigns a probability to each deficiency category, with the highest probability class being selected as the final prediction.

### 3.3. Clustering for Prediction Enhancement

A unique feature of this system is the clustering step that precedes the final prediction. Before the image is classified, a fuzzy clustering algorithm is applied to the image features. This step groups the uploaded image with other visually similar images in the dataset, effectively placing it into the correct visual context. The model then performs its prediction within this targeted group, improving the overall accuracy and reliability of the diagnosis.

## 4\. Web Interface & Key Features

The system is deployed via a user-friendly web-based interface that simplifies the diagnostic workflow:

  * **Image Upload**: Users can directly upload a microscopy tissue image.
  * **Automated Clustering & Prediction**: The system automatically applies the clustering and prediction algorithms upon image upload.
  * **Detailed Results**: The web interface displays the predicted vitamin deficiency along with relevant information on the deficiency's symptoms and potential treatments, providing immediate value to the user.
<img width="1512" height="873" alt="Screenshot 2025-03-10 221327" src="https://github.com/user-attachments/assets/bb2898b3-5bab-4219-bfdb-8c2a5fa65546" />
<img width="1529" height="802" alt="Screenshot 2025-03-10 221434" src="https://github.com/user-attachments/assets/90c489c1-4c09-43d3-9d00-a66c41cfaf74" />
<img width="1531" height="821" alt="Screenshot 2025-03-10 221519" src="https://github.com/user-attachments/assets/eff77f74-cead-4e2f-8aba-820517fc7632" />
<img width="1509" height="767" alt="Screenshot 2025-03-10 221543" src="https://github.com/user-attachments/assets/00d6fc1c-724c-4097-8de1-44a623b2f41d" />
<img width="1270" height="669" alt="Screenshot 2025-03-10 221633" src="https://github.com/user-attachments/assets/447fef0b-54bb-43f0-97df-e5769f901c61" />


## 5\. Performance Evaluation

The model's performance was rigorously evaluated using standard metrics for multiclass classification problems. The system achieved a high level of accuracy, outperforming traditional machine learning methods.

  * **Accuracy**: 85-95%
  * **Metrics**: Accuracy, Precision, Recall, and F1 Score.
    The high performance, particularly on low-contrast medical images, validates the effectiveness of the deep learning approach.

## 6\. Future Improvements

This project lays a strong foundation for future development, including:

  * **Transfer Learning**: Exploring the use of pre-trained models like ResNet or VGG16 to further improve model performance and training efficiency.
  * **Mobile Integration**: Adapting the system into a mobile application for real-time, on-site detection, making the diagnostic tool more accessible and convenient.

## 7\. Project Structure

The project code is organized as follows:

```
/
├── app.py                      # Main Flask application
├── static/
│   ├── img/                    # Directory for processed images
│   └── images/                 # Directory for original/clustered images
├── templates/
│   ├── first.html              # Landing page
│   ├── index.html              # Main application page for prediction
│   ├── index1.html             # Page for fuzzy clustering
│   ├── login.html              # Login page
│   ├── success.html            # Success page after clustering
│   └── chart.html              # Chart page (if applicable)
├── label_image.py              # TensorFlow model classification logic
├── image_fuzzy_clustering.py   # Fuzzy C-Means clustering logic
├── retrained_graph.pb          # Pre-trained TensorFlow model
└── retrained_labels.txt        # Labels for the model output
```

## 8\. Requirements

The project requires the following libraries to be installed:

  * Flask
  * TensorFlow
  * NumPy
  * SciPy
  * Pillow (PIL)
  * OpenCV
  * Matplotlib
