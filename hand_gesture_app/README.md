# Hand Gesture Recognition

A real-time **Hand Gesture Recognition** system built with **Python, PyTorch, OpenCV, and MediaPipe**. The project detects hand gestures through a webcam and uses a trained deep learning model to classify them into **97 gesture classes**.

The project is designed as a foundation for real-time sign/gesture recognition and can be extended toward sentence-level sign language recognition.

---

## Features

* Real-time webcam-based hand gesture detection
*  Hand landmark-based feature extraction
*  PyTorch deep learning model
*  Recognition of **97 gesture classes**
*  Real-time prediction using OpenCV
* Dataset class-distribution visualization
* Training pipeline included
*  Sample image and prediction output included
*  Modular training and inference scripts



##  Technologies Used

| Technology | Purpose                          

Python     | Programming language             
PyTorch    | Deep learning model and training 
OpenCV     | Webcam and image processing      
MediaPipe  | Hand landmark detection          
NumPy      | Numerical operations             
Matplotlib | Data visualization               

---

##  Project Structure


Hand_gesture/
│
├── livepredict.py             # Real-time webcam prediction
├── model.py                   # Model architecture
├── train.py                   # Model training script
├── class_names.txt            # Gesture class names
├── sentence_output.txt        # Example prediction output
├── sample_image.png           # Sample project image
├── class_distribution.png     # Dataset class distribution
├── .gitignore                 # Ignored files and folders
│
└── dataset/                   # Local dataset (not included in GitHub)


> **Note:** The complete dataset is approximately **1.6 GB** and is therefore not included in this repository.

---

##  Dataset

The project uses a hand gesture dataset containing multiple gesture classes.

The current model supports:

**97 gesture classes**

The dataset is used to train the gesture classification model.

Because the dataset is approximately **1.6 GB**, it is kept outside the GitHub repository.

Place the downloaded dataset inside the project directory using:


dataset/


The expected structure may vary depending on the dataset used for training.

---

## ⚙️ Installation

### 1. Clone the repository


git clone https://github.com/dikshitbhusal123/Hand_gesture.git


Move into the project:


cd Hand_gesture


### 2. Create a virtual environment



python3 -m venv venv





source venv/bin/activate


### 3. Install dependencies

Install the required packages:


pip install torch torchvision opencv-python mediapipe numpy matplotlib


---

##  Training the Model

Make sure the dataset is available in:


dataset/


Then run:


python train.py


The training script processes the dataset and trains the PyTorch gesture classification model.

The trained model can then be used by the real-time prediction script.

---

## Real-Time Gesture Recognition

After installing the dependencies and preparing the model, run:


python livepredict.py


The application will access your webcam and perform real-time gesture recognition.

### Pipeline


Webcam
   │
   ▼
Hand Detection
   │
   ▼
Hand Landmarks
   │
   ▼
Feature Extraction
   │
   ▼
PyTorch Model
   │
   ▼
Gesture Classification
   │
   ▼
Predicted Gesture


---

## Model Workflow

The recognition system follows these main steps:

### 1. Hand Detection

The webcam captures frames using OpenCV.

### 2. Landmark Extraction

Hand landmarks are detected and converted into numerical features.

A hand contains:


21 landmarks


Each landmark contains:


X coordinate
Y coordinate
Z coordinate


Therefore, the basic feature representation contains:


21 × 3 = 63 features


### 3. Model Prediction

The extracted features are passed to the trained PyTorch model.

The model produces predictions across:


97 gesture classes


### 4. Output

The predicted gesture is displayed in real time.

---

##  Dataset Visualization

The repository contains:


class_distribution.png


which provides a visualization of the distribution of gesture classes in the dataset.

This can be used to inspect class balance and identify classes with significantly different numbers of samples.

---

##  Future Improvements

Possible future extensions include:

* [ ] Sentence-level sign language recognition
* [ ] Continuous gesture recognition
* [ ] Two-hand gesture recognition
* [ ] Improved model accuracy
* [ ] Gesture sequence modeling using LSTM/GRU
* [ ] Transformer-based temporal recognition
* [ ] Text-to-sign language conversion
* [ ] 3D avatar-based sign visualization
* [ ] Mobile/web deployment
* [ ] Model optimization for real-time inference

---

##  Experiments

Future experiments can include dimensionality reduction and feature analysis techniques such as:

* PCA
* SVD
* Feature normalization
* Data augmentation
* Hyperparameter optimization

These techniques can be evaluated by comparing model accuracy, inference speed, and computational requirements.

---

##  Dataset Notice

The dataset is intentionally **not included in this repository** because of its approximately **1.6 GB size**.

Before using this project, download or obtain the appropriate dataset and place it inside:


dataset/


Please also verify the dataset's license and redistribution requirements before sharing or redistributing it.

---

##  Author

**Dikshit Bhusal**

Machine Learning Developer / ML Enthusiast

---

##  License

This project can be licensed under the terms specified in the repository.

If you plan to make the project open source, consider adding an appropriate license such as the MIT License.

---

##  Acknowledgments

This project uses open-source machine learning and computer vision technologies including:

* PyTorch
* OpenCV
* MediaPipe
* NumPy
* Matplotlib

---

##  Contributing

Contributions, improvements, and suggestions are welcome.

To contribute:

```bash
git clone https://github.com/dikshitbhusal123/Hand_gesture.git
cd Hand_gesture
```

Create a new branch:


git checkout -b feature/new-feature




