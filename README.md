# Machine-Learning

Fraud Detection System using Uniform Recognition
This project is a machine learning model developed to identify individuals impersonating law enforcement officers (Police, Army, Customs) in video calls or images by recognizing their uniforms. It uses a two-step computer vision process: first segmenting the clothing from an image, and then classifying the segmented clothing to determine if it is a valid uniform or civilian attire.
This model was trained using the MediaPipe Model Maker library from Google, which simplifies the process of training and deploying custom vision models.
Dataset Used for Training
The model was trained on a custom dataset of images collected from various online sources. The dataset is organized into distinct classes for accurate classification.
You can access and download the full dataset from the following link:
        https://colab.research.google.com/drive/15yf8sthCYCOMIEzN5RYedA-kAJ-SFBps#scrollTo=wZDO5uNnnYTp


System Architecture & Flow
The project follows a standard machine learning pipeline, from data preparation to final inference.
Block Diagram
This diagram illustrates the overall architecture of the system.
[ RAW IMAGE DATASET ] -> [ DATA CLEANING & VALIDATION ] -> [ DATA SPLITTING ] -> [ MODEL TRAINING (MediaPipe Model Maker) ] -> [ EXPORTED .tflite MODEL ] -> [ INFERENCE ]
      |                                                                                                                                              ^
      |                                                                                                                                              |
      +----------------------------------------------------------------------------------------------------------------------------------------------+
                                                                    (New Image for Prediction)

Flowchart of Operations
This flowchart details the logical steps executed by the Python script to train and use the model.
graph TD
    A[Start] --> B{Mount Google Drive};
    B --> C{Load Raw Image Paths};
    C --> D{Clean & Validate Dataset};
    D --> E{Load Clean Data into Model Maker};
    E --> F{Split Data: 80% Train, 10% Test, 10% Validation};
    F --> G{Define Model Specs: MobileNetV2, 30 Epochs};
    G --> H{Train the Image Classifier Model};
    H --> I{Evaluate Model on Test Data};
    I --> J{Export Trained Model as model.tflite};
    J --> K[Load Exported Model for Inference];
    K --> L{Input New Image};
    L --> M{Get Classification Prediction};
    M --> N[End];

Sample Outputs
Model Training Progress
During the model.create() step, the console displays the progress for each training epoch, showing the loss and accuracy.
Epoch 1/30
10/10 [==============================] - 15s 500ms/step - loss: 1.3012 - accuracy: 0.4500 - val_loss: 1.0123 - val_accuracy: 0.6000
Epoch 2/30
10/10 [==============================] - 5s 480ms/step - loss: 0.8521 - accuracy: 0.7500 - val_loss: 0.7890 - val_accuracy: 0.8000
...
Epoch 30/30
10/10 [==============================] - 5s 475ms/step - loss: 0.1025 - accuracy: 0.9850 - val_loss: 0.2345 - val_accuracy: 0.9400

Final Model Evaluation
After training, the model is evaluated on the unseen test dataset to determine its real-world accuracy.
Test Accuracy: 0.95
Test loss:0.2154, Test accuracy:0.9523

Sample Inference Result
When a new image is provided to the trained model, it outputs the predicted class along with a confidence score. This example shows a test on an image of a police officer.
Label: police_uniform, Confidence: 0.97
Label: civilian_clothes, Confidence: 0.02
Label: army_uniform, Confidence: 0.01
