# Anti-Spoofing System

## Project Description
This project aims to develop an algorithm to detect whether a face in a video is real or spoofed. The system uses a combination of feature extraction techniques and a deep learning model to achieve this goal.

## Datasets Used
- **[CASIA FASD Dataset](https://www.kaggle.com/datasets/minhnh2107/casiafasd)**: Consists of images with depth maps, containing both real and spoofed faces.
- **[LLC and CASIA Combined Dataset](https://www.kaggle.com/datasets/ahmedruhshan/lcc-fasd-casia-combined)**: A combination of two datasets, used for training and testing.
- **[iBeta Level 1 Liveness Detection Dataset](https://www.kaggle.com/datasets/trainingdatapro/ibeta-level-1-liveness-detection-dataset-part-1)**: A video dataset for testing, including short clips categorized into real and spoof.

## Libraries and Tools
- **Python**: `scipy`, `tensorflow`, `opencv`, `numpy`, `pandas`, `matplotlib`
- **Deep Learning Architecture**: MobileNet used for feature extraction and classification tasks.

## Feature Engineering Techniques
- **Haar Cascade**: Used for face detection.
- **Local Binary Patterns (LBP)**: Used for texture analysis.
- **Gabor Filters**: Used to extract spatial frequency features.
- **Histogram of Oriented Gradients (HOG)**: Used for object detection.
- **Difference of Gaussians (DOG)**: Used for edge detection.
- **SIFT (Scale-Invariant Feature Transform)**: Used for keypoint detection and matching.
- **FFT (Fast Fourier Transform)**: Used for analyzing frequency patterns.

## Workflow
Two workflows are used. One of which is accomplished with the use of a combination of **Feature Extraction** and **SVM**, and another one is done using **Deep Learning** on its own. It's worth nothing that the model used for deep learning part is MobileNetV2.

### 1. Feature Extraction
- **Face Detection**: The face is detected in the input video or image using the Haar Cascade classifier.
- **Texture Analysis**: Techniques like LBP and Gabor filters are applied to analyze the texture of the detected face.
- **Spatial and Frequency Analysis**: HOG, DOG, SIFT, and FFT are used to extract spatial and frequency features from the image, capturing essential details that differentiate real from spoofed faces.
- **SVM**: Finally all the extracted data are fed to SVM model after a space contraction by applying PCA on extracted features.
- **Training and Testing**: The model is trained on the combined dataset, and its performance is tested using the iBeta Level 1 Liveness Detection Dataset.
- **Classification**: Trained model outputs a prediction, indicating whether the face is real or spoofed.

### 2. Deep Learning Model
- **Model Architecture**: Images are fed into a MobileNet model, which is used for classification.
- **Training and Testing**: The model is trained on the combined dataset, and its performance is tested using the iBeta Level 1 Liveness Detection Dataset.
- **Classification**: The MobileNet model outputs a prediction, indicating whether the face is real or spoofed.

## Document and Files
To view document click [here](https://github.com/KamyarMoradian/Anti-Spoofing-System/blob/main/Final_Project.pdf). *This ducment is in Persian.*

In addition to all have been said, there is one additional file that is noteworthy. In this file we have gathered the information on checking various combinations of features introduced above to learn which one gives better results. You can view the results in this [file](https://github.com/KamyarMoradian/Anti-Spoofing-System/blob/main/predictions_features.csv).

## References
- [CASIA FASD Dataset](https://www.kaggle.com/datasets/minhnh2107/casiafasd)
- [LLC and CASIA Combined Dataset](https://www.kaggle.com/datasets/ahmedruhshan/lcc-fasd-casia-combined)
- [iBeta Level 1 Liveness Detection Dataset](https://www.kaggle.com/datasets/trainingdatapro/ibeta-level-1-liveness-detection-dataset-part-1)
- [OpenCV Cascade Classifier Documentation](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
- [Gabor Filter Explanation](https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac97)
