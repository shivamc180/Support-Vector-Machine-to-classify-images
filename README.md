# Cat vs Dog Image Classifier

This project implements a Support Vector Machine (SVM) model to classify images of cats and dogs. The model is trained on a subset of the Kaggle dataset "Dogs vs. Cats" due to hardware limitations, and a simple graphical user interface (GUI) is built using the Gradio library to allow for easy testing of the model.

## Dataset

The dataset used for this project is the [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) dataset from Kaggle. The full dataset contains 25,000 images of cats and dogs. Due to performance constraints, we used a subset of the dataset:

- **Total Images Used**: 4000 (2000 images of cats and 2000 images of dogs)
- **Image Resolution**: Images were resized to 256x256 pixels.
- **Training and Testing Split**: The dataset was split into training and testing sets, with 80% of the data used for training and 20% for testing.

## Preprocessing

The images were preprocessed as follows:

1. **Resizing**: Each image was resized to 256x256 pixels.
2. **Grayscale Conversion**: The images were converted to grayscale to reduce computational complexity.
3. **Flattening**: The images were flattened into a 1D array to be used as input for the SVM model.
4. **Standardization**: The pixel values were standardized to have a mean of 0 and a standard deviation of 1.

## Model

A Support Vector Machine (SVM) with a linear kernel was chosen for its simplicity and effectiveness in binary classification tasks. 

### Training

- **Kernel**: rbf
- **Regularization Parameter (C)**: 1.0
- **Training Time**: Due to the simplicity of the model and the reduced dataset size, the training time was manageable, even on lower-performance hardware.

### Evaluation

The model was evaluated using the following metrics:

Accuracy: 63%
Confusion Matrix:
 [[233 189]
 [106 273]]
Classification Report:
               precision    recall  f1-score   support

           0       0.69      0.55      0.61       422
           1       0.59      0.72      0.65       379

    accuracy                           0.63       801
   macro avg       0.64      0.64      0.63       801
weighted avg       0.64      0.63      0.63       801
- **Precision, Recall, and F1-Score**: These metrics were calculated for both classes (cats and dogs) to give a more detailed view of the model's performance.

## Gradio Interface

To make the model easy to use and test, a simple graphical user interface (GUI) was created using the [Gradio](https://gradio.app/) library. The interface allows users to upload an image of a cat or dog, and the model will predict the class (cat or dog) of the image.

## Dependencies

To run the project, you need the following Python packages:

   - numpy
   - scikit-learn
   - opencv-python
   - gradio

You can install the required packages using the following command:
```bash
pip install numpy scikit-learn opencv-python gradio
```


## Challenges and Considerations

   -  Hardware Limitations: Due to low-performance hardware, we had to reduce the dataset size to 2000 images. This may have impacted the model's accuracy.
   - Image Resolution: The images were resized to 256x256 pixels, which is a compromise between computational efficiency and retaining enough detail for classification.
   - SVM Limitations: While SVM is a solid choice for binary classification, it may not capture the complex patterns in image data as effectively as more advanced models like Convolutional Neural Networks (CNNs).

## Future Work

   - Increase Dataset Size: Train the model on a larger subset of the dataset if hardware constraints are resolved.
   - Explore Non-Linear Kernels: Use non-linear kernels like RBF to capture more complex patterns in the data.
   - Use a CNN Model: Implement a Convolutional Neural Network (CNN) for potentially better performance in image classification tasks.

## Conclusion

This project demonstrates the use of a Support Vector Machine (SVM) to classify images of cats and dogs, with a simple and intuitive interface built using Gradio. Despite the hardware limitations, the project provides a solid foundation for further exploration in image classification using machine learning.
