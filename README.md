# medical_diagnosis

🌟 Project Overview
    Medical Diagnosis with Neural Networks is a simple and lightweight Python-based application that demonstrates how a neural network can be used to predict medical conditions such as "Healthy" or "Sick" based on basic patient data inputs like fever and fatigue levels. This project is designed as an educational tool to help users understand how neural networks function and how they can be built and trained from scratch without relying on advanced machine learning libraries.

   This system processes simulated health-related data to train the model and predict whether a patient is "Healthy" or "Sick" based on certain input features. The final diagnosis prediction is visualized through a bar chart using Python's built-in Turtle graphics module.

🛠️ Technologies Used
   Python: The programming language used to write the neural network and handle data.

   Turtle Graphics: Python's built-in module used to visualize the prediction results as a bar chart, showing the classification outcome (Healthy vs. Sick).

💡 How the System Works

Data Preparation: The system uses simulated health data (e.g., fever, fatigue) and scales the values between 0 and 1 for neural network input.

Neural Network Design:

        Input Layer: Accepts features like fever and fatigue.

        Hidden Layer(s): Learns patterns from input data.

        Output Layer: Classifies as "Healthy" (0) or "Sick" (1).

Training: The network is trained with backpropagation, adjusting weights over multiple epochs using the Sigmoid activation function to minimize error.

Prediction: After training, the model predicts a diagnosis (Healthy or Sick) based on new data.

Visualization: The prediction is visualized using Turtle graphics, displaying a bar chart for "Healthy" vs. "Sick" probabilities.

