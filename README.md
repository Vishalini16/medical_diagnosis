# medical_diagnosis

# 🧠 Project overview

This project demonstrates the use of a **neural network** to predict medical conditions (e.g., "Healthy" or "Sick") based on patient data like fever and fatigue. The neural network is built from scratch in Python, without relying on advanced machine learning libraries. The system processes simulated health-related data, trains a neural network, and makes predictions with visualizations displayed via **Turtle graphics**.

This project is aimed at providing a simple, educational tool for understanding the basic principles of neural networks and medical diagnosis.

---


## 🛠️ Technologies Used

- **Python**: The main programming language used to implement the neural network.
- **Turtle Graphics**: For visualizing predictions in a bar chart format.
---

## 💡 How the System Works

1. **Data Preparation**: The system uses simulated health data (e.g., fever, fatigue) and scales the values between 0 and 1 to fit the neural network’s input format.
   
2. **Neural Network Design**:
   - **Input Layer**: Accepts features like fever and fatigue.
   - **Hidden Layer(s)**: Learns patterns from the input data.
   - **Output Layer**: Classifies the result as "Healthy" (0) or "Sick" (1).

3. **Training**: The network is trained using backpropagation, adjusting weights over multiple epochs with the **Sigmoid activation function** to minimize errors.

4. **Prediction**: After training, the model can predict if a new patient is Healthy or Sick based on their data (e.g., fever level, fatigue).

5. **Visualization**: Predictions are displayed using **Turtle graphics** as a bar chart, showing the probabilities for "Healthy" and "Sick".

---
