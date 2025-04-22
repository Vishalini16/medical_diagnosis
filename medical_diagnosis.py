import random
import math
import turtle
import time

# --- Activation functions ---
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# --- Initialize weights ---
def initialize_weights():
    w_input_hidden = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
    w_hidden_output = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
    return w_input_hidden, w_hidden_output

# --- Forward propagation ---
def forward(inputs, w1, w2):
    hidden = []
    for i in range(2):
        h = sigmoid(sum(inputs[j] * w1[i][j] for j in range(2)))
        hidden.append(h)
    
    outputs = []
    for i in range(2):
        o = sigmoid(sum(hidden[j] * w2[i][j] for j in range(2)))
        outputs.append(o)
    return hidden, outputs

# --- Backpropagation training ---
def train(X, y, epochs, lr):
    w1, w2 = initialize_weights()
    for epoch in range(epochs):
        for i in range(len(X)):
            inputs = X[i]
            targets = y[i]

            hidden, outputs = forward(inputs, w1, w2)

            # Calculate output layer error and update weights
            output_deltas = []
            for k in range(2):
                error = targets[k] - outputs[k]
                delta = error * sigmoid_derivative(outputs[k])
                output_deltas.append(delta)

            # Update hidden-output weights
            for i in range(2):
                for j in range(2):
                    w2[i][j] += lr * output_deltas[i] * hidden[j]

            # Calculate hidden layer error and update weights
            hidden_deltas = []
            for i in range(2):
                error = sum(output_deltas[k] * w2[k][i] for k in range(2))
                delta = error * sigmoid_derivative(hidden[i])
                hidden_deltas.append(delta)

            for i in range(2):
                for j in range(2):
                    w1[i][j] += lr * hidden_deltas[i] * inputs[j]
    return w1, w2

# --- Predict output from input ---
def predict(inputs, w1, w2):
    _, outputs = forward(inputs, w1, w2)
    return outputs

# --- Visual output using Turtle as a bar graph ---
def draw_bar_graph(values, labels):
    screen = turtle.Screen()
    screen.title("Medical Diagnosis Prediction (Bar Graph)")
    screen.bgcolor("white")
    t = turtle.Turtle()
    t.speed(0)
    t.hideturtle()
    t.penup()

    max_height = max(values)
    bar_width = 100
    spacing = 50
    start_x = -150

    for i, value in enumerate(values):
        bar_height = value * 200  # scale
        t.goto(start_x + i * (bar_width + spacing), -100)
        t.pendown()
        t.begin_fill()
        t.fillcolor("blue" if i == 0 else "red")
        for _ in range(2):
            t.forward(bar_width)
            t.left(90)
            t.forward(bar_height)
            t.left(90)
        t.end_fill()
        t.penup()
        t.goto(start_x + i * (bar_width + spacing) + bar_width / 2, bar_height - 90)
        t.write(f"{labels[i]}\n{values[i]:.2f}", align="center", font=("Arial", 12, "bold"))

    time.sleep(6)
    screen.bye()

# --- Sample patient data: [fever, fatigue] ---
X = [
    [0.1, 0.9],  # Sick
    [0.9, 0.1],  # Healthy
    [0.2, 0.8],  # Sick
    [0.85, 0.2], # Healthy
]

# Targets as one-hot encoding: [Healthy, Sick]
y = [
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0]
]

# --- Train the network ---
weights1, weights2 = train(X, y, epochs=1000, lr=0.5)

# --- Predict a new patient ---
new_patient = [0.2, 0.85]  # High fever & fatigue → likely Sick
prediction = predict(new_patient, weights1, weights2)

print(f"Predicted Probabilities:\nHealthy: {prediction[0]:.2f}, Sick: {prediction[1]:.2f}")
predicted_label = "Sick" if prediction[1] > prediction[0] else "Healthy"
print("Predicted Diagnosis:", predicted_label)

# --- Draw the prediction as a bar graph ---
draw_bar_graph(prediction, ["Healthy", "Sick"])
