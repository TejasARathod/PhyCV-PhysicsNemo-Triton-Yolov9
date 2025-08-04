import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("/home/genai/outputs/convert/flow_network.onnx")

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Generate 4 sets of random (x, y) values between 0 and 1
input_tensor = np.random.rand(4, 2).astype(np.float32)  # shape: [4, 2]

# Run inference
outputs = session.run([output_name], {input_name: input_tensor})

# Display inputs and corresponding outputs
for i in range(4):
    x, y = input_tensor[i]
    u, v, p = outputs[0][i]
    print(f"Input {i+1}: x = {x:.4f}, y = {y:.4f} → Output: u = {u:.4f}, v = {v:.4f}, p = {p:.4f}")

