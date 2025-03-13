import numpy as np
import joblib
import os
import tkinter as tk
from tkinter import messagebox

base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))


# Define full paths for model and scaler
model_path = os.path.join(base_path, "xgb_model.pkl")
scaler_path = os.path.join(base_path, "scaler.pkl")

# Check if model and scaler exist before loading
if not os.path.exists(model_path):
    messagebox.showerror("Error", "Model file 'xgb_model.pkl' not found! Place it in the same folder as the .exe.")
    sys.exit()

if not os.path.exists(scaler_path):
    messagebox.showerror("Error", "Scaler file 'scaler.pkl' not found! Place it in the same folder as the .exe.")
    sys.exit()

# Load the trained model and scaler
model = joblib.load('xgb_model.pkl')  # Ensure this file is in the same directory
scaler = joblib.load('scaler.pkl')    # Ensure this file is in the same directory



# Function to handle predictions
def predict():
    try:
        # Retrieve input values from user entries
        tf_val = float(entry_tf.get())
        tw_val = float(entry_tw.get())
        bf_val = float(entry_bf.get())
        hw_val = float(entry_hw.get())
        fyw_val = float(entry_fyw.get())
        fyf_val = float(entry_fyf.get())
        b_val = float(entry_b.get())
        L_val = float(entry_L.get())

        # Prevent division by zero
        if tw_val == 0 or tf_val == 0 or hw_val == 0 or b_val == 0:
            messagebox.showerror("Input Error", "Values cannot be zero.")
            return

        # Compute derived feature ratios
        tf_tw = tf_val / tw_val
        bf_2tf = bf_val / (2 * tf_val)
        hw_tw = hw_val / tw_val
        fyw_fyf = fyw_val / fyf_val
        bf_hw = bf_val / hw_val
        b_d = b_val / hw_val
        L_b = L_val / b_val

        # Create input feature array
        inputs = np.array([[tf_tw, bf_2tf, hw_tw, fyw_fyf, bf_hw, b_d, L_b]])

        # Add a placeholder column to match scaler's expected 8-column shape
        inputs_with_placeholder = np.hstack((inputs, np.zeros((inputs.shape[0], 1))))

        # Scale the inputs using the saved scaler
        inputs_scaled = scaler.transform(inputs_with_placeholder)

        # Remove the placeholder column before passing to the model
        inputs_scaled = inputs_scaled[:, :-1]

        # Predict using the trained model (output is scaled)
        y_pred_scaled = model.predict(inputs_scaled)

        # Create a dummy array to hold the scaled output for inverse transformation
        dummy_array = np.zeros((y_pred_scaled.shape[0], 8))  # 8 columns (7 input + 1 output)
        dummy_array[:, -1] = y_pred_scaled  # Place prediction in last column

        # Inverse transform to get the actual output
        y_pred_unscaled = scaler.inverse_transform(dummy_array)[:, -1]

        # Display the result
        result_label.config(text=f"Prediction: {y_pred_unscaled[0]:.2f} kN")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

# Create main window
root = tk.Tk()
root.title("Patch Loading Resistance Prediction")
root.geometry("400x450")  # Set window size

# Create labels and input fields
tk.Label(root, text="Flange Thickness (tf):").grid(row=0, column=0, sticky="w")
entry_tf = tk.Entry(root)
entry_tf.grid(row=0, column=1)

tk.Label(root, text="Web Thickness (tw):").grid(row=1, column=0, sticky="w")
entry_tw = tk.Entry(root)
entry_tw.grid(row=1, column=1)

tk.Label(root, text="Flange Width (bf):").grid(row=2, column=0, sticky="w")
entry_bf = tk.Entry(root)
entry_bf.grid(row=2, column=1)

tk.Label(root, text="Web Height (hw):").grid(row=3, column=0, sticky="w")
entry_hw = tk.Entry(root)
entry_hw.grid(row=3, column=1)

tk.Label(root, text="Web Yield Strength (fyw):").grid(row=4, column=0, sticky="w")
entry_fyw = tk.Entry(root)
entry_fyw.grid(row=4, column=1)

tk.Label(root, text="Flange Yield Strength (fyf):").grid(row=5, column=0, sticky="w")
entry_fyf = tk.Entry(root)
entry_fyf.grid(row=5, column=1)

tk.Label(root, text="Tension Field Width (b):").grid(row=6, column=0, sticky="w")
entry_b = tk.Entry(root)
entry_b.grid(row=6, column=1)

tk.Label(root, text="Moment Span Length (L):").grid(row=7, column=0, sticky="w")
entry_L = tk.Entry(root)
entry_L.grid(row=7, column=1)

# Create prediction button
predict_button = tk.Button(root, text="Predict", command=predict, bg="green", fg="white")
predict_button.grid(row=8, column=0, columnspan=2, pady=10)

# Create label for displaying the result
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 12, "bold"))
result_label.grid(row=9, column=0, columnspan=2)

# Run the Tkinter event loop
root.mainloop()
