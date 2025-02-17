from dotenv import load_dotenv
import os
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.io import wavfile  # Add this import at the top with other imports

class NeuralNetwork:
    def __init__(self, layers, sample_rate=44100):
        """
        Initialize neural network with layer sizes and audio capabilities
        layers: list of integers representing number of neurons in each layer
        sample_rate: audio sample rate in Hz
        """
        self.layers = layers
        self.sample_rate = sample_rate
        self.weights = []
        self.biases = []
        self.audio_buffer = np.zeros(sample_rate)  # 1 second buffer
        self.buffer_position = 0
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            # He initialization for weights
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0/layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward propagation with audio generation"""
        self.activations = [X]
        self.z_values = []
        
        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.relu(z))
        
        # Output layer with sigmoid
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(self.sigmoid(z))
        
        # Generate audio from activations
        self.update_audio_buffer(self.activations)
        
        return self.activations[-1]
    
    def generate_audio_frame(self, activations):
        """Convert neuron activations to audio samples"""
        # Use middle layer activations for audio
        middle_layer = activations[len(activations)//2]
        
        # Generate base frequencies for each neuron (between 20Hz and 2000Hz)
        frequencies = np.exp(np.linspace(np.log(20), np.log(2000), middle_layer.shape[1]))
        
        # Time array for one frame (100ms)
        frame_size = self.sample_rate // 10
        t = np.linspace(0, 0.1, frame_size)
        
        # Generate audio frame
        audio_frame = np.zeros(frame_size)
        for neuron, freq in zip(middle_layer[0], frequencies):
            amplitude = np.clip(neuron, 0, 1) * 0.1
            audio_frame += amplitude * np.sin(2 * np.pi * freq * t)
        
        return np.clip(audio_frame, -1, 1)

    def update_audio_buffer(self, activations):
        """Update the audio buffer with new samples"""
        audio_frame = self.generate_audio_frame(activations)
        frame_size = len(audio_frame)
        
        end_pos = self.buffer_position + frame_size
        if end_pos > len(self.audio_buffer):
            overflow = end_pos - len(self.audio_buffer)
            self.audio_buffer[self.buffer_position:] = audio_frame[:-overflow]
            self.audio_buffer[:overflow] = audio_frame[-overflow:]
            self.buffer_position = overflow
        else:
            self.audio_buffer[self.buffer_position:end_pos] = audio_frame
            self.buffer_position = end_pos if end_pos < len(self.audio_buffer) else 0

    def backward(self, X, y, learning_rate=0.01):
        """Backward propagation"""
        # Compute the loss derivative with respect to the output
        output_error = self.activations[-1] - y
        deltas = [output_error * self.sigmoid_derivative(self.z_values[-1])]
        
        # Backpropagate the error
        for i in range(len(self.z_values) - 2, -1, -1):
            delta = np.dot(deltas[-1], self.weights[i+1].T) * self.relu_derivative(self.z_values[i])
            deltas.append(delta)
        
        deltas.reverse()
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.activations[i].T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        """Train the neural network and save audio checkpoints"""
        history = []
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
            history.append(loss)
            self.backward(X, y, learning_rate)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                # Save audio checkpoint
                audio_filename = f"neural_audio_epoch_{epoch}.wav"
                audio_data = np.int16(self.audio_buffer * 32767)
                wavfile.write(audio_filename, self.sample_rate, audio_data)
        
        return history

def save_final_audio(model, filename="final_neural_audio.wav"):
    """Save the final state of the audio buffer"""
    audio_data = np.int16(model.audio_buffer * 32767)
    wavfile.write(filename, model.sample_rate, audio_data)
    print(f"Saved final audio to {filename}")

def plot_decision_boundary(model, X, y):
    """Plot decision boundary of the neural network"""
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh points
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Neural Network Decision Boundary')

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Generate a non-linearly separable dataset
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize combined audio buffer for 10 seconds
    combined_audio = np.zeros(44100 * 10)  # 10 seconds at 44.1kHz
    train_accuracies = []
    test_accuracies = []
    
    for iteration in range(10):
        print(f"\nIteration {iteration + 1}/10")
        
        # Create and train the neural network
        nn = NeuralNetwork([2, 16, 8, 1])
        history = nn.train(X_train, y_train.reshape(-1, 1), epochs=1000, learning_rate=0.1, verbose=False)
        
        # Store audio segment from this iteration
        start_idx = iteration * 44100  # Start of 1-second segment
        end_idx = (iteration + 1) * 44100  # End of 1-second segment
        combined_audio[start_idx:end_idx] = nn.audio_buffer
        
        # Make predictions
        train_predictions = nn.forward(X_train)
        train_accuracy = np.mean((train_predictions >= 0.5) == y_train.reshape(-1, 1))
        test_predictions = nn.forward(X_test)
        test_accuracy = np.mean((test_predictions >= 0.5) == y_test.reshape(-1, 1))
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save combined audio file
    audio_data = np.int16(combined_audio * 32767)
    wavfile.write("combined_neural_audio.wav", 44100, audio_data)
    print("\nSaved combined 10-second audio file to combined_neural_audio.wav")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average Training Accuracy: {np.mean(train_accuracies):.4f} ± {np.std(train_accuracies):.4f}")
    print(f"Average Test Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
    
    # Plot final iteration results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History (Final Iteration)')
    
    # Plot decision boundary for final model
    plt.subplot(1, 2, 2)
    plot_decision_boundary(nn, X, y)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()