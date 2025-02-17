import numpy as np
from scipy.io import wavfile
import pygame
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
import time

class AuditoryNeuralNetwork:
    """A neural network that teaches through sound"""
    
    def __init__(self, layer_sizes, sample_rate=44100):
        self.layers = layer_sizes
        self.sample_rate = sample_rate
        self.weights = []
        self.biases = []
        self.activations = []  # Store activations for backprop
        self.z_values = []     # Store z values for backprop
        
        pygame.mixer.init(frequency=sample_rate)
        self.frequency_maps = self._initialize_frequency_maps()
        
        # Initialize weights - we'll sonify this process
        print("\nListening to weight initialization...")
        for i in range(len(layer_sizes) - 1):
            # He initialization with sound
            weights = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/layer_sizes[i])
            self.weights.append(weights)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
            
            # Play sound representing weight distribution
            self._sonify_weights(weights, f"Layer {i} → {i+1}")
            time.sleep(1)

    def _initialize_frequency_maps(self):
        """Initialize frequency mappings for different network aspects"""
        return {
            'weight_min': 220,    # A3
            'weight_max': 880,    # A5
            'error_min': 220,     # A3
            'error_max': 880,     # A5
            'gradient_min': 330,  # E4
            'gradient_max': 660   # E5
        }

    def _sonify_weights(self, weights, description):
        """Convert weight matrix to sound - higher pitch = more positive weight"""
        print(f"\nWeight pattern: {description}")
        
        # Create a 1-second sound showing weight distribution
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t)
        
        # Map weights to frequencies: negative=low, zero=mid, positive=high
        min_freq, max_freq = 220, 880  # A3 to A5
        
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                weight = weights[i,j]
                # Map weight to frequency
                freq = np.interp(weight, [-1, 1], [min_freq, max_freq])
                amplitude = abs(weight) * 0.1
                audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        audio = np.clip(audio, -1, 1)
        self._play_audio(audio)
        
        # Explain what we're hearing
        positive_weights = np.sum(weights > 0.5)
        negative_weights = np.sum(weights < -0.5)
        print(f"Hearing {positive_weights} strong positive connections (high pitch)")
        print(f"and {negative_weights} strong negative connections (low pitch)")

    def learn_single_pattern(self, x, y, learning_rate=0.1, epochs=5):
        """Learn a single input-output pattern with audio feedback"""
        print("\nLearning a single pattern...")
        print("Listen to how the network adapts:")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}:")
            
            # Forward pass with audio
            prediction = self.forward_with_audio(x)
            error = y - prediction
            
            # Error sonification
            self._sonify_error(error)
            
            # Backward pass with audio feedback
            self._backward_with_audio(x, error, learning_rate)
            
            print(f"Target: {y[0,0]:.2f}, Prediction: {prediction[0,0]:.2f}")
            time.sleep(1)

    def _sonify_error(self, error):
        """Convert prediction error to sound"""
        duration = 0.5
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Error magnitude determines volume
        amplitude = min(abs(error[0,0]), 1) * 0.2
        
        # Error sign determines frequency
        freq = 440 * (2 if error[0,0] > 0 else 0.5)
        
        audio = amplitude * np.sin(2 * np.pi * freq * t)
        self._play_audio(audio)
        
        # Explain what we're hearing
        print(f"Error tone: {'high' if error[0,0] > 0 else 'low'} pitch = {'under' if error[0,0] > 0 else 'over'}shooting")

    def _backward_with_audio(self, x, error, learning_rate):
        """Backward propagation with audio feedback"""
        # Compute gradients with sound
        delta = error
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient computation sound
            grad_w = np.dot(self.activations[i].T, delta)
            
            # Sonify gradient updates
            self._sonify_gradient_update(grad_w, i)
            
            # Update weights
            self.weights[i] += learning_rate * grad_w
            self.biases[i] += learning_rate * np.sum(delta, axis=0, keepdims=True)
            
            # Compute delta for next layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])

    def _sonify_gradient_update(self, gradient, layer):
        """Convert gradient updates to sound"""
        duration = 0.3
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t)
        
        # Map gradient values to frequencies
        max_grad = np.max(np.abs(gradient))
        if max_grad > 0:
            normalized_grads = gradient / max_grad
            for i in range(gradient.shape[0]):
                for j in range(gradient.shape[1]):
                    grad = normalized_grads[i,j]
                    if abs(grad) > 0.1:  # Only play significant gradients
                        freq = 440 * (2 ** (grad))
                        audio += 0.1 * np.sin(2 * np.pi * freq * t)
        
        self._play_audio(np.clip(audio, -1, 1))

    def forward_with_audio(self, x):
        """Forward propagation with audio feedback"""
        self.activations = [x]
        self.z_values = []
        
        current_activation = x
        for i in range(len(self.weights)):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current_activation = self.relu(z)
            self.activations.append(current_activation)
            
            # Play activation sound
            self._sonify_weights(self.weights[i], f"Layer {i} activation")
        
        return current_activation

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU activation function"""
        return np.where(x > 0, 1, 0)

    def _play_audio(self, audio):
        """Play audio using pygame"""
        # Ensure audio is in the correct format (16-bit signed integers)
        audio_int = np.int16(audio * 32767)
        # Convert to stereo by duplicating the mono channel
        stereo_audio = np.column_stack((audio_int, audio_int))
        pygame.sndarray.make_sound(stereo_audio).play()
        # Small delay to let the sound play
        time.sleep(0.5)

    def interactive_learning_demo(self):
        """Interactive demonstration of neural network learning"""
        print("\nInteractive Learning Demo")
        print("========================")
        
        # 1. Single Neuron Learning
        print("\nLesson 1: Single Neuron Learning")
        print("Listen to how a single neuron learns an AND gate:")
        
        # Create simple AND gate training data
        X = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([[0], [0], [0], [1]])
        
        for i in range(len(X)):
            print(f"\nTraining on input {X[i]}")
            self.learn_single_pattern(X[i:i+1], y[i:i+1], epochs=3)
            time.sleep(1)
        
        # 2. Pattern Recognition
        print("\nLesson 2: Pattern Recognition")
        print("Listen to how the network responds to different patterns:")
        
        # Generate some moon-shaped data
        X_moons, y_moons = make_moons(n_samples=4, noise=0.1)
        
        for i in range(len(X_moons)):
            print(f"\nPattern {i+1}:")
            prediction = self.forward_with_audio(X_moons[i:i+1])
            print(f"Activation pattern for class {y_moons[i]}")
            time.sleep(1)
        
        # 3. Gradient Descent Symphony
        print("\nLesson 3: The Gradient Descent Symphony")
        print("Listen to the network learn a complex pattern...")
        
        # Train on a spiral pattern
        X_circles, y_circles = make_circles(n_samples=8, noise=0.1)
        
        for epoch in range(5):
            print(f"\nEpoch {epoch + 1}")
            total_error = 0
            for i in range(len(X_circles)):
                prediction = self.forward_with_audio(X_circles[i:i+1])
                error = y_circles[i] - prediction
                total_error += abs(error[0,0])
                self._backward_with_audio(X_circles[i:i+1], error, 0.1)
            print(f"Average error: {total_error/len(X_circles):.3f}")
            time.sleep(1)

def start_learning():
    """Begin the neural network learning journey"""
    print("Welcome to Neural Networks Through Sound!")
    print("=======================================")
    print("We'll learn neural networks by HEARING how they work.")
    
    # Create a simple network
    nn = AuditoryNeuralNetwork([2, 4, 3, 1])
    
    # Start the interactive learning demo
    nn.interactive_learning_demo()
    
    return nn

if __name__ == "__main__":
    nn = start_learning()