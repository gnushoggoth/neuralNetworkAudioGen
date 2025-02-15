import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pygame
import time

class SonicNeuralNetwork:
    def __init__(self, layers, sample_rate=44100):
        """
        A neural network that sonifies its internal states
        layers: list of integers for neurons per layer
        sample_rate: audio sample rate in Hz
        """
        self.layers = layers
        self.sample_rate = sample_rate
        self.weights = []
        self.biases = []
        self.audio_buffers = {
            'activations': np.zeros(sample_rate),
            'gradients': np.zeros(sample_rate),
            'decisions': np.zeros(sample_rate)
        }
        
        # Initialize weights with different frequency ranges per layer
        for i in range(len(layers) - 1):
            # He initialization
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0/layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
        # Initialize pygame mixer for real-time audio
        pygame.mixer.init(frequency=sample_rate)
        
        # Generate base frequencies for each layer
        self.layer_frequencies = self._generate_layer_frequencies()

    def _generate_layer_frequencies(self):
        """Generate distinct frequency ranges for each layer"""
        frequencies = {}
        base_freq = 220  # A3 note
        
        for i, neurons in enumerate(self.layers):
            # Each layer gets progressively higher frequencies
            layer_base = base_freq * (1.5 ** i)  # Changed from 2** to 1.5** for less extreme frequency jumps
            # Generate enough frequencies for each neuron in the layer
            frequencies[i] = np.linspace(
                layer_base,
                layer_base * 2,  # Up to one octave higher
                neurons  # Ensure we generate exactly enough frequencies for each neuron
            )
            
        return frequencies

    def sonify_layer(self, layer_idx, activations, duration=0.1):
        """Convert layer activations into sound"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t)
        
        # Get frequencies for this layer
        freqs = self.layer_frequencies[layer_idx]
        
        # Ensure activations are properly shaped
        if len(activations.shape) == 1:
            activations = activations.reshape(1, -1)
            
        # Handle shape mismatches by either truncating or padding
        if activations.shape[1] != len(freqs):
            if activations.shape[1] > len(freqs):
                # Truncate if too large
                activations = activations[:, :len(freqs)]
            else:
                # Pad with zeros if too small
                pad_width = ((0, 0), (0, len(freqs) - activations.shape[1]))
                activations = np.pad(activations, pad_width, mode='constant')
        
        # Generate sound for each neuron
        for neuron_idx, activation in enumerate(activations[0]):
            amplitude = np.clip(activation, 0, 1) * 0.1
            frequency = freqs[neuron_idx]
            
            fundamental = amplitude * np.sin(2 * np.pi * frequency * t)
            first_harmonic = 0.5 * amplitude * np.sin(4 * np.pi * frequency * t)
            second_harmonic = 0.25 * amplitude * np.sin(6 * np.pi * frequency * t)
            
            audio += fundamental + first_harmonic + second_harmonic
            
        return np.clip(audio, -1, 1)

    def forward_with_sound(self, X, play_audio=True):
        """Forward propagation with real-time audio feedback"""
        # Ensure input X is properly shaped
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        if X.shape[1] != self.layers[0]:
            raise ValueError(f"Input shape {X.shape} doesn't match network input size {self.layers[0]}")
            
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Apply activation function
            if i == len(self.weights) - 1:
                activation = self.sigmoid(z)
            else:
                activation = self.relu(z)
            
            self.activations.append(activation)
            
            if play_audio:
                audio = self.sonify_layer(i+1, activation)  # Use i+1 to skip input layer
                self._play_audio(audio)
                time.sleep(0.1)
        
        return self.activations[-1]

    def _play_audio(self, audio_data):
        """Play audio data using pygame"""
        # Convert to 16-bit PCM
        audio_data = np.int16(audio_data * 32767)
        # Convert mono to stereo by duplicating the channel
        stereo_data = np.column_stack((audio_data, audio_data))
        # Create pygame sound object and play
        sound = pygame.sndarray.make_sound(stereo_data)
        sound.play()

    def train_interactive(self, X, y, epochs=100, learning_rate=0.01):
        """Interactive training with audio-visual feedback"""
        plt.ion()  # Enable interactive plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        history = []
        
        for epoch in range(epochs):
            # Forward pass with sonification
            predictions = self.forward_with_sound(X)
            loss = self.compute_loss(predictions, y)
            history.append(loss)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if epoch % 5 == 0:
                # Update visualization
                ax1.clear()
                ax2.clear()
                
                # Plot decision boundary
                self.plot_decision_boundary(X, y, ax1)
                
                # Plot loss history
                ax2.plot(history)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_title(f'Training Loss (Epoch {epoch})')
                
                plt.pause(0.1)
                
        plt.ioff()
        return history

    def demo_network_sounds(self):
        """Demonstrate different network sounds"""
        print("\nDemonstrating network sounds:")
        
        # 1. Individual neuron activations
        print("1. Individual neuron sounds...")
        for layer_idx in range(len(self.layers)):
            print(f"\nLayer {layer_idx} neurons:")
            test_activation = np.zeros((1, self.layers[layer_idx]))
            for neuron in range(self.layers[layer_idx]):
                test_activation[0, neuron] = 1.0
                audio = self.sonify_layer(layer_idx, test_activation)
                self._play_audio(audio)
                time.sleep(0.5)
                test_activation[0, neuron] = 0.0

        # 2. Layer interaction sounds
        print("\n2. Layer interaction patterns...")
        test_patterns = [
            np.random.rand(1, self.layers[0]) for _ in range(3)  # Ensure correct input shape
        ]
        for pattern in test_patterns:
            self.forward_with_sound(pattern)
            time.sleep(1)

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, predictions, y):
        """Binary cross-entropy loss"""
        epsilon = 1e-15  # Small constant to avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    def backward(self, X, y, learning_rate):
        """Backward propagation with gradient sonification"""
        m = X.shape[0]
        
        # Initialize gradient storage
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        delta = self.activations[-1] - y.reshape(-1, 1)
        
        # Backward pass through layers
        for layer in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW[layer] = np.dot(self.activations[layer].T, delta) / m
            db[layer] = np.sum(delta, axis=0, keepdims=True) / m
            
            if layer > 0:
                # Compute delta for next layer
                delta = np.dot(delta, self.weights[layer].T)
                if layer != len(self.weights) - 1:  # For hidden layers
                    delta *= self.relu_derivative(self.z_values[layer-1])
            
            # Sonify gradients (optional)
            if layer == len(self.weights) - 1:  # Only sonify output layer gradients
                gradient_audio = self.sonify_gradients(dW[layer])
                self._play_audio(gradient_audio)
        
        # Update weights and biases
        for layer in range(len(self.weights)):
            self.weights[layer] -= learning_rate * dW[layer]
            self.biases[layer] -= learning_rate * db[layer]
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation function"""
        return np.where(x > 0, 1, 0)
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid activation function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def sonify_gradients(self, gradients, duration=0.05):
        """Convert gradients to sound"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t)
        
        # Use gradient magnitudes to modulate amplitude
        magnitude = np.abs(gradients).mean()
        frequency = 440 + (magnitude * 1000)  # Base frequency + gradient-based shift
        
        # Generate sound
        audio = magnitude * 0.1 * np.sin(2 * np.pi * frequency * t)
        
        return np.clip(audio, -1, 1)

    def plot_decision_boundary(self, X, y, ax):
        """Plot the decision boundary and data points"""
        # Set min and max values for both axes
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        # Create a mesh grid
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Flatten the mesh grid points and make predictions
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.forward_with_sound(mesh_points, play_audio=False)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Decision Boundary')

def create_educational_datasets():
    """Create various datasets for learning"""
    datasets = {
        'moons': make_moons(n_samples=200, noise=0.15),
        'circles': make_circles(n_samples=200, noise=0.15, factor=0.5),
        'blobs': make_blobs(n_samples=200, centers=2, cluster_std=1.0)
    }
    
    # Standardize all datasets
    for key in datasets:
        X, y = datasets[key]
        X = StandardScaler().fit_transform(X)
        datasets[key] = (X, y)
    
    return datasets

def main():
    # Initialize educational environment
    datasets = create_educational_datasets()
    
    print("Welcome to the Neural Network Auditory Learning System!")
    print("\nThis system will help you understand neural networks through:")
    print("1. Real-time visualization of decision boundaries")
    print("2. Auditory feedback of network states")
    print("3. Interactive training with multiple datasets")
    
    # Create network with audio capabilities
    nn = SonicNeuralNetwork([2, 8, 4, 1])
    
    # Demonstrate network sounds
    nn.demo_network_sounds()
    
    # Train on different datasets
    for dataset_name, (X, y) in datasets.items():
        print(f"\nTraining on {dataset_name} dataset...")
        nn.train_interactive(X, y, epochs=50)
        
        # Save final state audio
        final_audio = nn.forward_with_sound(X, play_audio=False)
        wavfile.write(f"final_state_{dataset_name}.wav", 
                     nn.sample_rate, 
                     np.int16(final_audio * 32767))

if __name__ == "__main__":
    main()