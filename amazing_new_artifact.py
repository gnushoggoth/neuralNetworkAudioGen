import numpy as np
from scipy.io import wavfile
import pygame
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import colorsys

@dataclass
class NeuronSound:
    """Represents the sound profile of a neuron"""
    base_frequency: float
    harmonics: List[float]
    timbre_weights: List[float]
    color: Tuple[float, float, float]  # HSV color for visualization

class EnhancedAuditoryNeuralNetwork:
    """A neural network that teaches through combined audio and visual feedback"""
    
    def __init__(self, layer_sizes: List[int], sample_rate: int = 44100):
        """Initialize the neural network with specified layer sizes and sample rate"""
        self.layers = layer_sizes
        self.sample_rate = sample_rate
        self.weights = []
        self.biases = []
        self.neuron_sounds = []
        self.activations = []  # Store activations for visualization
        self.z_values = []     # Store pre-activation values
        
        # Initialize pygame mixer for audio
        pygame.mixer.init(frequency=sample_rate)
        
        # Initialize interactive plotting
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Initialize network components
        self._initialize_neuron_sounds()
        self._initialize_network()
        
    def _initialize_neuron_sounds(self):
        """Create unique, musically-informed sound profiles for each neuron"""
        # Musical scale frequencies (A3 to A5)
        base_frequencies = [
            220.0,  # A3
            246.9,  # B3
            261.6,  # C4
            293.7,  # D4
            329.6,  # E4
            349.2,  # F4
            392.0,  # G4
            440.0,  # A4
            493.9,  # B4
            523.3,  # C5
            587.3,  # D5
            659.3,  # E5
            698.5,  # F5
            784.0,  # G5
            880.0   # A5
        ]
        
        for layer_idx, layer_size in enumerate(self.layers):
            layer_sounds = []
            for neuron in range(layer_size):
                # Choose base frequency from musical scale
                base_freq = base_frequencies[neuron % len(base_frequencies)]
                
                # Create harmonics and timbre
                harmonics = [1.0, 2.0, 3.0, 4.0, 5.0]
                weights = [1.0 / (i + 1) for i in range(len(harmonics))]
                
                # Generate unique color for this neuron
                hue = (layer_idx / len(self.layers) + 
                      neuron / (layer_size * 3)) % 1.0
                color = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                
                layer_sounds.append(NeuronSound(
                    base_frequency=base_freq,
                    harmonics=harmonics,
                    timbre_weights=weights,
                    color=color
                ))
            self.neuron_sounds.append(layer_sounds)
    
    def _initialize_network(self):
        """Initialize network weights and biases with audio feedback"""
        print("\nInitializing neural network...")
        
        for i in range(len(self.layers) - 1):
            # He initialization
            weights = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2.0/self.layers[i])
            biases = np.zeros((1, self.layers[i+1]))
            
            self.weights.append(weights)
            self.biases.append(biases)
            
            # Create initialization sound
            self._sonify_layer_initialization(i)
            time.sleep(0.3)
    
    def _create_neuron_tone(self, neuron_sound: NeuronSound, activation: float, 
                           duration: float) -> np.ndarray:
        """Generate a complex tone for a neuron based on its sound profile"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        tone = np.zeros_like(t)
        
        if activation <= 0:
            return tone
        
        # Generate harmonics with envelope
        envelope = np.exp(-3 * t/duration)
        for harmonic, weight in zip(neuron_sound.harmonics, neuron_sound.timbre_weights):
            frequency = neuron_sound.base_frequency * harmonic
            amplitude = activation * weight * 0.5
            tone += amplitude * np.sin(2 * np.pi * frequency * t)
        
        return tone * envelope
    
    def _play_audio(self, audio: np.ndarray):
        """Safely play audio through pygame"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        audio_int = np.int16(audio * 32767)
        stereo_audio = np.column_stack((audio_int, audio_int))
        pygame.sndarray.make_sound(stereo_audio).play()
        time.sleep(len(audio) / self.sample_rate)
    
    def _sonify_layer_initialization(self, layer_idx: int):
        """Create sound for layer initialization"""
        duration = 0.3
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create a chord using the first few neurons in the layer
        layer_audio = np.zeros_like(t)
        for i in range(min(3, len(self.neuron_sounds[layer_idx]))):
            neuron_sound = self.neuron_sounds[layer_idx][i]
            layer_audio += self._create_neuron_tone(neuron_sound, 0.5, duration)
        
        self._play_audio(layer_audio)
    
    def _sonify_layer_activity(self, layer_idx: int, activations: np.ndarray, 
                              description: str = "", duration: float = 0.5):
        """Create audio representation of layer activity"""
        print(f"\nLayer {layer_idx} Activity: {description}")
        
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        layer_audio = np.zeros_like(t)
        
        activations_flat = activations.reshape(-1)  # Flatten activations
        for neuron_idx, activation in enumerate(activations_flat):
            if neuron_idx >= len(self.neuron_sounds[layer_idx]):
                break
            
            neuron_sound = self.neuron_sounds[layer_idx][neuron_idx]
            neuron_tone = self._create_neuron_tone(neuron_sound, activation, duration)
            layer_audio += neuron_tone
            
            # Print activation level
            activity_level = ('highly active' if activation > 0.7 else
                            'moderately active' if activation > 0.3 else 'quiet')
            print(f"  Neuron {neuron_idx}: {activation:.3f} ({activity_level})")
        
        if np.any(layer_audio != 0):
            self._play_audio(layer_audio)
    
    def forward_pass(self, X: np.ndarray, visualize: bool = True) -> np.ndarray:
        """Forward propagation with audio-visual feedback"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        self.activations = []  # Reset activations
        self.z_values = []     # Reset z_values
        current_activation = X
        self.activations.append(current_activation)
        
        for i in range(len(self.weights)):
            # Compute weighted sums
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Apply activation function (ReLU for hidden layers, sigmoid for output)
            if i == len(self.weights) - 1:
                current_activation = self.sigmoid(z)
            else:
                current_activation = self.relu(z)
            
            self.activations.append(current_activation)
            
            # Generate audio feedback
            self._sonify_layer_activity(i, current_activation)
            
            if visualize:
                self._update_visualization()
        
        return current_activation
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
              learning_rate: float = 0.01, batch_size: int = 32):
        """Train the network with audio-visual feedback"""
        history = []
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward pass
                predictions = self.forward_pass(X_batch, visualize=(i==0))
                loss = self.compute_loss(predictions, y_batch)
                total_loss += loss
                
                # Backward pass with gradient sonification
                self._backward_pass(X_batch, y_batch, learning_rate)
            
            avg_loss = total_loss / (len(X) / batch_size)
            history.append(avg_loss)
            
            if epoch % 5 == 0:
                print(f"\nEpoch {epoch}: Loss = {avg_loss:.4f}")
                self._update_visualization(history)
        
        return history
    
    def _backward_pass(self, X: np.ndarray, y: np.ndarray, learning_rate: float):
        """Backward propagation with gradient sonification"""
        m = X.shape[0]
        delta = self.activations[-1] - y.reshape(-1, 1)
        
        for layer in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW = np.dot(self.activations[layer].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Sonify significant gradient updates
            self._sonify_gradients(dW, layer)
            
            # Update weights and biases
            self.weights[layer] -= learning_rate * dW
            self.biases[layer] -= learning_rate * db
            
            # Compute delta for next layer
            if layer > 0:
                delta = np.dot(delta, self.weights[layer].T)
                delta *= self.relu_derivative(self.z_values[layer-1])
    
    def _sonify_gradients(self, gradients: np.ndarray, layer_idx: int):
        """Convert gradient updates to sound"""
        duration = 0.1
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Use gradient magnitudes to modulate frequency and amplitude
        magnitude = np.abs(gradients).mean()
        if magnitude > 0.01:  # Only play for significant updates
            frequency = 440 + (magnitude * 1000)
            audio = magnitude * 0.1 * np.sin(2 * np.pi * frequency * t)
            self._play_audio(audio)
    
    def _update_visualization(self, history: List[float] = None):
        """Update the network visualization"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot network architecture
        self._plot_network_architecture(self.ax1)
        
        # Plot loss history if available
        if history is not None:
            self.ax2.plot(history)
            self.ax2.set_xlabel('Epoch')
            self.ax2.set_ylabel('Loss')
            self.ax2.set_title('Training Loss')
        else:
            self.ax2.set_title('Network Activity')
            self._plot_activation_heatmap(self.ax2)
        
        plt.pause(0.01)
    
    def _plot_network_architecture(self, ax):
        """Plot network architecture with activations"""
        ax.set_title('Network Architecture')
        
        layer_positions = np.linspace(0, 1, len(self.layers))
        max_neurons = max(self.layers)
        
        for layer_idx, n_neurons in enumerate(self.layers):
            neuron_positions = np.linspace(-0.5, 0.5, n_neurons)
            
            # Plot neurons
            for i, pos in enumerate(neuron_positions):
                activation = 0
                if layer_idx < len(self.activations):
                    activation = self.activations[layer_idx][0, i] if i < self.activations[layer_idx].shape[1] else 0
                
                color = self.neuron_sounds[layer_idx][i].color
                size = 1000 * (activation if activation > 0 else 0.2)
                ax.scatter(layer_positions[layer_idx], pos, s=size, 
                          c=[color], alpha=0.6)
                
                # Plot connections to next layer
                if layer_idx < len(self.layers) - 1:
                    next_positions = np.linspace(-0.5, 0.5, self.layers[layer_idx + 1])
                    for next_pos in next_positions:
                        ax.plot([layer_positions[layer_idx], layer_positions[layer_idx + 1]],
                               [pos, next_pos], 'gray', alpha=0.1)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.6, 0.6)
        ax.axis('off')
    
    def _plot_activation_heatmap(self, ax):
        """Plot activation heatmap"""
        # Create a matrix of activations with fixed shape
        max_neurons = max(self.layers)
        data = np.zeros((len(self.activations), max_neurons))
        
        for i, activation in enumerate(self.activations):
            # Ensure activation is 2D
            if len(activation.shape) == 1:
                activation = activation.reshape(1, -1)
            # Fill in the actual values
            data[i, :activation.shape[1]] = activation[0, :]
        
        im = ax.imshow(data, aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Layer')
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU activation function"""
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -100, 100)))
    
    def compute_loss(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """Compute binary cross-entropy loss"""
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

def create_demo_datasets():
    """Create various datasets for demonstration"""
    datasets = {
        'moons': make_moons(n_samples=100, noise=0.15),  # Reduced sample size
        'circles': make_circles(n_samples=100, noise=0.15, factor=0.5),  # Reduced sample size
        'blobs': make_blobs(n_samples=100, centers=2, cluster_std=1.0)  # Reduced sample size
    }
    
    # Standardize all datasets
    for key in datasets:
        X, y = datasets[key]
        X = StandardScaler().fit_transform(X)
        datasets[key] = (X, y)
    
    return datasets

def main():
    """Main demonstration of the enhanced auditory neural network"""
    print("Welcome to the Enhanced Auditory Neural Network!")
    print("=============================================")
    print("This system combines visual and auditory feedback to help you understand")
    print("neural networks through multiple senses.")
    
    try:
        # Create network with audio-visual capabilities
        nn = EnhancedAuditoryNeuralNetwork([2, 4, 3, 1])  # Simplified architecture
        
        # Load demonstration datasets
        datasets = create_demo_datasets()
        
        # Train on different datasets
        for dataset_name, (X, y) in datasets.items():
            print(f"\nTraining on {dataset_name} dataset...")
            try:
                history = nn.train(X, y, epochs=20, learning_rate=0.01, batch_size=16)  # Reduced epochs
                
                # Save final state audio
                final_prediction = nn.forward_pass(X[0:1])
                wavfile.write(f"final_state_{dataset_name}.wav", 
                            nn.sample_rate, 
                            np.int16(final_prediction * 32767))
            except Exception as e:
                print(f"Error training on {dataset_name} dataset: {str(e)}")
                continue
        
        plt.ioff()
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()