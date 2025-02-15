import numpy as np
from scipy.io import wavfile
import pygame
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import colorsys

@dataclass
class NeuronSound:
    """Represents the sound profile of a neuron"""
    base_frequency: float  # Base frequency for this neuron
    harmonics: List[float]  # List of harmonic multipliers
    timbre_weights: List[float]  # Weights for each harmonic

class AuditoryNeuralTeacher:
    """A neural network that teaches through interactive sound and visualization"""
    
    def __init__(self, layer_sizes: List[int], sample_rate: int = 44100):
        self.layers = layer_sizes
        self.sample_rate = sample_rate
        self.weights = []
        self.biases = []
        self.neuron_sounds = []  # Sound profiles for each neuron
        
        pygame.mixer.init(frequency=sample_rate)
        
        # Initialize unique sound profiles for each neuron
        self._initialize_neuron_sounds()
        
        # Initialize network with educational sound design
        self._initialize_network_with_sound()

    def _initialize_neuron_sounds(self):
        """Create unique, musically-informed sound profiles for each neuron"""
        # Use musical intervals based on the harmonic series
        base_frequencies = [
            220.0,  # A3
            247.5,  # B3
            261.6,  # C4
            293.7,  # D4
            329.6,  # E4
            349.2,  # F4
            392.0,  # G4
            440.0   # A4
        ]
        
        # Create sound profiles for each layer
        self.neuron_sounds = []
        for layer_size in self.layers:
            layer_sounds = []
            for neuron in range(layer_size):
                # Choose base frequency from musical scale
                base_freq = base_frequencies[neuron % len(base_frequencies)]
                # Create harmonics (more harmonics for deeper layers)
                harmonics = [1.0, 2.0, 3.0, 4.0, 5.0]
                # Weights decrease for higher harmonics
                weights = [1.0 / (i + 1) for i in range(len(harmonics))]
                
                layer_sounds.append(NeuronSound(
                    base_frequency=base_freq,
                    harmonics=harmonics,
                    timbre_weights=weights
                ))
            self.neuron_sounds.append(layer_sounds)

    def _create_neuron_tone(self, neuron_sound: NeuronSound, activation: float, duration: float) -> np.ndarray:
        """Generate a complex tone for a neuron based on its activation"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        tone = np.zeros_like(t)
        
        # Generate harmonics
        for harmonic, weight in zip(neuron_sound.harmonics, neuron_sound.timbre_weights):
            frequency = neuron_sound.base_frequency * harmonic
            # Amplitude modulation based on activation
            amplitude = activation * weight * 0.5
            tone += amplitude * np.sin(2 * np.pi * frequency * t)
            
        # Apply envelope
        envelope = np.exp(-3 * t/duration)
        return tone * envelope

    def _sonify_layer_activity(self, layer_idx: int, activations: np.ndarray, 
                              description: str = "", duration: float = 0.5):
        """Create an audio representation of an entire layer's activity"""
        print(f"\nLayer {layer_idx} Activity: {description}")
        
        # Create combined audio for all neurons in the layer
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        layer_audio = np.zeros_like(t)
        
        for neuron_idx, activation in enumerate(activations[0]):
            neuron_sound = self.neuron_sounds[layer_idx][neuron_idx]
            neuron_tone = self._create_neuron_tone(neuron_sound, activation, duration)
            layer_audio += neuron_tone
            
            # Print neuron activity details
            print(f"  Neuron {neuron_idx}: {activation:.3f} "
                  f"({'highly active' if activation > 0.7 else 'moderately active' if activation > 0.3 else 'quiet'})")
        
        # Normalize and play
        layer_audio = layer_audio / np.max(np.abs(layer_audio))
        self._play_audio(layer_audio)

    def teach_forward_pass(self, x: np.ndarray, pause_duration: float = 1.0):
        """Educational forward pass that explains each step"""
        print("\n=== Forward Pass Demonstration ===")
        print("Listen to how information flows through the network...")
        
        current_activation = x
        for i in range(len(self.weights)):
            print(f"\nLayer {i} → {i+1}:")
            print("1. Computing weighted sums...")
            
            # Weighted sum computation
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            
            # Sonify pre-activation
            self._sonify_layer_activity(i, z, "Pre-activation (weighted sums)")
            time.sleep(pause_duration)
            
            print("\n2. Applying ReLU activation...")
            # Activation
            current_activation = self.relu(z)
            
            # Sonify post-activation
            self._sonify_layer_activity(i, current_activation, "Post-activation (after ReLU)")
            time.sleep(pause_duration)
            
            # Explain what happened
            self._explain_layer_transition(i, z, current_activation)
        
        return current_activation

    def _explain_layer_transition(self, layer_idx: int, pre_activation: np.ndarray, 
                                post_activation: np.ndarray):
        """Provide educational explanation of what happened in this layer"""
        n_silent = np.sum(post_activation <= 0)
        n_active = np.sum(post_activation > 0)
        
        print(f"\nLayer {layer_idx} Analysis:")
        print(f"- {n_active} neurons became active")
        print(f"- {n_silent} neurons were silenced by ReLU")
        
        if n_active > 0:
            max_idx = np.argmax(post_activation)
            print(f"- Neuron {max_idx} had the strongest activation: {post_activation[0,max_idx]:.3f}")
            print(f"  Listen to its distinct tone...")
            self._sonify_single_neuron(layer_idx, max_idx, post_activation[0,max_idx])

    def teach_basic_concepts(self):
        """Interactive lesson on basic neural network concepts"""
        print("\n=== Neural Network Fundamentals ===")
        
        # 1. Single Neuron Behavior
        print("\nLesson 1: Single Neuron Activation")
        print("Listen to how a neuron responds to different inputs...")
        
        test_inputs = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for input_val in test_inputs:
            print(f"\nInput: {input_val}")
            # Create artificial single-neuron activation
            activation = max(0, input_val)  # ReLU
            self._sonify_single_neuron(0, 0, activation)
            time.sleep(0.5)
        
        # 2. Layer Interaction
        print("\nLesson 2: Layer Interaction")
        print("Listen to how information flows between layers...")
        
        # Generate simple pattern
        x = np.array([[0.5, 0.8]])
        self.teach_forward_pass(x)
        
        # 3. Pattern Recognition
        print("\nLesson 3: Pattern Recognition")
        print("Listen to how the network responds to different patterns...")
        
        # Generate some simple patterns
        X_patterns = np.array([
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 1]
        ])
        
        for pattern in X_patterns:
            print(f"\nPattern: {pattern}")
            self.teach_forward_pass(pattern.reshape(1, -1))
            time.sleep(1)

    def _play_audio(self, audio: np.ndarray):
        """Play audio using pygame"""
        audio_int = np.int16(audio * 32767)
        stereo_audio = np.column_stack((audio_int, audio_int))
        pygame.sndarray.make_sound(stereo_audio).play()
        time.sleep(len(audio) / self.sample_rate)

    def _sonify_single_neuron(self, layer_idx: int, neuron_idx: int, activation: float):
        """Play the sound of a single neuron"""
        neuron_sound = self.neuron_sounds[layer_idx][neuron_idx]
        tone = self._create_neuron_tone(neuron_sound, activation, 0.5)
        self._play_audio(tone)

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)

    def _initialize_network_with_sound(self):
        """Initialize network weights and biases with auditory feedback"""
        print("\nInitializing neural network...")
        
        # Initialize weights and biases for each layer
        for i in range(len(self.layers) - 1):
            # Initialize weights with Xavier/Glorot initialization
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2.0 / self.layers[i])
            bias_vector = np.zeros((1, self.layers[i + 1]))
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
            
            # Play a short sound to indicate layer initialization
            print(f"Layer {i} → {i+1} initialized")
            initialization_tone = self._create_neuron_tone(
                self.neuron_sounds[i][0],
                0.5,
                0.2
            )
            self._play_audio(initialization_tone)
            time.sleep(0.3)

def start_neural_exploration():
    """Begin an interactive neural network learning session"""
    print("Welcome to Neural Network Auditory Learning!")
    print("==========================================")
    print("We'll learn neural networks by HEARING how they work.")
    print("Each neuron has its own musical voice...")
    
    # Create network with a simple architecture
    nn = AuditoryNeuralTeacher([2, 4, 3, 1])
    
    # Start the interactive learning session
    nn.teach_basic_concepts()
    
    return nn

if __name__ == "__main__":
    nn = start_neural_exploration()