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
    base_frequency: float
    harmonics: List[float]
    timbre_weights: List[float]

class AuditoryNeuralTeacher:
    def __init__(self, layer_sizes: List[int], sample_rate: int = 44100):
        """Initialize the neural network with specified layer sizes and sample rate"""
        self.layers = layer_sizes
        self.sample_rate = sample_rate
        self.weights = []
        self.biases = []
        self.neuron_sounds = []
        
        # Calculate maximum layer width for ASCII diagram
        self.max_layer_width = max(layer_sizes) * 4  # 4 chars per neuron
        
        # Initialize pygame mixer
        pygame.mixer.init(frequency=sample_rate)
        
        # Initialize network components
        self._initialize_neuron_sounds()
        self._initialize_network()
    
    def _initialize_neuron_sounds(self):
        """Create unique, musically-informed sound profiles for each neuron"""
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
        
        # Create sound profiles for ALL neurons in ALL layers
        for layer_idx, layer_size in enumerate(self.layers):
            layer_sounds = []
            for neuron in range(layer_size):
                # Choose base frequency from musical scale
                base_freq = base_frequencies[neuron % len(base_frequencies)]
                harmonics = [1.0, 2.0, 3.0, 4.0, 5.0]
                weights = [1.0 / (i + 1) for i in range(len(harmonics))]
                
                layer_sounds.append(NeuronSound(
                    base_frequency=base_freq,
                    harmonics=harmonics,
                    timbre_weights=weights
                ))
            self.neuron_sounds.append(layer_sounds)
    
    def _initialize_network(self):
        """Initialize network weights and biases"""
        print("\nInitializing neural network...")
        
        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2.0 / self.layers[i])
            bias_vector = np.zeros((1, self.layers[i + 1]))
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
            
            print(f"Layer {i} → {i+1} initialized")
            # Create initialization sound
            if len(self.neuron_sounds) > i and len(self.neuron_sounds[i]) > 0:
                initialization_tone = self._create_neuron_tone(
                    self.neuron_sounds[i][0],
                    0.5,
                    0.2
                )
                self._play_audio(initialization_tone)
                time.sleep(0.3)
    
    def _create_neuron_tone(self, neuron_sound: NeuronSound, activation: float, duration: float) -> np.ndarray:
        """Generate a complex tone for a neuron"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        tone = np.zeros_like(t)
        
        # Skip if activation is zero or negative
        if activation <= 0:
            return tone
        
        # Generate harmonics
        for harmonic, weight in zip(neuron_sound.harmonics, neuron_sound.timbre_weights):
            frequency = neuron_sound.base_frequency * harmonic
            amplitude = activation * weight * 0.5
            tone += amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope
        envelope = np.exp(-3 * t/duration)
        return tone * envelope
    
    def _play_audio(self, audio: np.ndarray):
        """Play audio safely"""
        # Avoid division by zero
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        audio_int = np.int16(audio * 32767)
        stereo_audio = np.column_stack((audio_int, audio_int))
        pygame.sndarray.make_sound(stereo_audio).play()
        time.sleep(len(audio) / self.sample_rate)
    
    def _sonify_layer_activity(self, layer_idx: int, activations: np.ndarray, 
                              description: str = "", duration: float = 0.5):
        """Create audio for layer activity"""
        print(f"\nLayer {layer_idx} Activity: {description}")
        
        if layer_idx >= len(self.neuron_sounds):
            print(f"Warning: No sounds defined for layer {layer_idx}")
            return
        
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        layer_audio = np.zeros_like(t)
        
        for neuron_idx, activation in enumerate(activations[0]):
            if neuron_idx >= len(self.neuron_sounds[layer_idx]):
                break
                
            neuron_sound = self.neuron_sounds[layer_idx][neuron_idx]
            neuron_tone = self._create_neuron_tone(neuron_sound, activation, duration)
            layer_audio += neuron_tone
            
            activity_level = ('highly active' if activation > 0.7 
                            else 'moderately active' if activation > 0.3 
                            else 'quiet')
            print(f"  Neuron {neuron_idx}: {activation:.3f} ({activity_level})")
        
        # Only play if we have any sound
        if np.any(layer_audio != 0):
            self._play_audio(layer_audio)
    
    def _sonify_single_neuron(self, layer_idx: int, neuron_idx: int, activation: float):
        """Play single neuron sound"""
        if (layer_idx < len(self.neuron_sounds) and 
            neuron_idx < len(self.neuron_sounds[layer_idx])):
            neuron_sound = self.neuron_sounds[layer_idx][neuron_idx]
            tone = self._create_neuron_tone(neuron_sound, activation, 0.5)
            if np.any(tone != 0):
                self._play_audio(tone)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _create_network_ascii(self, activations=None):
        """Create ASCII representation of the network with optional activations"""
        diagram = ["\nNetwork Architecture:\n"]
        max_neurons = max(self.layers)
        
        # Create each layer
        for layer_idx, n_neurons in enumerate(self.layers):
            # Layer header
            layer_label = f"Layer {layer_idx}"
            diagram.append(f"{layer_label:^{self.max_layer_width}}")
            
            # Neurons
            neurons = []
            for i in range(n_neurons):
                if activations is not None and len(activations) > layer_idx:
                    # Show activation value if available
                    act_val = activations[layer_idx][0][i] if layer_idx < len(activations) else 0
                    neuron_str = f"({act_val:4.2f})" if act_val > 0 else "( -- )"
                else:
                    neuron_str = "( ○ )"
                neurons.append(neuron_str)
            
            # Center the neurons
            padding = (max_neurons - n_neurons) * 6 // 2  # 6 chars per neuron
            diagram.append(" " * padding + " ".join(neurons))
            
            # Add connections to next layer
            if layer_idx < len(self.layers) - 1:
                next_n_neurons = self.layers[layer_idx + 1]
                connections = []
                for i in range(max(n_neurons, next_n_neurons)):
                    if i < n_neurons and i < next_n_neurons:
                        connections.append("  │  ")
                    elif i < n_neurons:
                        connections.append("  │  ")
                    else:
                        connections.append("     ")
                diagram.append(" " * padding + " ".join(connections))
        
        return "\n".join(diagram)

    def _create_activation_heatmap(self, activations):
        """Create ASCII heatmap of neuron activations"""
        heatmap = ["\nActivation Heatmap:\n"]
        
        # Define intensity characters
        intensity_chars = " ▁▂▃▄▅▆▇█"
        
        for layer_idx, layer_activations in enumerate(activations):
            heatmap.append(f"Layer {layer_idx}:")
            for i, activation in enumerate(layer_activations[0]):
                # Convert activation to intensity index (0-8)
                intensity = min(int(activation * 8), 8) if activation > 0 else 0
                bar = intensity_chars[intensity] * 10
                heatmap.append(f"  Neuron {i}: {bar} {activation:6.3f}")
            heatmap.append("")
        
        return "\n".join(heatmap)

    def teach_forward_pass(self, x: np.ndarray, pause_duration: float = 1.0):
        """Demonstrate forward pass with audio and explanations"""
        print("\n=== Forward Pass Demonstration ===")
        print("Listen to how information flows through the network...")
        
        # Print initial network state
        print(self._create_network_ascii())
        
        all_activations = [x]  # Store all activations for visualization
        current_activation = x
        for i in range(len(self.weights)):
            print(f"\nLayer {i} → {i+1}:")
            
            # Compute weighted sums
            print("1. Computing weighted sums...")
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self._sonify_layer_activity(i, z, "Pre-activation (weighted sums)")
            time.sleep(pause_duration)
            
            # Apply activation function
            print("\n2. Applying ReLU activation...")
            current_activation = self.relu(z)
            self._sonify_layer_activity(i+1, current_activation, "Post-activation (after ReLU)")
            time.sleep(pause_duration)
            
            # Store activation for visualization
            all_activations.append(current_activation)
            
            # Show network state with current activations
            print(self._create_network_ascii(all_activations))
            print(self._create_activation_heatmap(all_activations))
            
            self._explain_layer_transition(i, z, current_activation)
        
        return current_activation
    
    def _explain_layer_transition(self, layer_idx: int, pre_activation: np.ndarray, 
                                post_activation: np.ndarray):
        """Explain layer transition"""
        n_silent = np.sum(post_activation <= 0)
        n_active = np.sum(post_activation > 0)
        
        print(f"\nLayer {layer_idx} Analysis:")
        print(f"- {n_active} neurons became active")
        print(f"- {n_silent} neurons were silenced by ReLU")
        
        if n_active > 0:
            max_idx = np.argmax(post_activation)
            print(f"- Neuron {max_idx} had the strongest activation: {post_activation[0,max_idx]:.3f}")
            print(f"  Listen to its distinct tone...")
            self._sonify_single_neuron(layer_idx+1, max_idx, post_activation[0,max_idx])
    
    def teach_basic_concepts(self):
        """Interactive lesson on basic neural network concepts"""
        print("\n=== Neural Network Fundamentals ===")
        
        # Lesson 1: Single Neuron Behavior
        print("\nLesson 1: Single Neuron Activation")
        print("Listen to how a neuron responds to different inputs...")
        
        test_inputs = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for input_val in test_inputs:
            print(f"\nInput: {input_val}")
            activation = max(0, input_val)  # ReLU
            self._sonify_single_neuron(0, 0, activation)
            time.sleep(0.5)
        
        # Lesson 2: Layer Interaction
        print("\nLesson 2: Layer Interaction")
        print("Listen to how information flows between layers...")
        x = np.array([[0.5, 0.8]])
        self.teach_forward_pass(x)
        
        # Lesson 3: Pattern Recognition
        print("\nLesson 3: Pattern Recognition")
        print("Listen to how the network responds to different patterns...")
        
        patterns = np.array([
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 1]
        ])
        
        for pattern in patterns:
            print(f"\nPattern: {pattern}")
            self.teach_forward_pass(pattern.reshape(1, -1))
            time.sleep(1)

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