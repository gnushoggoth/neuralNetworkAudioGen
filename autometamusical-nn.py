import numpy as np
from scipy.io import wavfile
import pygame
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class WaveformType(Enum):
    SINE = "sine"
    SQUARE = "square"
    SAWTOOTH = "sawtooth"
    TRIANGLE = "triangle"

@dataclass
class MusicalNeuron:
    """Enhanced sound profile for a neuron with musical properties"""
    base_frequency: float
    harmonics: List[float]
    harmonic_weights: List[float]
    waveform_type: WaveformType
    attack: float
    decay: float
    sustain: float
    release: float
    vibrato_rate: float
    vibrato_depth: float

class HarmonicNeuralNetwork:
    def __init__(self, layer_sizes: List[int], sample_rate: int = 128000):  # Updated to 128kHz
        """Initialize the neural network with specified layer sizes"""
        self.layers = layer_sizes
        self.sample_rate = sample_rate
        self.neurons = []
        self.weights = []
        self.biases = []
        self.activations = []
        self.z_values = []
        
        # Initialize pygame mixer with higher sample rate
        pygame.mixer.init(frequency=sample_rate)
        
        # Musical scale frequencies (based on just intonation)
        self.base_frequencies = [
            220.00,  # A3
            247.50,  # B3
            264.00,  # C4
            293.33,  # D4
            330.00,  # E4
            352.00,  # F4
            396.00,  # G4
            440.00   # A4
        ]
        
        # Initialize network components
        self._initialize_musical_neurons()
        self._initialize_weights_and_biases()

    def generate_waveform(self, frequency: float, amplitude: float, duration: float, 
                         wave_type: WaveformType) -> np.ndarray:
        """Generate different types of waveforms"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        
        if wave_type == WaveformType.SINE:
            return amplitude * np.sin(2 * np.pi * frequency * t)
        elif wave_type == WaveformType.SQUARE:
            return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        elif wave_type == WaveformType.SAWTOOTH:
            return amplitude * (2 * (t * frequency - np.floor(0.5 + t * frequency)))
        elif wave_type == WaveformType.TRIANGLE:
            return amplitude * (2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1)
        else:
            raise ValueError(f"Unknown waveform type: {wave_type}")

    def _initialize_musical_neurons(self):
        """Create musically-informed neurons for each layer"""
        waveform_types = [WaveformType.SINE, WaveformType.TRIANGLE, 
                         WaveformType.SAWTOOTH, WaveformType.SQUARE]
        
        for layer_idx, layer_size in enumerate(self.layers):
            layer_neurons = []
            for neuron_idx in range(layer_size):
                base_freq = self.base_frequencies[neuron_idx % len(self.base_frequencies)]
                harmonics = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                weights = [1.0, 0.5, 0.33, 0.25, 0.2, 0.16]
                
                # Cycle through waveform types for variety
                waveform = waveform_types[neuron_idx % len(waveform_types)]
                
                # ADSR envelope parameters (milliseconds)
                attack = 15.0 + (neuron_idx * 0.5)
                decay = 45.0 + (neuron_idx * 1.0)
                sustain = 0.7 - (neuron_idx * 0.02)
                release = 200.0 + (neuron_idx * 2.0)
                
                # Subtle vibrato
                vibrato_rate = 5.0 + (neuron_idx * 0.2)
                vibrato_depth = 0.015
                
                layer_neurons.append(MusicalNeuron(
                    base_frequency=base_freq,
                    harmonics=harmonics,
                    harmonic_weights=weights,
                    waveform_type=waveform,
                    attack=attack,
                    decay=decay,
                    sustain=sustain,
                    release=release,
                    vibrato_rate=vibrato_rate,
                    vibrato_depth=vibrato_depth
                ))
            self.neurons.append(layer_neurons)

    def _initialize_weights_and_biases(self):
        """Initialize network weights and biases"""
        print("\nInitializing neural network with musical feedback...")
        
        for i in range(len(self.layers) - 1):
            # He initialization
            weights = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2.0/self.layers[i])
            biases = np.zeros((1, self.layers[i+1]))
            
            self.weights.append(weights)
            self.biases.append(biases)
            
            # Create initialization sound
            duration = 0.5
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            init_audio = np.zeros_like(t)
            
            # Add tones for first few neurons in the layer
            for j in range(min(3, len(self.neurons[i]))):
                neuron = self.neurons[i][j]
                init_audio += self._create_neuron_tone(neuron, 0.5, duration)
            
            self._play_audio(init_audio)
            time.sleep(0.2)

    def _create_adsr_envelope(self, duration: float, neuron: MusicalNeuron) -> np.ndarray:
        """Create ADSR envelope for smooth sound shaping"""
        samples = int(self.sample_rate * duration)
        envelope = np.zeros(samples)
        
        attack_samples = int(neuron.attack * self.sample_rate / 1000)
        decay_samples = int(neuron.decay * self.sample_rate / 1000)
        release_samples = int(neuron.release * self.sample_rate / 1000)
        
        attack_end = min(attack_samples, samples)
        envelope[:attack_end] = np.linspace(0, 1, attack_end)
        
        decay_end = min(attack_end + decay_samples, samples)
        if decay_end > attack_end:
            envelope[attack_end:decay_end] = np.linspace(1, neuron.sustain, decay_end - attack_end)
        
        sustain_end = max(0, samples - release_samples)
        if sustain_end > decay_end:
            envelope[decay_end:sustain_end] = neuron.sustain
        
        if samples > sustain_end:
            envelope[sustain_end:] = np.linspace(neuron.sustain, 0, samples - sustain_end)
        
        return envelope

    def _create_neuron_tone(self, neuron: MusicalNeuron, activation: float, duration: float) -> np.ndarray:
        """Generate a complex musical tone for a neuron"""
        if activation <= 0:
            return np.zeros(int(self.sample_rate * duration))
        
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        tone = np.zeros_like(t)
        
        # Apply vibrato
        vibrato = neuron.vibrato_depth * np.sin(2 * np.pi * neuron.vibrato_rate * t)
        frequency_mod = 1 + vibrato
        
        # Generate harmonics with the specified waveform
        for harmonic, weight in zip(neuron.harmonics, neuron.harmonic_weights):
            frequency = neuron.base_frequency * harmonic * frequency_mod
            harmonic_wave = self.generate_waveform(
                frequency, weight * activation, duration, neuron.waveform_type
            )
            tone += harmonic_wave
        
        # Apply envelope and soft clipping
        envelope = self._create_adsr_envelope(duration, neuron)
        tone = tone * envelope
        tone = np.tanh(tone)  # Soft clip
        
        return tone

    def _play_audio(self, audio: np.ndarray):
        """Safely play audio with stereo enhancement"""
        max_val = np.max(np.abs(audio))
        if (max_val > 0):
            audio = audio / max_val
        
        # Create stereo effect
        left_channel = audio
        right_channel = np.roll(audio, 22)  # Small delay for stereo width
        stereo_audio = np.column_stack((
            np.int16(left_channel * 32767),
            np.int16(right_channel * 32767)
        ))
        
        pygame.sndarray.make_sound(stereo_audio).play()
        time.sleep(len(audio) / self.sample_rate)

    def _create_weight_based_audio(self, duration: float) -> np.ndarray:
        """Create audio based on direct weight-to-time mapping"""
        num_samples = int(self.sample_rate * duration)
        
        # Calculate total number of weights
        total_weights = sum(w.size for w in self.weights)
        
        # Divide time linearly based on number of weights
        samples_per_weight = num_samples // total_weights
        
        # Initialize audio buffer
        audio = np.zeros(num_samples)
        
        current_sample = 0
        for layer_idx, weight_matrix in enumerate(self.weights):
            # Flatten weights and get mean for each time step
            flat_weights = weight_matrix.flatten()
            
            for weight_idx, weight in enumerate(flat_weights):
                if current_sample + samples_per_weight > num_samples:
                    break
                    
                # Direct mapping of weight to amplitude
                amplitude = np.tanh(weight)  # Normalize to [-1, 1]
                
                # Get corresponding neuron's base frequency
                source_neuron = weight_idx % self.layers[layer_idx]
                freq = self.base_frequencies[source_neuron % len(self.base_frequencies)]
                
                # Generate the tone for this time segment
                t = np.linspace(0, duration * samples_per_weight / num_samples, 
                              samples_per_weight)
                segment = amplitude * np.sin(2 * np.pi * freq * t)
                
                # Add to audio buffer
                audio[current_sample:current_sample + samples_per_weight] = segment
                current_sample += samples_per_weight
        
        return audio

    def _sonify_layer_activity(self, layer_idx: int, activations: np.ndarray, 
                              description: str = "", duration: float = 0.75):
        """Create musical audio representation of layer activity using weight-based timing"""
        if layer_idx >= len(self.neurons):
            return np.zeros(int(self.sample_rate * duration))  # Return empty audio if layer invalid
        
        print(f"\nLayer {layer_idx} Activity: {description}")
        
        # Initialize audio buffer
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        layer_audio = np.zeros_like(t)
        
        # Create weight-based audio
        if layer_idx < len(self.weights):
            weights = self.weights[layer_idx]
            samples_per_weight = len(t) // weights.size
            
            for i, weight in enumerate(weights.flatten()):
                if i * samples_per_weight >= len(t):
                    break
                    
                # Map weight to frequency and amplitude
                amplitude = np.tanh(weight) * 0.5
                base_freq = self.base_frequencies[i % len(self.base_frequencies)]
                
                # Generate time segment for this weight
                segment_t = t[i*samples_per_weight:(i+1)*samples_per_weight]
                if len(segment_t) > 0:  # Ensure we have samples to generate
                    segment = amplitude * np.sin(2 * np.pi * base_freq * segment_t)
                    layer_audio[i*samples_per_weight:(i+1)*samples_per_weight] = segment
        
        # Modulate audio based on activations
        activations_flat = activations.flatten()
        for i, activation in enumerate(activations_flat):
            if activation > 0.1 and i < len(self.neurons[layer_idx]):
                neuron = self.neurons[layer_idx][i]
                neuron_sound = self._create_neuron_tone(neuron, activation, duration)
                layer_audio += neuron_sound * activation
        
        # Normalize audio
        if np.any(layer_audio != 0):
            layer_audio = layer_audio / np.max(np.abs(layer_audio))
        
        # Only play audio if requested (during training/visualization)
        if description:  # Only play if description provided (interactive mode)
            self._play_audio(layer_audio)
        
        return layer_audio  # Always return the audio data

    def forward_pass(self, X: np.ndarray, play_audio: bool = True) -> np.ndarray:
        """Forward propagation with musical feedback"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        self.activations = [X]
        self.z_values = []
        current_activation = X
        
        for i in range(len(self.weights)):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            current_activation = self.sigmoid(z) if i == len(self.weights) - 1 else self.relu(z)
            self.activations.append(current_activation)
            
            if play_audio:
                self._sonify_layer_activity(i, current_activation)
                time.sleep(0.2)
        
        return current_activation

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
              learning_rate: float = 0.01, batch_size: int = 32):
        """Train the network with musical feedback"""
        history = []
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                predictions = self.forward_pass(X_batch, play_audio=(i == 0))
                loss = self.compute_loss(predictions, y_batch)
                total_loss += loss
                
                self._backward_pass(X_batch, y_batch, learning_rate)
            
            avg_loss = total_loss / (len(X) / batch_size)
            history.append(avg_loss)
            
            if epoch % 5 == 0:
                print(f"\nEpoch {epoch}: Loss = {avg_loss:.4f}")
        
        return history

    def _backward_pass(self, X: np.ndarray, y: np.ndarray, learning_rate: float):
        """Backward propagation with gradient sonification"""
        m = X.shape[0]
        delta = self.activations[-1] - y.reshape(-1, 1)
        
        for layer in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[layer].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Sonify significant gradient updates
            if np.max(np.abs(dW)) > 0.01:
                self._sonify_gradients(dW, layer)
            
            self.weights[layer] -= learning_rate * dW
            self.biases[layer] -= learning_rate * db
            
            if layer > 0:
                delta = np.dot(delta, self.weights[layer].T)
                delta *= self.relu_derivative(self.z_values[layer-1])

    def _sonify_gradients(self, gradients: np.ndarray, layer_idx: int):
        """Convert gradient updates to musical sounds"""
        duration = 0.1
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t)
        
        max_grad = np.max(np.abs(gradients))
        if max_grad > 0:
            for i in range(min(3, gradients.shape[0])):  # Sonify top 3 gradients
                magnitude = np.max(np.abs(gradients[i]))
                if magnitude > 0.01:
                    freq = 440 * (1 + magnitude)
                    audio += 0.2 * self.generate_waveform(
                        freq, magnitude/max_grad, duration, WaveformType.SINE
                    )
        
        if np.any(audio != 0):
            self._play_audio(audio)

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

    def save_audio_sample(self, filename: str, audio: np.ndarray):
        """Save audio sample to WAV file"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        wavfile.write(filename, self.sample_rate, audio.astype(np.float32))


def demo_network():
    """Demonstrate the musical neural network"""
    print("Initializing Musical Neural Network...")
    nn = HarmonicNeuralNetwork([2, 4, 3, 1])
    
    # 1. Demonstrate different waveform types
    print("\nDemonstrating basic waveforms...")
    duration = 1.0
    frequency = 440  # A4 note
    amplitude = 0.5
    
    for wave_type in WaveformType:
        print(f"\nGenerating {wave_type.value} wave")
        wave = nn.generate_waveform(frequency, amplitude, duration, wave_type)
        nn._play_audio(wave)
        # Save waveform to file
        nn.save_audio_sample(f"{wave_type.value}_wave.wav", wave)
        time.sleep(0.5)
    
    # 2. Demonstrate neural network with musical feedback
    print("\nDemonstrating neural network learning...")
    
    # Create simple XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Train network with musical feedback
    print("\nTraining network on XOR problem...")
    history = nn.train(X, y, epochs=50, learning_rate=0.1, batch_size=1)
    
    # Test network with musical feedback
    print("\nTesting network predictions...")
    for i in range(len(X)):
        prediction = nn.forward_pass(X[i:i+1])
        print(f"\nInput: {X[i]}")
        print(f"Target: {y[i][0]}")
        print(f"Prediction: {prediction[0][0]:.3f}")
        time.sleep(1)
    
    return nn


def create_musical_composition(nn: HarmonicNeuralNetwork, duration: float = 5.0):
    """Create a musical composition using the neural network"""
    print("\nCreating musical composition...")
    
    # Generate a sequence of inputs
    t = np.linspace(0, 2*np.pi, 20)
    inputs = np.column_stack((np.sin(t), np.cos(t)))
    
    # Initialize audio buffer
    sample_length = int(nn.sample_rate * duration)
    composition = np.zeros(sample_length)
    
    # Generate sounds for each input
    step_duration = duration / len(inputs)
    for i, input_vector in enumerate(inputs):
        # Forward pass without playing audio
        prediction = nn.forward_pass(input_vector.reshape(1, -1), play_audio=False)
        
        # Create sound for this step
        for layer_idx, layer_activation in enumerate(nn.activations):
            if layer_idx > 0:  # Skip input layer
                start_idx = int(i * step_duration * nn.sample_rate)
                end_idx = int((i + 1) * step_duration * nn.sample_rate)
                if end_idx > len(composition):
                    end_idx = len(composition)
                
                # Generate layer sound without playing it
                layer_sound = nn._sonify_layer_activity(
                    layer_idx-1, 
                    layer_activation, 
                    duration=step_duration
                )
                
                if layer_sound is not None and len(layer_sound) > 0:
                    # Ensure we don't exceed buffer length
                    sound_length = min(len(layer_sound), end_idx - start_idx)
                    
                    # Add crossfade
                    fade_length = int(0.1 * nn.sample_rate)  # 100ms crossfade
                    if i > 0 and start_idx + fade_length < len(composition):
                        # Apply fadeout to previous segment
                        fadeout = np.linspace(1, 0, fade_length)
                        composition[start_idx:start_idx+fade_length] *= fadeout
                        # Apply fadein to new segment
                        fadein = np.linspace(0, 1, fade_length)
                        layer_sound[:fade_length] *= fadein
                    
                    # Add to composition
                    composition[start_idx:start_idx+sound_length] += layer_sound[:sound_length]
    
    # Normalize final composition
    max_amplitude = np.max(np.abs(composition))
    if max_amplitude > 0:
        composition = composition / max_amplitude
    
    # Save composition
    nn.save_audio_sample("neural_composition.wav", composition)
    print("\nComposition saved as 'neural_composition.wav'")
    
    return composition


if __name__ == "__main__":
    # Run demo
    nn = demo_network()
    
    # Create musical composition
    composition = create_musical_composition(nn)
    
    print("\nDemo complete! Check the generated WAV files for the audio output.")