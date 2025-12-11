# Handwritten Digit Recognition - KNN and Neural Network

A C# WPF application implementing multiple machine learning algorithms for handwritten digit recognition using the MNIST dataset.

## 📋 Overview

This project demonstrates different approaches to handwritten digit recognition:
- **K-Nearest Neighbors (KNN)** algorithm
- **Simple Artificial Neural Network (ANN)**
- **Deep Neural Network (DNN)** with GPU acceleration
- **Convolutional Neural Network (CNN)** from scratch
- **ML.NET** integration for machine learning models

The application provides an interactive UI for drawing digits and testing the recognition algorithms in real-time.

## 🚀 Features

- **Multiple Recognition Algorithms**
  - K-Nearest Neighbors with configurable K value
  - Simple perceptron-based neural network
  - Deep neural network with multiple layers and GPU acceleration (ILGPU)
  - Convolutional Neural Network (CNN) with convolution and pooling layers
  - ML.NET-based classification models

- **Interactive Drawing Canvas**
  - Draw digits directly on a 28x28 pixel canvas
  - Real-time digit recognition
  - Clear and redraw functionality

- **Training Capabilities**
  - Load MNIST dataset from CSV files
  - Train ANN models with progress tracking
  - Save and load trained models (JSON format)
  - Visual progress indicators during training

- **Image Processing**
  - Display MNIST dataset images
  - Generate digit images using KNN averaging
  - Generate average digit images with SimpleDigitGenerator
  - Image browsing with context menu actions
  - Batch loading of training images

- **GPU Acceleration**
  - CUDA GPU support for Deep Neural Network training and inference
  - Automatic fallback to CPU if GPU is not available
  - Real-time GPU availability display in UI

## 🏗️ Project Structure

```
├── אלגוריתם שכן קרוב זיהוי מספרים בכתב יד/   # Main WPF Application
│   ├── Services/
│   │   ├── KnnDigitRecognizer.cs          # K-Nearest Neighbors implementation
│   │   ├── SimpleANN.cs                   # Simple neural network
│   │   ├── NeuralNetwork.cs               # Neural network wrapper
│   │   ├── DeepNeuralNetwork.cs           # Deep learning model with GPU support
│   │   ├── ConvolutionalNeuralNetwork.cs  # CNN from scratch
│   │   └── SimpleDigitGenerator.cs        # Digit image generator
│   ├── ViewModels/
│   │   ├── MainViewModel.cs               # Main view logic
│   │   └── ViewModelBase.cs               # Base ViewModel class
│   ├── Models/
│   │   └── ImageModel.cs                  # Image data model
│   ├── Commands/
│   │   ├── RelayCommand.cs                # Command implementation
│   │   └── RelayCommandAsync.cs           # Async command support
│   ├── ML.DotNet/                         # ML.NET Model 1
│   ├── ML.DotNet2/                        # ML.NET Model 2
│   └── MainWindow.xaml                    # Main UI
├── MLModel1_ConsoleApp1/                  # Console application demo
└── README.md                              # This file
```

## 🛠️ Technologies

- **.NET 8.0** - Application framework
- **WPF (Windows Presentation Foundation)** - User interface
- **C#** - Programming language
- **ML.NET** - Machine learning framework
- **ILGPU** - GPU acceleration library
- **Newtonsoft.Json** - Model serialization
- **MVVM Pattern** - Architecture pattern

### NuGet Packages
- Microsoft.ML (1.7.1)
- Microsoft.ML.FastTree (1.7.1)
- Microsoft.ML.Vision (1.7.1)
- ILGPU (1.5.3)
- ILGPU.Algorithms (1.5.3)
- Newtonsoft.Json (13.0.3)
- SciSharp.TensorFlow.Redist (2.3.1)
- System.Drawing.Common (7.0.0)

## 📦 Installation

### Prerequisites
- Visual Studio 2022 or later
- .NET 8.0 SDK
- Windows OS (for WPF)
- (Optional) CUDA-compatible GPU for accelerated training

### Steps
1. Clone the repository:
```bash
git clone https://github.com/YAKOV-FRIDMAN/Nearest_Neighbor_Algorithm_and_Neural_Network_Handwritten_Number_Recognition.git
```

2. Open the solution file:
```
אלגוריתם שכן קרוב זיהוי מספרים בכתב יד.sln
```

3. Restore NuGet packages:
```bash
dotnet restore
```

4. Build the solution:
```bash
dotnet build
```

5. Run the application:
```bash
dotnet run --project "אלגוריתם שכן קרוב זיהוי מספרים בכתב יד"
```

## 📖 Usage

### Loading Training Data
1. Click "Select File..." button
2. Choose a MNIST CSV file (format: label, pixel1, pixel2, ..., pixel784)
3. Wait for the data to load and images to display

### Training Models

#### Simple ANN
1. Load training data first
2. Click "Train ANN model" button
3. Monitor progress bar (100 epochs)
4. Click "Save To File" to save the trained model
5. Use "Load ANN File model" to load a saved model

#### Deep Neural Network
1. Load training data first
2. Check GPU availability displayed in the UI (shows CUDA status)
3. Click "Train ANN model 1" button
4. Monitor progress bar (GPU-accelerated if available)
5. Click "Save To File 1" to save the trained model
6. Use "Load ANN File model 1" to load a saved model

#### Convolutional Neural Network (CNN)
1. Load training data first
2. Click "Train CNN model" button
3. Monitor progress bar during training
4. Click "Save To File 2" to save the trained CNN model
5. Use "Load CNN File model" to load a saved CNN model

### Recognizing Digits

#### Using Drawing Canvas
1. Draw a digit on the canvas (bottom right)
2. Click "Extract Pixels" button
3. View the recognized digit below the canvas
4. Click "Clear" to reset the canvas

#### Using Loaded Images
1. Right-click on any loaded image
2. Select "זיהוי מספר" (Recognize Number) from context menu
3. View classification results

### Generating Digits
1. Enter a digit (0-9) in the "NumberToGenerator" field
2. Set a threshold value
3. Click "Generator" button
4. The system uses KNN to generate an average digit image

## 🧮 Algorithms

### K-Nearest Neighbors (KNN)
- Uses Euclidean distance for similarity measurement
- Configurable K value (default: 3)
- Classifies based on majority voting of K nearest neighbors
- Can generate average digit images from nearest neighbors

### Simple Artificial Neural Network
- Single-layer perceptron model
- 10 output neurons (one per digit)
- Simple learning rule with 0.1 learning rate
- 100 training epochs
- Uses random weight initialization

### Deep Neural Network
- Multi-layer architecture
- 784 input neurons (28x28 pixels)
- Configurable hidden layers
- GPU acceleration using ILGPU (CUDA support)
- Automatic fallback to CPU if GPU unavailable
- SIMD optimizations for CPU operations
- He weight initialization for better convergence
- Supports backpropagation training
- Progress tracking during training

### Convolutional Neural Network (CNN)
- Implemented from scratch without external ML frameworks
- Architecture: Input (28x28) → Conv1 (8 filters, 3x3) → ReLU → MaxPool (2x2) → Conv2 (16 filters, 3x3) → ReLU → MaxPool (2x2) → Flatten → Dense (128) → ReLU → Output (10) → Softmax
- Uses convolution and max-pooling layers
- Batch processing support
- Adam optimizer for training
- Configurable learning rate and epochs

## 📊 MNIST Dataset Format

The application expects CSV files with the following format:
```
label,pixel0,pixel1,pixel2,...,pixel783
5,0,0,0,...,0
0,0,0,0,...,0
4,0,0,0,...,0
```

**Note:** The file must include a header row (first line shown above).

- First column: digit label (0-9)
- Remaining 784 columns: pixel values (0-255) for 28x28 image

## 💾 Model Storage

Trained models are saved in JSON format with the following structure:
- `ann200.json` - Simple ANN model
- `fullModel1000.json` - Deep neural network model
- `testmm.json` - Test model
- `*.json` - CNN models (saved via "Save To File 2" button)

Models contain serialized weights, biases, and network architecture.

## 🎯 Performance

The accuracy depends on:
- Training data size
- Number of epochs
- Algorithm choice
- Model architecture

Typical performance:
- KNN (K=3): ~95-97% accuracy
- Simple ANN: ~90-92% accuracy
- Deep Neural Network: ~96-98% accuracy (faster with GPU)
- CNN: ~97-99% accuracy (with sufficient training epochs)

## 🐛 Troubleshooting

### Common Issues

**Application won't start:**
- Ensure .NET 8.0 SDK is installed
- Verify all NuGet packages are restored
- Check that ILGPU packages are properly installed

**Can't load CSV file:**
- Check file format matches MNIST specification
- Ensure file has header row
- Verify pixel values are 0-255

**Training is slow:**
- Large datasets take time (be patient)
- Progress bars show training status
- Consider reducing training epochs for testing

**ML.NET models not working:**
- Verify ML.NET packages are installed
- Check that model files (.zip) exist
- Rebuild ML.NET models if necessary

**GPU acceleration not working:**
- Verify CUDA-compatible GPU is installed
- Check ILGPU packages are properly installed
- Application will automatically fall back to CPU if GPU is unavailable
- GPU status is displayed in the UI

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is available for educational purposes.

## 👤 Author

YAKOV-FRIDMAN

## 🙏 Acknowledgments

- MNIST Database: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- ML.NET Team for the machine learning framework
- The .NET and C# communities

## 📚 Additional Resources

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [ML.NET Documentation](https://docs.microsoft.com/en-us/dotnet/machine-learning/)
- [K-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network)
- [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [ILGPU Documentation](https://github.com/m4rs-mt/ILGPU)

---

**Note:** The project directory name "אלגוריתם שכן קרוב זיהוי מספרים בכתב יד" is Hebrew for "Nearest Neighbor Algorithm - Handwritten Digit Recognition"
