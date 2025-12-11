using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics; // Added for SIMD
using System.Text;
using System.Threading.Tasks;

namespace TestAi.Services
{
    public class DeepNeuralNetwork : IDisposable
    {
        private class NetworkModel
        {
            public List<double[][]> Weights { get; set; }
            public List<double[]> Biases { get; set; }
        }

        // Flattened weights for better memory locality and GPU compatibility
        // weights[layer][neuron * inputSize + inputIndex]
        private List<double[]> weights; 
        private List<double[]> biases;    
        private List<int> layerInputSizes; // Input size for each layer

        double learningRate = 0.001; // Reduced learning rate for stability
        public event Action<double> TrainingProgressChanged;

        // GPU Support
        private Context gpuContext;
        private Accelerator gpuAccelerator;
        private bool useGpu = false;

        // GPU Buffers
        private List<MemoryBuffer1D<double, Stride1D.Dense>> gpuWeights;
        private List<MemoryBuffer1D<double, Stride1D.Dense>> gpuBiases;
        
        // Reusable GPU Buffers for Mini-Batch
        private MemoryBuffer1D<double, Stride1D.Dense> gpuSharedInputs;
        private MemoryBuffer1D<double, Stride1D.Dense> gpuSharedDeltas;
        private int maxBatchSize = 128; // Default max batch size for allocation

        // Pre-compiled Kernels
        private Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, int, int, double> weightsKernel;
        private Action<Index1D, ArrayView1D<double, Stride1D.Dense>, ArrayView1D<double, Stride1D.Dense>, int, double> biasesKernel;

        public bool UseGpu
        {
            get { return useGpu; }
            set 
            { 
                useGpu = value;
                if (useGpu) InitializeGpu();
                else DisposeGpu();
            }
        }

        public DeepNeuralNetwork(int inputSize, int[] layerSizes)
        {
            var rnd = new Random();
            
            // Initialize lists
            weights = new List<double[]>();
            biases = new List<double[]>();
            layerInputSizes = new List<int>();

            int previousSize = inputSize; // The size of the input layer

            // Initialize weights for each layer dynamically
            foreach (int size in layerSizes)
            {
                // Track the input size for this layer
                layerInputSizes.Add(previousSize);

                double[] layerWeights = new double[size * previousSize];
                double[] layerBiases = new double[size];

                // He Initialization (optimized for ReLU)
                double stdDev = Math.Sqrt(2.0 / previousSize);

                for (int i = 0; i < size; i++)
                {
                    layerBiases[i] = 0.0; // Initialize biases to 0

                    for (int j = 0; j < previousSize; j++)
                    {
                        // Box-Muller transform for normal distribution
                        double u1 = 1.0 - rnd.NextDouble();
                        double u2 = 1.0 - rnd.NextDouble();
                        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                        
                        layerWeights[i * previousSize + j] = randStdNormal * stdDev;
                    }
                }
                weights.Add(layerWeights);
                biases.Add(layerBiases);
                previousSize = size; // Update the size for the next layer's input
            }
        }

        private void InitializeGpu()
        {
            if (gpuContext != null) return;

            gpuContext = Context.Create(builder => builder.Cuda().CPU());

            // Prefer Cuda (NVIDIA), fallback to CPU
            var cudaDevice = gpuContext.GetCudaDevices()[0];
            if (cudaDevice != null)
            {
                gpuAccelerator = cudaDevice.CreateAccelerator(gpuContext);
            }
            else
            {
                // Fallback or explicit CPU choice if no GPU found
                gpuAccelerator = gpuContext.GetCPUDevices()[0]?.CreateAccelerator(gpuContext);
            }

            if (gpuAccelerator != null)
            {
                gpuWeights = new List<MemoryBuffer1D<double, Stride1D.Dense>>();
                gpuBiases = new List<MemoryBuffer1D<double, Stride1D.Dense>>();
                
                // Allocate buffers for weights and biases
                for (int i = 0; i < weights.Count; i++)
                {
                    var wBuffer = gpuAccelerator.Allocate1D<double>(weights[i].Length);
                    wBuffer.CopyFromCPU(weights[i]);
                    gpuWeights.Add(wBuffer);

                    var bBuffer = gpuAccelerator.Allocate1D<double>(biases[i].Length);
                    bBuffer.CopyFromCPU(biases[i]);
                    gpuBiases.Add(bBuffer);
                }

                // Allocate shared buffers for batch inputs and deltas
                // Find maximum size needed
                int maxInputSize = layerInputSizes.Max();
                int maxOutputSize = biases.Max(b => b.Length);
                int maxLayerSize = Math.Max(maxInputSize, maxOutputSize);

                // Allocate for max batch size
                gpuSharedInputs = gpuAccelerator.Allocate1D<double>(maxBatchSize * maxLayerSize);
                gpuSharedDeltas = gpuAccelerator.Allocate1D<double>(maxBatchSize * maxLayerSize);

                // Compile Kernels Once
                weightsKernel = gpuAccelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView1D<double, Stride1D.Dense>,
                    ArrayView1D<double, Stride1D.Dense>,
                    ArrayView1D<double, Stride1D.Dense>,
                    int,
                    int,
                    double>(UpdateWeightsKernel);

                biasesKernel = gpuAccelerator.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView1D<double, Stride1D.Dense>,
                    ArrayView1D<double, Stride1D.Dense>,
                    int,
                    double>(UpdateBiasesKernel);
            }
        }

        private void DisposeGpu()
        {
            if (gpuWeights != null)
            {
                foreach (var buffer in gpuWeights) buffer.Dispose();
                gpuWeights.Clear();
                gpuWeights = null;
            }
            if (gpuBiases != null)
            {
                foreach (var buffer in gpuBiases) buffer.Dispose();
                gpuBiases.Clear();
                gpuBiases = null;
            }

            gpuSharedInputs?.Dispose();
            gpuSharedInputs = null;
            gpuSharedDeltas?.Dispose();
            gpuSharedDeltas = null;

            gpuAccelerator?.Dispose();
            gpuContext?.Dispose();
            gpuAccelerator = null;
            gpuContext = null;
        }

        public void Dispose()
        {
            DisposeGpu();
        }

        public static bool IsCudaAvailable()
        {
            using var context = Context.Create(builder => builder.Cuda());
            return context.GetCudaDevices().Count > 0;
        }
        // get name of the accelerator
        public static string GetAcceleratorName()
        {
            using var context = Context.Create(builder => builder.Cuda().CPU());
            var cudaDevices = context.GetCudaDevices();
            if (cudaDevices.Count > 0)
            {
                return cudaDevices[0].Name;
            }
            else
            {
                var cpuDevices = context.GetCPUDevices();
                if (cpuDevices.Count > 0)
                {
                    return cpuDevices[0].Name;
                }
            }
            return "No Accelerator Available";
        }

        public void Train(List<Tuple<double[], int>> trainingData, int batchSize = 32)
        {
            int epochs = 100; 
            var rnd = new Random();

            // Check if normalization is needed (heuristic: if any value > 1.0 in first few samples)
            bool shouldNormalize = trainingData.Take(Math.Min(trainingData.Count, 50))
                                               .Any(t => t.Item1.Any(v => v > 1.0));

            // Ensure GPU buffers are large enough if using GPU
            if (useGpu && batchSize > maxBatchSize)
            {
                maxBatchSize = batchSize;
                DisposeGpu();
                InitializeGpu();
            }

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Shuffle training data
                var shuffledData = trainingData.OrderBy(x => rnd.Next()).ToList();
                double totalLoss = 0;

                // Process in batches
                for (int i = 0; i < shuffledData.Count; i += batchSize)
                {
                    var batch = shuffledData.Skip(i).Take(batchSize).ToList();
                    int currentBatchSize = batch.Count;

                    // Containers for batch data
                    var batchLayerOutputs = new List<List<double[]>>(currentBatchSize);
                    var batchDeltas = new List<double[][]>(currentBatchSize);
                    var batchOriginalInputs = new List<double[]>(currentBatchSize);

                    // Forward and Backprop for each sample in batch
                    foreach (var (rawInput, expectedIndex) in batch)
                    {
                        double[] input = rawInput;
                        if (shouldNormalize)
                        {
                            input = new double[rawInput.Length];
                            for (int k = 0; k < rawInput.Length; k++) input[k] = rawInput[k] / 255.0;
                        }

                        // Forward propagation
                        var layerOutputs = ForwardPropagate(input);
                        
                        // Convert expected index to output array (One-Hot Encoding)
                        double[] expectedOutputs = new double[layerOutputs.Last().Length];
                        expectedOutputs[expectedIndex] = 1;

                        // Calculate Loss (optional, for monitoring)
                        totalLoss += CalculateCost(layerOutputs.Last(), expectedIndex);

                        // Backpropagation
                        var deltas = Backpropagate(expectedOutputs, layerOutputs, input);

                        batchLayerOutputs.Add(layerOutputs);
                        batchDeltas.Add(deltas);
                        batchOriginalInputs.Add(input);
                    }

                    // Update weights using the accumulated gradients from the batch
                    UpdateWeights(batchDeltas, batchLayerOutputs, batchOriginalInputs);
                }

                double avgLoss = totalLoss / shuffledData.Count;
              

                double trainingProgress = (double)(epoch + 1) / epochs * 100.0;
                TrainingProgressChanged?.Invoke(trainingProgress); // עכשיו 0–100

                // Console.WriteLine($"Epoch {epoch + 1}, Loss: {avgLoss}");
            }
        }

        // Kernels for GPU (Mini-Batch)
        public static void UpdateWeightsKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> weights,
            ArrayView1D<double, Stride1D.Dense> batchInputs,
            ArrayView1D<double, Stride1D.Dense> batchDeltas,
            int inputSize,
            int batchSize,
            double learningRate)
        {
            // index corresponds to a weight index: neuron * inputSize + inputIndex
            int neuron = index / inputSize;
            int inputIndex = index % inputSize;
            int neurons = batchDeltas.IntLength / batchSize; // Total neurons in this layer

            double gradientSum = 0;

            // Accumulate gradients over the batch
            for (int b = 0; b < batchSize; b++)
            {
                // batchDeltas is [batchSize * neurons]
                // batchInputs is [batchSize * inputSize]
                double delta = batchDeltas[b * neurons + neuron];
                double inputVal = batchInputs[b * inputSize + inputIndex];
                gradientSum += delta * inputVal;
            }

            // Average gradient update
            weights[index] -= learningRate * (gradientSum / batchSize);
        }

        public static void UpdateBiasesKernel(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> biases,
            ArrayView1D<double, Stride1D.Dense> batchDeltas,
            int batchSize,
            double learningRate)
        {
            int neuron = index;
            int neurons = biases.IntLength;
            double gradientSum = 0;

            // Accumulate gradients over the batch
            for (int b = 0; b < batchSize; b++)
            {
                gradientSum += batchDeltas[b * neurons + neuron];
            }

            biases[neuron] -= learningRate * (gradientSum / batchSize);
        }

        private void UpdateWeights(List<double[][]> batchDeltas, List<List<double[]>> batchLayerOutputs, List<double[]> batchOriginalInputs)
        {
            int currentBatchSize = batchDeltas.Count;
            if (currentBatchSize == 0) return;

            if (useGpu && gpuAccelerator != null)
            {
                for (int layer = 0; layer < weights.Count; layer++)
                {
                    int inputSize = layerInputSizes[layer];
                    int neurons = biases[layer].Length;

                    // 1. Flatten batch inputs and deltas for this layer
                    // We reuse the shared buffers, so we need to copy data to them
                    // Note: This copy is from CPU to GPU, but done once per batch per layer
                    
                    // Prepare arrays on CPU first (could be optimized further with pinned memory or direct copy if ILGPU supports it easily)
                    double[] flatInputs = new double[currentBatchSize * inputSize];
                    double[] flatDeltas = new double[currentBatchSize * neurons];

                    // Parallelize flattening
                    Parallel.For(0, currentBatchSize, b =>
                    {
                        double[] inputs = (layer == 0) ? batchOriginalInputs[b] : batchLayerOutputs[b][layer - 1];
                        double[] deltas = batchDeltas[b][layer];
                        
                        Array.Copy(inputs, 0, flatInputs, b * inputSize, inputSize);
                        Array.Copy(deltas, 0, flatDeltas, b * neurons, neurons);
                    });

                    // 2. Copy to GPU (using subviews of the shared buffers)
                    var inputsView = gpuSharedInputs.View.SubView(0, flatInputs.Length);
                    var deltasView = gpuSharedDeltas.View.SubView(0, flatDeltas.Length);
                    
                    inputsView.CopyFromCPU(flatInputs);
                    deltasView.CopyFromCPU(flatDeltas);

                    // 3. Launch Kernels
                    // Update Weights
                    weightsKernel((int)gpuWeights[layer].Length, gpuWeights[layer].View, inputsView, deltasView, inputSize, currentBatchSize, learningRate);

                    // Update Biases
                    biasesKernel((int)gpuBiases[layer].Length, gpuBiases[layer].View, deltasView, currentBatchSize, learningRate);

                    // 4. Synchronize and copy back to CPU
                    gpuAccelerator.Synchronize();
                    gpuWeights[layer].CopyToCPU(weights[layer]);
                    gpuBiases[layer].CopyToCPU(biases[layer]);
                }
                return;
            }

            // CPU Fallback with Mini-Batch
            for (int layer = 0; layer < weights.Count; layer++)
            {
                double[] layerWeights = weights[layer];
                double[] layerBiases = biases[layer];
                int inputSize = layerInputSizes[layer];
                
                // Use Parallel.For to utilize multi-core CPU
                Parallel.For(0, layerBiases.Length, neuron =>
                {
                    double biasGradientSum = 0;
                    
                    // Accumulate bias gradients
                    for (int b = 0; b < currentBatchSize; b++)
                    {
                        biasGradientSum += batchDeltas[b][layer][neuron];
                    }
                    
                    // Update Bias
                    layerBiases[neuron] -= learningRate * (biasGradientSum / currentBatchSize);

                    // Update Weights using SIMD
                    int weightOffset = neuron * inputSize;
                    int vectorSize = Vector<double>.Count;
                    
                    // Optimized accumulation of gradients
                    double[] weightGradients = new double[inputSize];

                    for (int b = 0; b < currentBatchSize; b++)
                    {
                        double delta = batchDeltas[b][layer][neuron];
                        double[] inputs = (layer == 0) ? batchOriginalInputs[b] : batchLayerOutputs[b][layer - 1];

                        int i = 0;
                        // Vectorized loop
                        for (; i <= inputSize - vectorSize; i += vectorSize)
                        {
                            var gVec = new Vector<double>(weightGradients, i);
                            var iVec = new Vector<double>(inputs, i);
                            gVec += iVec * delta;
                            gVec.CopyTo(weightGradients, i);
                        }
                        // Remaining elements
                        for (; i < inputSize; i++)
                        {
                            weightGradients[i] += delta * inputs[i];
                        }
                    }

                    // Apply weight updates
                    for (int i = 0; i < inputSize; i++)
                    {
                        layerWeights[weightOffset + i] -= learningRate * (weightGradients[i] / currentBatchSize);
                    }
                });
            }
        }


        private double[][] Backpropagate(double[] expectedOutputs, List<double[]> layerOutputs, double[] input)
        {
            int numLayers = weights.Count;
            double[][] deltas = new double[numLayers][];

            // 1. Calculate error for the output layer (Softmax + Cross-Entropy)
            // The derivative simplifies to: output - expected
            double[] outputLayerOutputs = layerOutputs.Last();
            double[] outputError = new double[outputLayerOutputs.Length];
            
            // Parallelize output error calculation
            Parallel.For(0, outputLayerOutputs.Length, i =>
            {
                outputError[i] = outputLayerOutputs[i] - expectedOutputs[i];
            });
            deltas[numLayers - 1] = outputError;

            // 2. Backpropagate the error to hidden layers
            for (int layer = numLayers - 2; layer >= 0; layer--)
            {
                double[] currentLayerOutputs = layerOutputs[layer];
                double[] nextLayerDeltas = deltas[layer + 1];
                double[] nextLayerWeights = weights[layer + 1];
                int nextLayerInputSize = layerInputSizes[layer + 1];
                
                double[] layerError = new double[currentLayerOutputs.Length];

                // Parallelize hidden layer error calculation
                Parallel.For(0, currentLayerOutputs.Length, i =>
                {
                    double error = 0;
                    // i is the neuron index in current layer
                    // It corresponds to the input index i in the next layer
                    for (int j = 0; j < nextLayerDeltas.Length; j++)
                    {
                        // Weight connecting current neuron i to next neuron j
                        // nextLayerWeights is flat: [j * nextLayerInputSize + i]
                        error += nextLayerDeltas[j] * nextLayerWeights[j * nextLayerInputSize + i];
                    }
                    
                    // Derivative of ReLU: 1 if output > 0, else 0
                    double derivative = currentLayerOutputs[i] > 0 ? 1.0 : 0.0;
                    layerError[i] = error * derivative;
                });
                deltas[layer] = layerError;
            }

            return deltas;
        }


        private double CalculateCost(double[] output, int expectedOutput)
        {
            // Cross-Entropy Loss
            double epsilon = 1e-15;
            double predictedProb = Math.Max(epsilon, output[expectedOutput]); // Avoid log(0)
            return -Math.Log(predictedProb);
        }

        private List<double[]> ForwardPropagate(double[] inputs)
        {
            List<double[]> layerOutputs = new List<double[]>();
            double[] currentInputs = inputs;

            for (int layer = 0; layer < weights.Count; layer++)
            {
                double[] layerWeights = weights[layer];
                double[] layerBiases = biases[layer];
                int inputSize = layerInputSizes[layer];

                double[] newActivations = new double[layerBiases.Length];
                
                // Calculate Z (linear combination)
                double[] Z = new double[layerBiases.Length];

                // Parallelize neuron activation calculation
                Parallel.For(0, layerBiases.Length, i =>
                {
                    double activation = layerBiases[i];
                    int weightOffset = i * inputSize;
                    
                    // SIMD Dot Product
                    int vectorSize = Vector<double>.Count;
                    var accVector = Vector<double>.Zero;
                    int j = 0;

                    // Process in chunks
                    for (; j <= inputSize - vectorSize; j += vectorSize)
                    {
                        var wVec = new Vector<double>(layerWeights, weightOffset + j);
                        var inVec = new Vector<double>(currentInputs, j);
                        accVector += wVec * inVec;
                    }

                    // Sum up the vector results
                    activation += Vector.Dot(accVector, Vector<double>.One);

                    // Process remaining elements
                    for (; j < inputSize; j++)
                    {
                        activation += layerWeights[weightOffset + j] * currentInputs[j];
                    }

                    Z[i] = activation;
                });

                // Apply Activation Function
                if (layer == weights.Count - 1)
                {
                    // Softmax for Output Layer
                    double maxZ = Z.Max(); // For numerical stability
                    double sumExp = 0;
                    // Softmax requires sum, so we calculate exponentials first
                    for (int i = 0; i < Z.Length; i++)
                    {
                        newActivations[i] = Math.Exp(Z[i] - maxZ);
                        sumExp += newActivations[i];
                    }
                    for (int i = 0; i < Z.Length; i++)
                    {
                        newActivations[i] /= sumExp;
                    }
                }
                else
                {
                    // ReLU for Hidden Layers - can be parallelized
                    Parallel.For(0, Z.Length, i =>
                    {
                        newActivations[i] = Math.Max(0, Z[i]);
                    });
                }

                layerOutputs.Add(newActivations);
                currentInputs = newActivations; // Update inputs for the next layer
            }

            return layerOutputs; // Return all layers outputs
        }


        public int Classify(double[] inputData)
        {
            double[] input = inputData;
            // Auto-normalize if needed
            if (inputData.Any(x => x > 1.0))
            {
                input = new double[inputData.Length];
                for (int k = 0; k < inputData.Length; k++) input[k] = inputData[k] / 255.0;
            }

            var outputs = ForwardPropagate(input).Last();
            
            // Find index of max value
            int maxIndex = 0;
            double maxValue = outputs[0];
            for (int i = 1; i < outputs.Length; i++)
            {
                if (outputs[i] > maxValue)
                {
                    maxValue = outputs[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        public async Task SaveModel(string filePath)
        {
            // Convert flat structure back to jagged for serialization compatibility
            var jaggedWeights = new List<double[][]>();
            for (int i = 0; i < weights.Count; i++)
            {
                int neurons = biases[i].Length;
                int inputSize = layerInputSizes[i];
                double[][] layer = new double[neurons][];
                for (int n = 0; n < neurons; n++)
                {
                    layer[n] = new double[inputSize];
                    Array.Copy(weights[i], n * inputSize, layer[n], 0, inputSize);
                }
                jaggedWeights.Add(layer);
            }

            var model = new NetworkModel { Weights = jaggedWeights, Biases = biases };
            var json = JsonConvert.SerializeObject(model);
            await File.WriteAllTextAsync(filePath, json);
        }

        public async Task LoadModel(string filePath)
        {
            if (!File.Exists(filePath)) return;

            var json = await File.ReadAllTextAsync(filePath);
            try 
            {
                var model = JsonConvert.DeserializeObject<NetworkModel>(json);
                if (model != null && model.Weights != null && model.Biases != null)
                {
                    // Convert jagged to flat
                    weights = new List<double[]>();
                    biases = model.Biases;
                    layerInputSizes = new List<int>();

                    foreach (var layer in model.Weights)
                    {
                        int neurons = layer.Length;
                        int inputSize = layer[0].Length;
                        layerInputSizes.Add(inputSize);

                        double[] flatLayer = new double[neurons * inputSize];
                        for (int n = 0; n < neurons; n++)
                        {
                            Array.Copy(layer[n], 0, flatLayer, n * inputSize, inputSize);
                        }
                        weights.Add(flatLayer);
                    }
                    
                    // Re-initialize GPU if needed
                    if (useGpu)
                    {
                        DisposeGpu();
                        InitializeGpu();
                    }
                    return;
                }
            }
            catch {}

            // Fallback logic for old format if needed...
        }
    }

    public class Neuron
    {
        public double[] Weights { get; set; }

        public double CalculateOutput(double[] inputs)
        {
            double output = 0;
            for (int i = 0; i < Weights.Length; i++)
            {
                output += Weights[i] * inputs[i];
            }
            // Assuming the use of a sigmoid activation function
            return 1 / (1 + Math.Exp(-output));
        }
    }

    public class Layer
    {
        public Neuron[] Neurons { get; set; }

        public double[] ForwardPropagate(double[] inputs)
        {
            double[] outputs = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                outputs[i] = Neurons[i].CalculateOutput(inputs);
            }
            return outputs;
        }
    }
}
