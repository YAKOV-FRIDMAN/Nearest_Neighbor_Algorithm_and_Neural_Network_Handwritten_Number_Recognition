using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using System.Text.Json;

namespace TestAi.Services
{
    /// <summary>
    /// A Convolutional Neural Network implemented from scratch.
    /// Architecture:
    /// Input (28x28) -> Conv1 (8x3x3) -> ReLU -> MaxPool (2x2) 
    /// -> Conv2 (16x3x3) -> ReLU -> MaxPool (2x2) 
    /// -> Flatten -> Dense (128) -> ReLU -> Output (10) -> Softmax
    /// </summary>
    public class ConvolutionalNeuralNetwork
    {
        // ===========================
        // 1. HYPERPARAMETERS & CONFIG
        // ===========================
        private double learningRate = 0.01;
        private const int InputDepth = 1;
        private const int InputRows = 28;
        private const int InputCols = 28;

        // Layer 1: Conv 8 filters, 3x3
        private const int C1Filters = 8;
        private const int C1Kernel = 3;

        // Layer 2: Conv 16 filters, 3x3
        private const int C2Filters = 16;
        private const int C2Kernel = 3;

        // Dense Layers
        private const int FcNeurons = 128;
        private const int OutputNeurons = 10;

        // ===========================
        // 2. WEIGHTS & BIASES
        // ===========================
        // 4D Arrays for Filters: [FilterIndex][Depth][Row][Col]
        private double[][][][] c1Weights;
        private double[] c1Biases;

        private double[][][][] c2Weights;
        private double[] c2Biases;

        // 2D Arrays for Dense Weights: [NeuronIndex][InputIndex]
        private double[][] fcWeights;
        private double[] fcBiases;

        private double[][] outWeights;
        private double[] outBiases;

        private Random rnd = new Random();

        public event Action<double> TrainingProgressChanged;

        public ConvolutionalNeuralNetwork()
        {
            InitializeWeights();
        }

        // ===========================
        // 3. INITIALIZATION
        // ===========================
        private void InitializeWeights()
        {
            // He Initialization for ReLU layers: std = sqrt(2.0 / fan_in)

            // Conv1: Input depth 1, Kernel 3x3 -> FanIn = 9
            c1Weights = InitFilters(C1Filters, InputDepth, C1Kernel, 9);
            c1Biases = new double[C1Filters];

            // Conv2: Input depth 8 (from C1), Kernel 3x3 -> FanIn = 8*3*3 = 72
            c2Weights = InitFilters(C2Filters, C1Filters, C2Kernel, 72);
            c2Biases = new double[C2Filters];

            // Calculate Flattened Size
            // 28x28 -> Conv1(valid) -> 26x26 -> Pool(2x2) -> 13x13
            // 13x13 -> Conv2(valid) -> 11x11 -> Pool(2x2) -> 5x5
            // Flattened: 16 filters * 5 * 5 = 400
            int flattenedSize = C2Filters * 5 * 5;

            // FC Layer: FanIn = 400
            fcWeights = InitMatrix(FcNeurons, flattenedSize, flattenedSize);
            fcBiases = new double[FcNeurons];

            // Output Layer: FanIn = 128
            outWeights = InitMatrix(OutputNeurons, FcNeurons, FcNeurons);
            outBiases = new double[OutputNeurons];
        }

        private double[][][][] InitFilters(int count, int depth, int size, int fanIn)
        {
            var filters = new double[count][][][];
            double stdDev = Math.Sqrt(2.0 / fanIn);

            for (int f = 0; f < count; f++)
            {
                filters[f] = new double[depth][][];
                for (int d = 0; d < depth; d++)
                {
                    filters[f][d] = new double[size][];
                    for (int r = 0; r < size; r++)
                    {
                        filters[f][d][r] = new double[size];
                        for (int c = 0; c < size; c++)
                        {
                            filters[f][d][r][c] = GetRandomGaussian() * stdDev;
                        }
                    }
                }
            }
            return filters;
        }

        private double[][] InitMatrix(int rows, int cols, int fanIn)
        {
            var matrix = new double[rows][];
            double stdDev = Math.Sqrt(2.0 / fanIn);

            for (int i = 0; i < rows; i++)
            {
                matrix[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    matrix[i][j] = GetRandomGaussian() * stdDev;
                }
            }
            return matrix;
        }

        private double GetRandomGaussian()
        {
            double u1 = 1.0 - rnd.NextDouble();
            double u2 = 1.0 - rnd.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }

        // ===========================
        // 4. PUBLIC API
        // ===========================

        public async Task SaveModel(string filePath)
        {
            var modelData = new
            {
                C1Weights = c1Weights,
                C1Biases = c1Biases,
                C2Weights = c2Weights,
                C2Biases = c2Biases,
                FcWeights = fcWeights,
                FcBiases = fcBiases,
                OutWeights = outWeights,
                OutBiases = outBiases
            };

            var options = new JsonSerializerOptions { WriteIndented = true }; 
            using (var stream = File.Create(filePath))
            {
                await JsonSerializer.SerializeAsync(stream, modelData, options);
            }
        }

        public async Task LoadModel(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("Model file not found", filePath);

            using (var stream = File.OpenRead(filePath))
            { 
                var modelData = await JsonSerializer.DeserializeAsync<ModelData>(stream);
                if (modelData != null)
                {
                    c1Weights = modelData.C1Weights;
                    c1Biases = modelData.C1Biases;
                    c2Weights = modelData.C2Weights;
                    c2Biases = modelData.C2Biases;
                    fcWeights = modelData.FcWeights;
                    fcBiases = modelData.FcBiases;
                    outWeights = modelData.OutWeights;
                    outBiases = modelData.OutBiases;
                }
            }
        }

        private class ModelData
        {
            public double[][][][] C1Weights { get; set; }
            public double[] C1Biases { get; set; }
            public double[][][][] C2Weights { get; set; }
            public double[] C2Biases { get; set; }
            public double[][] FcWeights { get; set; }
            public double[] FcBiases { get; set; }
            public double[][] OutWeights { get; set; }
            public double[] OutBiases { get; set; }
        }

        public void Train(List<Tuple<double[], int>> trainingData, int epochs, int batchSize)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Shuffle
                var shuffled = trainingData.OrderBy(x => rnd.Next()).ToList();
                double totalLoss = 0;
                int correct = 0;

                for (int i = 0; i < shuffled.Count; i += batchSize)
                {
                    var batch = shuffled.Skip(i).Take(batchSize).ToList();
                    ProcessBatch(batch);

                    // Optional: Calculate loss/accuracy for monitoring (on last item of batch)
                    // This is skipped for performance in pure training loop

                    double progress = ((double)epoch / epochs * 100.0) + ((double)i / shuffled.Count * (100.0 / epochs));
                    TrainingProgressChanged?.Invoke(progress);
                }

                // Simple progress log
                // Console.WriteLine($"Epoch {epoch + 1} complete.");
                TrainingProgressChanged?.Invoke((double)(epoch + 1) / epochs * 100.0);
            }
        }

        public int Classify(double[] image)
        {
            var inputVol = ReshapeInput(image);
            var cache = Forward(inputVol);

            // Argmax
            int maxIndex = 0;
            double maxVal = cache.FinalOutput[0];
            for (int i = 1; i < cache.FinalOutput.Length; i++)
            {
                if (cache.FinalOutput[i] > maxVal)
                {
                    maxVal = cache.FinalOutput[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        // ===========================
        // 5. CORE LOGIC (Forward/Back)
        // ===========================

        private void ProcessBatch(List<Tuple<double[], int>> batch)
        {
            // Accumulators for gradients
            var dc1W = CreateZeroFilters(c1Weights);
            var dc1B = new double[c1Biases.Length];
            var dc2W = CreateZeroFilters(c2Weights);
            var dc2B = new double[c2Biases.Length];
            var dfcW = CreateZeroMatrix(fcWeights);
            var dfcB = new double[fcBiases.Length];
            var doutW = CreateZeroMatrix(outWeights);
            var doutB = new double[outBiases.Length];

            // Parallel processing of batch is possible, but we use sequential for safety/simplicity here
            foreach (var sample in batch)
            {
                var inputVol = ReshapeInput(sample.Item1);
                int label = sample.Item2;

                // 1. Forward
                var cache = Forward(inputVol);

                // 2. Backward
                var grads = Backward(cache, label, inputVol);

                // 3. Accumulate
                AccumulateFilters(dc1W, grads.dC1W);
                AccumulateVector(dc1B, grads.dC1B);
                AccumulateFilters(dc2W, grads.dC2W);
                AccumulateVector(dc2B, grads.dC2B);
                AccumulateMatrix(dfcW, grads.dFcW);
                AccumulateVector(dfcB, grads.dFcB);
                AccumulateMatrix(doutW, grads.dOutW);
                AccumulateVector(doutB, grads.dOutB);
            }

            // 4. Update Weights (SGD)
            double batchFactor = learningRate / batch.Count;
            ApplyUpdates(c1Weights, dc1W, batchFactor);
            ApplyUpdates(c1Biases, dc1B, batchFactor);
            ApplyUpdates(c2Weights, dc2W, batchFactor);
            ApplyUpdates(c2Biases, dc2B, batchFactor);
            ApplyUpdates(fcWeights, dfcW, batchFactor);
            ApplyUpdates(fcBiases, dfcB, batchFactor);
            ApplyUpdates(outWeights, doutW, batchFactor);
            ApplyUpdates(outBiases, doutB, batchFactor);
        }

        // Container for intermediate values needed for backprop
        private class ForwardCache
        {
            public double[][][] C1Out; // Post-ReLU
            public double[][][] P1Out;
            public int[][][] P1Indices; // Max indices
            public double[][][] C2Out; // Post-ReLU
            public double[][][] P2Out;
            public int[][][] P2Indices;
            public double[] Flattened;
            public double[] FcOut; // Post-ReLU
            public double[] FinalOutput; // Softmax
        }

        private ForwardCache Forward(double[][][] input)
        {
            var c = new ForwardCache();

            // Conv1 -> ReLU
            var z1 = Convolve(input, c1Weights, c1Biases, 1);
            c.C1Out = Relu3D(z1);

            // MaxPool1
            var (p1, idx1) = MaxPool(c.C1Out, 2);
            c.P1Out = p1;
            c.P1Indices = idx1;

            // Conv2 -> ReLU
            var z2 = Convolve(c.P1Out, c2Weights, c2Biases, 1);
            c.C2Out = Relu3D(z2);

            // MaxPool2
            var (p2, idx2) = MaxPool(c.C2Out, 2);
            c.P2Out = p2;
            c.P2Indices = idx2;

            // Flatten
            c.Flattened = Flatten(c.P2Out);

            // FC -> ReLU
            var zFc = DenseForward(c.Flattened, fcWeights, fcBiases);
            c.FcOut = Relu1D(zFc);

            // Output -> Softmax
            var zOut = DenseForward(c.FcOut, outWeights, outBiases);
            c.FinalOutput = Softmax(zOut);

            return c;
        }

        private class Gradients
        {
            public double[][][][] dC1W; public double[] dC1B;
            public double[][][][] dC2W; public double[] dC2B;
            public double[][] dFcW; public double[] dFcB;
            public double[][] dOutW; public double[] dOutB;
        }

        private Gradients Backward(ForwardCache c, int label, double[][][] input)
        {
            var g = new Gradients();

            // 1. Output Layer Gradient (Softmax + CrossEntropy)
            // dL/dz = predicted - target
            double[] dOutZ = new double[OutputNeurons];
            Array.Copy(c.FinalOutput, dOutZ, OutputNeurons);
            dOutZ[label] -= 1.0;

            // Gradients for Output Weights/Biases
            g.dOutW = OuterProduct(dOutZ, c.FcOut);
            g.dOutB = dOutZ;

            // 2. Backprop to FC Layer
            // dL/d(FcOut) = W_out^T * dOutZ
            double[] dFcOut = MatMulTransposed(outWeights, dOutZ);

            // ReLU Derivative for FC
            double[] dFcZ = new double[FcNeurons];
            for (int i = 0; i < FcNeurons; i++)
                dFcZ[i] = (c.FcOut[i] > 0) ? dFcOut[i] : 0;

            g.dFcW = OuterProduct(dFcZ, c.Flattened);
            g.dFcB = dFcZ;

            // 3. Backprop to Flattened Layer
            double[] dFlattened = MatMulTransposed(fcWeights, dFcZ);

            // Unflatten
            double[][][] dP2 = Unflatten(dFlattened, c.P2Out.Length, c.P2Out[0].Length, c.P2Out[0][0].Length);

            // 4. Backprop MaxPool2
            double[][][] dC2Out = MaxPoolBackward(dP2, c.P2Indices, c.C2Out.Length, c.C2Out[0].Length, c.C2Out[0][0].Length);

            // ReLU Derivative for Conv2
            ApplyReluDerivative(dC2Out, c.C2Out);

            // 5. Backprop Conv2
            // dL/dFilter = Conv(Input, dOutput)
            // dL/dInput = FullConv(dOutput, RotatedFilter)
            g.dC2W = ConvolveBackpropFilters(c.P1Out, dC2Out, C2Kernel);
            g.dC2B = SumChannels(dC2Out);
            double[][][] dP1 = ConvolveBackpropInput(dC2Out, c2Weights);

            // 6. Backprop MaxPool1
            double[][][] dC1Out = MaxPoolBackward(dP1, c.P1Indices, c.C1Out.Length, c.C1Out[0].Length, c.C1Out[0][0].Length);

            // ReLU Derivative for Conv1
            ApplyReluDerivative(dC1Out, c.C1Out);

            // 7. Backprop Conv1
            g.dC1W = ConvolveBackpropFilters(input, dC1Out, C1Kernel);
            g.dC1B = SumChannels(dC1Out);
            // No need to compute dInput for the image itself

            return g;
        }

        // ===========================
        // 6. HELPER OPERATIONS
        // ===========================

        // --- Convolution ---
        private double[][][] Convolve(double[][][] input, double[][][][] filters, double[] biases, int stride)
        {
            int numFilters = filters.Length;
            int depth = input.Length;
            int inH = input[0].Length;
            int inW = input[0][0].Length;
            int kSize = filters[0][0].Length;

            int outH = (inH - kSize) / stride + 1;
            int outW = (inW - kSize) / stride + 1;

            var output = new double[numFilters][][];

            // Parallelize over filters
            Parallel.For(0, numFilters, f =>
            {
                output[f] = new double[outH][];
                for (int i = 0; i < outH; i++)
                {
                    output[f][i] = new double[outW];
                    for (int j = 0; j < outW; j++)
                    {
                        double sum = biases[f];
                        // Convolve over depth
                        for (int d = 0; d < depth; d++)
                        {
                            for (int ki = 0; ki < kSize; ki++)
                            {
                                for (int kj = 0; kj < kSize; kj++)
                                {
                                    sum += input[d][i * stride + ki][j * stride + kj] * filters[f][d][ki][kj];
                                }
                            }
                        }
                        output[f][i][j] = sum;
                    }
                }
            });

            return output;
        }

        // --- Pooling ---
        private (double[][][], int[][][]) MaxPool(double[][][] input, int size)
        {
            int depth = input.Length;
            int inH = input[0].Length;
            int inW = input[0][0].Length;
            int outH = inH / size;
            int outW = inW / size;

            var output = new double[depth][][];
            var indices = new int[depth][][]; // Stores flat index of max value

            Parallel.For(0, depth, d =>
            {
                output[d] = new double[outH][];
                indices[d] = new int[outH][];
                for (int i = 0; i < outH; i++)
                {
                    output[d][i] = new double[outW];
                    indices[d][i] = new int[outW];
                    for (int j = 0; j < outW; j++)
                    {
                        double maxVal = double.MinValue;
                        int maxIdx = -1;

                        for (int ki = 0; ki < size; ki++)
                        {
                            for (int kj = 0; kj < size; kj++)
                            {
                                int r = i * size + ki;
                                int c = j * size + kj;
                                double val = input[d][r][c];
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                    maxIdx = r * inW + c; // Store flat index
                                }
                            }
                        }
                        output[d][i][j] = maxVal;
                        indices[d][i][j] = maxIdx;
                    }
                }
            });

            return (output, indices);
        }

        // --- Activations ---
        private double[][][] Relu3D(double[][][] input)
        {
            int d = input.Length;
            int h = input[0].Length;
            int w = input[0][0].Length;
            var output = new double[d][][];

            Parallel.For(0, d, i =>
            {
                output[i] = new double[h][];
                for (int r = 0; r < h; r++)
                {
                    output[i][r] = new double[w];
                    for (int c = 0; c < w; c++)
                    {
                        output[i][r][c] = Math.Max(0, input[i][r][c]);
                    }
                }
            });
            return output;
        }

        private double[] Relu1D(double[] input)
        {
            return input.Select(x => Math.Max(0, x)).ToArray();
        }

        private double[] Softmax(double[] input)
        {
            double max = input.Max();
            double sum = 0;
            double[] exps = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                exps[i] = Math.Exp(input[i] - max);
                sum += exps[i];
            }
            for (int i = 0; i < input.Length; i++)
            {
                exps[i] /= sum;
            }
            return exps;
        }

        // --- Dense Operations ---
        private double[] DenseForward(double[] input, double[][] weights, double[] biases)
        {
            int neurons = weights.Length;
            int inputs = input.Length;
            double[] output = new double[neurons];

            Parallel.For(0, neurons, i =>
            {
                double sum = biases[i];
                for (int j = 0; j < inputs; j++)
                {
                    sum += weights[i][j] * input[j];
                }
                output[i] = sum;
            });
            return output;
        }

        private double[] Flatten(double[][][] input)
        {
            int d = input.Length;
            int h = input[0].Length;
            int w = input[0][0].Length;
            double[] flat = new double[d * h * w];
            int idx = 0;
            for (int i = 0; i < d; i++)
                for (int r = 0; r < h; r++)
                    for (int c = 0; c < w; c++)
                        flat[idx++] = input[i][r][c];
            return flat;
        }

        private double[][][] Unflatten(double[] flat, int d, int h, int w)
        {
            var vol = new double[d][][];
            int idx = 0;
            for (int i = 0; i < d; i++)
            {
                vol[i] = new double[h][];
                for (int r = 0; r < h; r++)
                {
                    vol[i][r] = new double[w];
                    for (int c = 0; c < w; c++)
                    {
                        vol[i][r][c] = flat[idx++];
                    }
                }
            }
            return vol;
        }

        // --- Backprop Helpers ---

        private double[][][] MaxPoolBackward(double[][][] dOut, int[][][] indices, int inD, int inH, int inW)
        {
            var dIn = new double[inD][][];
            // Initialize dIn with zeros
            for (int i = 0; i < inD; i++)
            {
                dIn[i] = new double[inH][];
                for (int r = 0; r < inH; r++) dIn[i][r] = new double[inW];
            }

            int outH = dOut[0].Length;
            int outW = dOut[0][0].Length;

            // Route gradients
            for (int d = 0; d < inD; d++)
            {
                for (int r = 0; r < outH; r++)
                {
                    for (int c = 0; c < outW; c++)
                    {
                        int flatIdx = indices[d][r][c];
                        int inR = flatIdx / inW;
                        int inC = flatIdx % inW;
                        dIn[d][inR][inC] += dOut[d][r][c];
                    }
                }
            }
            return dIn;
        }

        private void ApplyReluDerivative(double[][][] gradients, double[][][] activations)
        {
            int d = gradients.Length;
            int h = gradients[0].Length;
            int w = gradients[0][0].Length;

            Parallel.For(0, d, i =>
            {
                for (int r = 0; r < h; r++)
                {
                    for (int c = 0; c < w; c++)
                    {
                        if (activations[i][r][c] <= 0)
                            gradients[i][r][c] = 0;
                    }
                }
            });
        }

        private double[][][][] ConvolveBackpropFilters(double[][][] input, double[][][] dOut, int kSize)
        {
            // dL/dW = Conv(Input, dOut)
            // Input: [Depth][InH][InW]
            // dOut: [NumFilters][OutH][OutW]
            // Result: [NumFilters][Depth][K][K]

            int numFilters = dOut.Length;
            int depth = input.Length;
            var dW = new double[numFilters][][][];

            Parallel.For(0, numFilters, f =>
            {
                dW[f] = new double[depth][][];
                for (int d = 0; d < depth; d++)
                {
                    dW[f][d] = new double[kSize][];
                    for (int ki = 0; ki < kSize; ki++)
                    {
                        dW[f][d][ki] = new double[kSize];
                        for (int kj = 0; kj < kSize; kj++)
                        {
                            // Cross-correlation between input slice and dOut slice
                            double sum = 0;
                            int outH = dOut[f].Length;
                            int outW = dOut[f][0].Length;

                            for (int i = 0; i < outH; i++)
                            {
                                for (int j = 0; j < outW; j++)
                                {
                                    sum += input[d][i + ki][j + kj] * dOut[f][i][j];
                                }
                            }
                            dW[f][d][ki][kj] = sum;
                        }
                    }
                }
            });
            return dW;
        }

        private double[][][] ConvolveBackpropInput(double[][][] dOut, double[][][][] filters)
        {
            // dL/dInput = Full Convolution of dOut (padded) with Rotated Filters
            // This is complex to implement efficiently. 
            // Simplified: Iterate over input pixels and sum contributions from dOut * weights.

            int numFilters = filters.Length;
            int depth = filters[0].Length;
            int kSize = filters[0][0].Length;
            int outH = dOut[0].Length;
            int outW = dOut[0][0].Length;

            // Input size was (OutH + K - 1)
            int inH = outH + kSize - 1;
            int inW = outW + kSize - 1;

            var dIn = new double[depth][][];
            for (int d = 0; d < depth; d++)
            {
                dIn[d] = new double[inH][];
                for (int r = 0; r < inH; r++) dIn[d][r] = new double[inW];
            }

            // Accumulate gradients
            // For every position in dOut(f, i, j), it was computed using Input(d, i+ki, j+kj) * W(f, d, ki, kj)
            // So dInput(d, i+ki, j+kj) += dOut(f, i, j) * W(f, d, ki, kj)

            // This loop order is safe for accumulation if not parallelized on d or r/c of dIn
            // We parallelize on filters to avoid race conditions on dIn? No, multiple filters contribute to same dIn.
            // We must be careful. Serial is safest here for "from scratch".

            for (int f = 0; f < numFilters; f++)
            {
                for (int i = 0; i < outH; i++)
                {
                    for (int j = 0; j < outW; j++)
                    {
                        double grad = dOut[f][i][j];
                        if (grad == 0) continue;

                        for (int d = 0; d < depth; d++)
                        {
                            for (int ki = 0; ki < kSize; ki++)
                            {
                                for (int kj = 0; kj < kSize; kj++)
                                {
                                    dIn[d][i + ki][j + kj] += grad * filters[f][d][ki][kj];
                                }
                            }
                        }
                    }
                }
            }

            return dIn;
        }

        private double[] SumChannels(double[][][] vol)
        {
            int d = vol.Length;
            double[] sums = new double[d];
            for (int i = 0; i < d; i++)
            {
                double sum = 0;
                int h = vol[i].Length;
                int w = vol[i][0].Length;
                for (int r = 0; r < h; r++)
                    for (int c = 0; c < w; c++)
                        sum += vol[i][r][c];
                sums[i] = sum;
            }
            return sums;
        }

        private double[][] OuterProduct(double[] a, double[] b)
        {
            int rows = a.Length;
            int cols = b.Length;
            var mat = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                mat[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    mat[i][j] = a[i] * b[j];
                }
            }
            return mat;
        }

        private double[] MatMulTransposed(double[][] W, double[] v)
        {
            // v is column vector. Result = W^T * v
            int rows = W.Length;
            int cols = W[0].Length;
            double[] res = new double[cols];

            for (int i = 0; i < rows; i++)
            {
                double val = v[i];
                if (val == 0) continue;
                for (int j = 0; j < cols; j++)
                {
                    res[j] += W[i][j] * val;
                }
            }
            return res;
        }

        // --- Utils ---
        private double[][][] ReshapeInput(double[] flat)
        {
            var vol = new double[1][][];
            vol[0] = new double[InputRows][];
            for (int i = 0; i < InputRows; i++)
            {
                vol[0][i] = new double[InputCols];
                for (int j = 0; j < InputCols; j++)
                {
                    vol[0][i][j] = flat[i * InputCols + j] / 255.0;
                }
            }
            return vol;
        }

        private double[][][][] CreateZeroFilters(double[][][][] template)
        {
            int f = template.Length;
            int d = template[0].Length;
            int k = template[0][0].Length;
            var z = new double[f][][][];
            for (int i = 0; i < f; i++)
            {
                z[i] = new double[d][][];
                for (int j = 0; j < d; j++)
                {
                    z[i][j] = new double[k][];
                    for (int r = 0; r < k; r++) z[i][j][r] = new double[k];
                }
            }
            return z;
        }

        private double[][] CreateZeroMatrix(double[][] template)
        {
            int r = template.Length;
            int c = template[0].Length;
            var z = new double[r][];
            for (int i = 0; i < r; i++) z[i] = new double[c];
            return z;
        }

        private void AccumulateFilters(double[][][][] acc, double[][][][] grads)
        {
            for (int f = 0; f < acc.Length; f++)
                for (int d = 0; d < acc[0].Length; d++)
                    for (int r = 0; r < acc[0][0].Length; r++)
                        for (int c = 0; c < acc[0][0][0].Length; c++)
                            acc[f][d][r][c] += grads[f][d][r][c];
        }

        private void AccumulateMatrix(double[][] acc, double[][] grads)
        {
            for (int i = 0; i < acc.Length; i++)
                for (int j = 0; j < acc[0].Length; j++)
                    acc[i][j] += grads[i][j];
        }

        private void AccumulateVector(double[] acc, double[] grads)
        {
            for (int i = 0; i < acc.Length; i++) acc[i] += grads[i];
        }

        private void ApplyUpdates(double[][][][] weights, double[][][][] grads, double factor)
        {
            for (int f = 0; f < weights.Length; f++)
                for (int d = 0; d < weights[0].Length; d++)
                    for (int r = 0; r < weights[0][0].Length; r++)
                        for (int c = 0; c < weights[0][0][0].Length; c++)
                            weights[f][d][r][c] -= grads[f][d][r][c] * factor;
        }

        private void ApplyUpdates(double[][] weights, double[][] grads, double factor)
        {
            for (int i = 0; i < weights.Length; i++)
                for (int j = 0; j < weights[0].Length; j++)
                    weights[i][j] -= grads[i][j] * factor;
        }

        private void ApplyUpdates(double[] biases, double[] grads, double factor)
        {
            for (int i = 0; i < biases.Length; i++)
                biases[i] -= grads[i] * factor;
        }
    }
}