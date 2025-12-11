using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestAi.Services
{
    public class NeuralNetwork
    {
        private double[][][] weights;  // Weights for each neuron in each layer
        private double[][] weightsTest;
        public event Action<double> TrainingProgressChanged;
        DeepNeuralNetwork deepNeuralNetwor;
        public NeuralNetwork()
        {
            // Updated architecture:
            // Input: 784 (28x28 pixels)
            // Hidden Layer 1: 128 neurons (good capacity for features)
            // Hidden Layer 2: 64 neurons (refining features)
            // Output Layer: 10 neurons (digits 0-9)
            deepNeuralNetwor = new DeepNeuralNetwork(784, new int[] { 128, 64, 10 });
            deepNeuralNetwor.TrainingProgressChanged += (e) => { TrainingProgressChanged?.Invoke(e); }; 
        }

        public bool UseGpu
        {
            get { return deepNeuralNetwor.UseGpu; }
            set { deepNeuralNetwor.UseGpu = value; }
        }

        public bool IsCudaAvailable => DeepNeuralNetwork.IsCudaAvailable();

        public void Train(List<Tuple<double[], int>> trainingData)
        {
            deepNeuralNetwor.Train(trainingData);
            return;
            // deepNeuralNetwork.Train(trainingData);  
            // Initialize weights randomly
            var rnd = new Random();
            int numLayers = 3; // Example: 3 layers
            int neuronsPerLayer = 10; // Example: 10 neurons per layer
            int numFeatures = trainingData[0].Item1.Length;

            weights = new double[numLayers][][];
            for (int layer = 0; layer < numLayers; layer++)
            {
                weights[layer] = new double[neuronsPerLayer][];
                for (int neuron = 0; neuron < neuronsPerLayer; neuron++)
                {
                    weights[layer][neuron] = new double[numFeatures];
                    for (int weightIndex = 0; weightIndex < numFeatures; weightIndex++)
                        weights[layer][neuron][weightIndex] = rnd.NextDouble();
                }
            }

            int totalEpochs = 1000;
            for (int epoch = 0; epoch < totalEpochs; epoch++)
            {
                foreach (var data in trainingData)
                {
                    int label = data.Item2;
                    double[] features = data.Item1;

                    for (int layer = 0; layer < numLayers; layer++)
                    {
                        for (int neuron = 0; neuron < neuronsPerLayer; neuron++)
                        {
                            double output = 0;
                            for (int j = 0; j < features.Length; j++)
                                output += weights[layer][neuron][j] * features[j];

                            // Simple Perceptron learning rule for now (only really suitable for the last layer)
                            double error = (neuron == label ? 1 : 0) - (output > 0 ? 1 : 0);
                            for (int j = 0; j < features.Length; j++)
                                weights[layer][neuron][j] += 0.1 * error * features[j];
                        }
                    }

                    double trainingProgress = (double)(epoch + 1) / 10;
                    TrainingProgressChanged?.Invoke(trainingProgress);
                }
            }
        }

        public int Classify(double[] newDataPoint)
        {
            return   deepNeuralNetwor.Classify(newDataPoint);    
            int numLayers = weights.Length;
            int neuronsPerLayer = weights[0].Length;

            double[] input = newDataPoint;

            // Feedforward through all layers
            for (int layer = 0; layer < numLayers; layer++)
            {
                double[] output = new double[neuronsPerLayer];
                for (int neuron = 0; neuron < neuronsPerLayer; neuron++)
                {
                    double neuronOutput = 0;
                    for (int j = 0; j < input.Length; j++)
                        neuronOutput += weights[layer][neuron][j] * input[j];
                    output[neuron] = neuronOutput;
                }
                input = output;  // Output of this layer is input to the next
            }

            // Find the neuron with the highest output in the last layer
            double maxOutput = input[0];
            int maxOutputIndex = 0;
            for (int neuron = 1; neuron < neuronsPerLayer; neuron++)
            {
                if (input[neuron] > maxOutput)
                {
                    maxOutput = input[neuron];
                    maxOutputIndex = neuron;
                }
            }

            return maxOutputIndex;  // This is the classification
        }

       

        public async Task SaveModel(string filePath)
        {
            deepNeuralNetwor.SaveModel(filePath);
            return;
            var json = JsonConvert.SerializeObject(weights);
            await File.WriteAllTextAsync(filePath, json);
        }

        public async Task LoadModel(string filePath)
        {
            deepNeuralNetwor.LoadModel(filePath);
            return;
            var json = await File.ReadAllTextAsync(filePath);
            weights = JsonConvert.DeserializeObject <double[][][]>(json);
            ConvertModel();
            //var json1 = JsonConvert.SerializeObject(weightsTest);
           // await File.WriteAllTextAsync(@"C:\Users\User\source\repos\YAKOV-FRIDMAN\Nearest_Neighbor_Algorithm_and_Neural_Network_Handwritten_Number_Recognition\אלגוריתם שכן קרוב זיהוי מספרים בכתב יד\testmm.json", json1);
        }

        public void ConvertModel()
        {
            weightsTest = new double[10][];
            for (int i = 0; i < weightsTest.GetLength(0); i++)
                weightsTest[i] = new double[784];

            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 784; j++)
                {
                    try
                    {
                        double m = (weights[0][i][j] + weights[1][i][j] + weights[2][i][j]) / 3;
                        weightsTest[i][j] = m;
                    }
                    catch (Exception e)
                    {

                         
                    }
                  
                }
            }
        }
    }
}
