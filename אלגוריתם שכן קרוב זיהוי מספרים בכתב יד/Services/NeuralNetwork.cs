using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace אלגוריתם_שכן_קרוב_זיהוי_מספרים_בכתב_יד.Services
{
    public class NeuralNetwork
    {
        private double[][][] weights;  // Weights for each neuron in each layer
        public event Action<double> TrainingProgressChanged;

        public void Train(List<Tuple<double[], int>> trainingData)
        {
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

                    double trainingProgress = (double)(epoch + 1) / totalEpochs;
                    TrainingProgressChanged?.Invoke(trainingProgress);
                }
            }
        }

        public int Classify(double[] newDataPoint)
        {
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
            var json = JsonConvert.SerializeObject(weights);
            await File.WriteAllTextAsync(filePath, json);
        }

        public async Task LoadModel(string filePath)
        {
            var json = await File.ReadAllTextAsync(filePath);
            weights = JsonConvert.DeserializeObject <double[][][]>(json);
        }
    }
}
