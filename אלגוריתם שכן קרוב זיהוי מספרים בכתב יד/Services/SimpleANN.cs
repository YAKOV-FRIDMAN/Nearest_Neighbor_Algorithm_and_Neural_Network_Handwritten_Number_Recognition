using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace אלגוריתם_שכן_קרוב_זיהוי_מספרים_בכתב_יד.Services
{
    public class SimpleANN
    {
        private double[][] weights;  // Weights for each neuron
        public double TrainingProgress { get; private set; } = 0.0;
        public event Action<double> TrainingProgressChanged;
        public void Train(List<Tuple<double[], int>> trainingData)
        {
            ;
            // Initialize weights randomly
            var rnd = new Random();
            weights = new double[10][];
            for (int i = 0; i < 10; i++)
            {
                weights[i] = new double[trainingData[0].Item1.Length];
                for (int j = 0; j < weights[i].Length; j++)
                    weights[i][j] = rnd.NextDouble();
            }
            int totalEpochs = 100; // Assume 100 epochs for simplicity
            // A very simplistic training procedure
            for (int epoch = 0; epoch < 100; epoch++)
            {
                foreach (var data in trainingData)
                {
                    int label = data.Item2;
                    double[] features = data.Item1;

                    for (int i = 0; i < 10; i++)
                    {
                        double output = 0;
                        for (int j = 0; j < features.Length; j++)
                            output += weights[i][j] * features[j];

                        // Simple Perceptron learning rule
                        double error = (i == label ? 1 : 0) - (output > 0 ? 1 : 0);
                        for (int j = 0; j < features.Length; j++)
                            weights[i][j] += 0.1 * error * features[j];
                    }
                    TrainingProgress = (double)(epoch + 1);/// totalEpochs;
                    TrainingProgressChanged?.Invoke(TrainingProgress);
                }
            
            }
        }

        public int Classify(double[] newDataPoint)
        {
            double bestOutput = double.MinValue;
            int bestLabel = -1;
            for (int i = 0; i < 10; i++)
            {
                double output = 0;
                for (int j = 0; j < newDataPoint.Length; j++)
                    output += weights[i][j] * newDataPoint[j];

                if (output > bestOutput)
                {
                    bestOutput = output;
                    bestLabel = i;
                }
            }

            return bestLabel;
        }
      
        public byte[] GenerateImage(double[] newDataPoint, int width = 28, int height = 28)
        {
            int label = Classify(newDataPoint);  // Classify the new data point to find the relevant neuron
            double[] relevantWeights = weights[label];  // Get the weights of the relevant neuron

            // Normalize the weights to be between 0 and 255
            double minWeight = relevantWeights.Min();
            double maxWeight = relevantWeights.Max();
            double scale = 255 / (maxWeight - minWeight);
              byte[] normalizedWeights = relevantWeights.Select(w => (byte)((w - minWeight) * scale)).ToArray();
             // byte[] normalizedWeights = relevantWeights.Select(w => (byte)Math.Round((w - minWeight) * scale)).ToArray();


            // Create a new image using the normalized weights
            byte[] imageBytes = new byte[width * height];
            for (int i = 0; i < imageBytes.Length; i++)
            {
                imageBytes[i] = normalizedWeights[i % normalizedWeights.Length];
            }

            //for (int y = 0; y < height; y++)
            //{
            //    for (int x = 0; x < width; x++)
            //    {
            //        int index = y * width + x;
            //        if (index < normalizedWeights.Length)
            //        {
            //            imageBytes[index] = normalizedWeights[index];
            //        }
            //    }
            //}


            return imageBytes;
        }
        public byte[] GenerateImage2(double[] newDataPoint, int width = 28, int height = 28)
        {
            int label = Classify(newDataPoint);  // Classify the new data point to find the relevant neuron
            double[] relevantWeights = weights[label];  // Get the weights of the relevant neuron

            // Simulate the action of the network on the new data point
            double[] networkOutput = new double[width * height];
            for (int i = 0; i < networkOutput.Length; i++)
            {
                for (int j = 0; j < newDataPoint.Length; j++)
                {
                    networkOutput[i] += relevantWeights[j] * newDataPoint[j];
                }
            }

            // Normalize the network output to be between 0 and 255
            double minOutput = networkOutput.Min();
            double maxOutput = networkOutput.Max();
            double scale = 255 / (maxOutput - minOutput);
            byte[] imageBytes = networkOutput.Select(o => (byte)((o - minOutput) * scale)).ToArray();

            return imageBytes;
        }

        public async Task SaveModel(string filePath)
        {
            var json = JsonConvert.SerializeObject(weights);
           await File.WriteAllTextAsync(filePath, json);
        }

        public async Task LoadModel(string filePath)
        {
            var json = await File.ReadAllTextAsync(filePath);
            weights = JsonConvert.DeserializeObject<double[][]>(json);
        }
        
    }
}
