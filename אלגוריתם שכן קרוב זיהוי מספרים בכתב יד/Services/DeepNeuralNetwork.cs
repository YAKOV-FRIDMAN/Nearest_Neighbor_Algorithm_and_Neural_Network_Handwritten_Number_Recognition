using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestAi.Services
{
    public class DeepNeuralNetwork
    {
        private List<double[][]> weights; // List of layers, with 2D array for each layer's weights
        double learningRate = 0.01; // קצב למידה
        public DeepNeuralNetwork(int inputSize, int[] layerSizes)
        {
            var rnd = new Random();
            weights = new List<double[][]>();

            int previousSize = inputSize; // The size of the input layer

            // Initialize weights for each layer
            foreach (int size in layerSizes)
            {
                double[][] layerWeights = new double[size][];
                for (int i = 0; i < size; i++)
                {
                    layerWeights[i] = new double[previousSize];
                    for (int j = 0; j < previousSize; j++)
                    {
                        // Random weights initialization
                        layerWeights[i][j] = rnd.NextDouble() * 2 - 1; // Between -1 and 1
                    }
                }
                weights.Add(layerWeights);
                previousSize = size; // Update the size for the next layer's input
            }
        }

        public void Train(List<Tuple<double[], int>> trainingData)
        {
            int epochs = 1000; // מספר אפוחות לאימון

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var (input, expectedIndex) in trainingData)
                {
                    // Forward propagation
                    var layerOutputs = ForwardPropagate(input);

                    // Convert expected index to output array
                    double[] expectedOutputs = new double[layerOutputs.Last().Length];
                    expectedOutputs[expectedIndex] = 1;

                    // Backpropagation
                    var gradients = Backpropagate(expectedOutputs, layerOutputs);

                    // Weights update
                    UpdateWeights(gradients, layerOutputs);
                }
            }
        }


        private void UpdateWeights(double[][] gradients, List<double[]> layerOutputs)
        {
            // התחל מהשכבה השנייה כיוון שהקלטים לשכבה הראשונה הם הנתונים המקוריים
            for (int layer = 1; layer < weights.Count; layer++)
            {
                double[][] layerWeights = weights[layer];
                double[] previousLayerOutputs = layerOutputs[layer - 1]; // השתמש בפלט מהשכבה הקודמת לעדכון המשקולות

                for (int neuron = 0; neuron < layerWeights.Length; neuron++)
                {
                    for (int weightIndex = 0; weightIndex < layerWeights[neuron].Length; weightIndex++)
                    {
                        double gradient = gradients[layer][neuron];
                        double input = previousLayerOutputs[weightIndex];
                        // עדכן את המשקולת בהתאם לגרדיאנט ולקלט
                        weights[layer][neuron][weightIndex] -= learningRate * gradient * input;
                    }
                }
            }
        }


        private double[][] Backpropagate(double[] expectedOutputs, List<double[]> layerOutputs)
        {
            int numLayers = weights.Count;
            double[][] gradients = new double[numLayers][];

            // Calculate error for the output layer
            double[] outputLayerOutputs = layerOutputs.Last();
            double[] outputError = new double[outputLayerOutputs.Length];
            for (int i = 0; i < outputLayerOutputs.Length; i++)
            {
                // Assuming the use of a sigmoid activation function
                outputError[i] = (outputLayerOutputs[i] - expectedOutputs[i]) * outputLayerOutputs[i] * (1 - outputLayerOutputs[i]);
            }
            gradients[numLayers - 1] = outputError;

            // Backpropagate the error
            for (int layer = numLayers - 2; layer >= 0; layer--)
            {
                double[] layerOutputsPrev = layerOutputs[layer];
                double[] layerError = new double[layerOutputsPrev.Length];
                double[] nextLayerError = gradients[layer + 1];
                double[][] nextLayerWeights = weights[layer + 1];

                for (int i = 0; i < layerOutputsPrev.Length; i++)
                {
                    double error = 0;
                    for (int j = 0; j < nextLayerWeights.Length; j++)
                    {
                        error += nextLayerError[j] * nextLayerWeights[j][i];
                    }
                    // Assuming the use of a sigmoid activation function for hidden layers as well
                    layerError[i] = error * layerOutputsPrev[i] * (1 - layerOutputsPrev[i]);
                }
                gradients[layer] = layerError;
            }

            return gradients;
        }


        private double CalculateCost(double[] output, int expectedOutput)
        {
            double cost = 0;
            for (int i = 0; i < output.Length; i++)
            {
                cost += Math.Pow((output[i] - (i == expectedOutput ? 1 : 0)), 2);
            }
            return cost / output.Length;
        }

        private List<double[]> ForwardPropagate(double[] inputs)
        {
            List<double[]> layerOutputs = new List<double[]>();
            double[] activations = inputs;

            foreach (var layerWeights in weights)
            {
                double[] newActivations = new double[layerWeights.Length];
                for (int i = 0; i < layerWeights.Length; i++)
                {
                    double activation = 0;
                    for (int j = 0; j < layerWeights[i].Length; j++)
                    {
                        activation += layerWeights[i][j] * activations[j];
                    }
                    // Assuming the use of a sigmoid activation function
                    newActivations[i] = 1 / (1 + Math.Exp(-activation));
                }
                layerOutputs.Add(newActivations);
                activations = newActivations; // Update activations for the next layer
            }

            return layerOutputs; // Return all layers outputs
        }


        public int Classify(double[] inputData)
        {
            double[] output = inputData; // Start with input data

            foreach (var layerWeights in weights)
            {
                double[] layerOutput = new double[layerWeights.Length];
                for (int i = 0; i < layerWeights.Length; i++)
                {
                    double neuronOutput = 0;
                    for (int j = 0; j < layerWeights[i].Length; j++)
                    {
                        neuronOutput += layerWeights[i][j] * output[j];
                    }
                    // Example with a Sigmoid activation function
                    layerOutput[i] = 1 / (1 + Math.Exp(-neuronOutput));
                }
                output = layerOutput; // Use current layer's output as next layer's input
            }

            // Assuming the last layer is for classification and has one neuron per class
            int maxIndex = Array.IndexOf(output, output.Max());
            return maxIndex; // Index of the highest output value corresponds to the class
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
