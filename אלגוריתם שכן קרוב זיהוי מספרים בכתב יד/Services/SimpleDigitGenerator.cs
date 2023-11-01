using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace אלגוריתם_שכן_קרוב_זיהוי_מספרים_בכתב_יד.Services
{
    internal class SimpleDigitGenerator
    {
        private List<Tuple<double[], int>> trainingData;

        public SimpleDigitGenerator(List<Tuple<double[], int>> trainingData)
        {
            this.trainingData = trainingData;
        }

        public byte[] GenerateAverageDigit(int digit)
        {
            var imagesOfDigit = trainingData.Where(data => data.Item2 == digit);
            int numImages = imagesOfDigit.Count();
            int imageSize = trainingData[0].Item1.Length;
            double[] sumImage = new double[imageSize];

            foreach (var image in imagesOfDigit)
            {
                for (int i = 0; i < imageSize; i++)
                {
                    sumImage[i] += image.Item1[i];
                }
            }

            byte[] averageImage = new byte[imageSize];

            for (int i = 0; i < imageSize; i++)
            {
                double averageValue = sumImage[i] / numImages;
                averageImage[i] = (byte)(averageValue * 255); // Scaling the value to the range 0-255
            }

            return averageImage;
        }
        public byte[] GenerateAverageDigit2(int digit, double threshold = 0.2)
        {
            var imagesOfDigit = trainingData.Where(data => data.Item2 == digit);
            int numImages = imagesOfDigit.Count();
            int imageSize = trainingData[0].Item1.Length;

            double[] sumImage = new double[imageSize];
            double[] sumSquaredImage = new double[imageSize];

            foreach (var image in imagesOfDigit)
            {
                for (int i = 0; i < imageSize; i++)
                {
                    sumImage[i] += image.Item1[i];
                    sumSquaredImage[i] += image.Item1[i] * image.Item1[i];
                }
            }

            byte[] averageImage = new byte[imageSize];

            for (int i = 0; i < imageSize; i++)
            {
                double mean = sumImage[i] / numImages;
                double variance = (sumSquaredImage[i] / numImages) - (mean * mean);
                double standardDeviation = Math.Sqrt(variance);

                if (standardDeviation < threshold)
                {
                    averageImage[i] = (byte)(mean * 255);
                }
                else
                {
                    averageImage[i] = 0;
                }
            }

            return averageImage;
        }
        public byte[] GenerateAverageDigit3(int digit, double threshold = 0.2)
        {
            var imagesOfDigit = trainingData.Where(data => data.Item2 == digit);
            int numImages = imagesOfDigit.Count();
            int imageSize = trainingData[0].Item1.Length;

            double[] sumImage = new double[imageSize];
            double[] sumSquaredImage = new double[imageSize];

            foreach (var image in imagesOfDigit)
            {
                for (int i = 0; i < imageSize; i++)
                {
                    sumImage[i] += image.Item1[i];
                    sumSquaredImage[i] += image.Item1[i] * image.Item1[i];
                }
            }

            byte[] averageImage = new byte[imageSize];

            for (int i = 0; i < imageSize; i++)
            {
                double mean = sumImage[i] / numImages;
                double variance = (sumSquaredImage[i] / numImages) - (mean * mean);
                double standardDeviation = Math.Sqrt(variance);

                if (standardDeviation < threshold)
                {
                    averageImage[i] = (byte)(255 - (mean * 255));  // Invert the pixel value
                }
                else
                {
                    averageImage[i] = 255;  // Set the pixel value to white
                }
            }

            return averageImage;
        }
    }
}
