using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace אלגוריתם_שכן_קרוב_זיהוי_מספרים_בכתב_יד.Services
{
    public class KnnDigitRecognizer
    {
        private int k;
        private List<Tuple<double[], int>> trainingData;

        public KnnDigitRecognizer(int k, List<Tuple<double[], int>> trainingData)
        {
            this.k = k;
            this.trainingData = trainingData;
        }
     
        private double Distance(double[] a, double[] b)
        {
            return Math.Sqrt(a.Zip(b, (x, y) => Math.Pow(x - y, 2)).Sum());
        }

        public int Classify(double[] newDataPoint)
        {
            var neighbors = trainingData
                .Select(data => new { Distance = Distance(newDataPoint, data.Item1), Label = data.Item2 })
                .OrderBy(data => data.Distance)
                .Take(k)
                .GroupBy(data => data.Label)
                .OrderByDescending(group => group.Count())
                .First()
                .Key;

            return neighbors;
        }

        public byte[] GenerateDigit(double[] newDataPoint)
        {
            var nearestNeighbors = trainingData
                .OrderBy(data => Distance(data.Item1, newDataPoint))
                .Take(k);

            int imageSize = trainingData[0].Item1.Length;
            double[] sumImage = new double[imageSize];

            foreach (var neighbor in nearestNeighbors)
            {
                for (int i = 0; i < imageSize; i++)
                {
                    sumImage[i] += neighbor.Item1[i];
                }
            }

            byte[] averageImage = new byte[imageSize];

            for (int i = 0; i < imageSize; i++)
            {
                averageImage[i] = (byte)((sumImage[i] / k) * 255);
            }

            return averageImage;
        }

    }
}


// מצמידים שתי מערכים אחד ליד השני
// בעצם רצים על שני המערכים 
//double distance = 0;
//int i = 0;
//foreach (var item in a)
//{
//    distance += (item - b[i]) * (item - b[i]);
//    i++;
//}
//var r = a.Zip(b, (aa, bb) => CalculatePower(aa - bb, 2)).Sum();

//public double CalculatePower(double baseValue, int exponent)
//{
//    double result = 1;
//    for (int i = 0; i < exponent; i++)
//    {
//        result *= baseValue;
//    }
//    return result;
//}
//public static double CalculateSquareRoot(double n)
//{
//    if (n < 0)
//    {
//        throw new ArgumentException("לא ניתן למצוא שורש ריבועי למספר שלילי");
//    }

//    if (n == 0) return 0;

//    double estimate = n / 2.0;
//    const double threshold = 0.000001; // אפשר לשנות את האפסים כאן בהתאם לדיוק הרצוי

//    while (true)
//    {
//        double newEstimate = (estimate + n / estimate) / 2.0;
//        if (Math.Abs(newEstimate - estimate) < threshold)
//        {
//            return newEstimate;
//        }
//        estimate = newEstimate;
//    }
//}