using Microsoft.Win32;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Security.Cryptography.Xml;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using TestAi.Commands;
using TestAi.Models;
using TestAi.Services;

namespace TestAi.ViewModels
{
    internal class MainViewModel : ViewModelBase
    {
        public ObservableCollection<ImageModel> Images { get; set; } = new ObservableCollection<ImageModel>();
        public RelayCommandAsync SelectFile { get; set; }
        public RelayCommandAsync LoadMore { get; set; }
        public RelayCommandAsync TraimANNModel { get; set; }
        public RelayCommandAsync TraimANNModel1 { get; set; }
        public RelayCommandAsync TrainCNNModel { get; set; }
        public RelayCommandAsync SaveANNModelToFile { get; set; }
        public RelayCommandAsync SaveANNModelToFile1 { get; set; }
        public RelayCommandAsync SaveCNNModelToFile { get; set; }
        public RelayCommandAsync loadANNModelFile { get; set; }
        public RelayCommandAsync loadANNModelFile1 { get; set; }
        public RelayCommandAsync LoadCNNModelFile { get; set; }
        public RelayCommandAsync Generator { get; set; }
        private string[] lines;
        private string fileLocation;

        public string FileLocation
        {
            get { return fileLocation; }
            set { fileLocation = value; OnPropertyChanged(); }
        }
        private string resultNumber;

        public string ResultNumber
        {
            get { return resultNumber; }
            set { resultNumber = value; OnPropertyChanged(); }
        }
        private double trainingProgress;

        public double TrainingProgress
        {
            get { return trainingProgress; }
            set { trainingProgress = value; OnPropertyChanged(); }
        }
        private double trainingProgress1;

        public double TrainingProgress1
        {
            get { return trainingProgress1; }
            set { trainingProgress1 = value; OnPropertyChanged(); }
        }
        private double trainingProgress2;

        public double TrainingProgress2
        {
            get { return trainingProgress2; }
            set { trainingProgress2 = value; OnPropertyChanged(); }
        }
        private int numberToGenerator;

        public int NumberToGenerator
        {
            get { return numberToGenerator; }
            set { numberToGenerator = value; OnPropertyChanged(); }
        }
        private double threshold;

        public double Threshold
        {
            get { return threshold; }
            set { threshold = value; OnPropertyChanged(); }
        }

        private bool _useGpu;
        public bool UseGpu
        {
            get { return _useGpu; }
            set 
            { 
                _useGpu = value; 
                if (neuralNetwork != null) neuralNetwork.UseGpu = value;
                OnPropertyChanged(); 
            }
        }

        private string _isGpuAvailable;
        public string IsGpuAvailable
        {
            get { return _isGpuAvailable; }
            set { _isGpuAvailable = value; OnPropertyChanged(); }
        }

        private KnnDigitRecognizer knn;
        SimpleANN simpleANN;
        NeuralNetwork neuralNetwork;
        ConvolutionalNeuralNetwork cnn;

        public MainViewModel()
        {
            SelectFile = new RelayCommandAsync(SelectFileExecute);
            LoadMore = new RelayCommandAsync(LoadMoreExecute);
            TraimANNModel = new RelayCommandAsync(TraimANNModelExecute);
            TrainCNNModel = new RelayCommandAsync(TrainCNNModelExecute);
            SaveANNModelToFile = new RelayCommandAsync(SaveANNModelToFileExecute);
            loadANNModelFile = new RelayCommandAsync(loadANNModelFileExecute);
            SaveCNNModelToFile = new RelayCommandAsync(SaveCNNModelToFileExecute);
            LoadCNNModelFile = new RelayCommandAsync(LoadCNNModelFileExecute);
            Generator = new RelayCommandAsync(GeneratorExecute);
            Threshold = 101;
            TraimANNModel1 = new RelayCommandAsync(TraimANNModel1rExecute);
            SaveANNModelToFile1 = new RelayCommandAsync(SaveANNModelToFile1Execute);
            loadANNModelFile1 = new RelayCommandAsync(loadANNModelFile1Execute);

            // Check for GPU availability
            try
            {
                IsGpuAvailable = $"{DeepNeuralNetwork.IsCudaAvailable()} - {DeepNeuralNetwork.GetAcceleratorName()}";
            }
            catch
            {
                IsGpuAvailable = false.ToString();
            }
        }

        private async Task loadANNModelFile1Execute()
        {
            neuralNetwork = new NeuralNetwork();
            neuralNetwork.UseGpu = UseGpu; // Apply current setting

            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                await neuralNetwork.LoadModel(openFileDialog.FileName);
            }
        }

        private async Task SaveANNModelToFile1Execute()
        {
            if (neuralNetwork != null)
            {
                SaveFileDialog saveFileDialog = new SaveFileDialog();
                if (saveFileDialog.ShowDialog() == true)
                {
                    await neuralNetwork.SaveModel(saveFileDialog.FileName + ".json");
                }
            }
        }

        private async Task SaveCNNModelToFileExecute()
        {
            if (cnn != null)
            {
                SaveFileDialog saveFileDialog = new SaveFileDialog();
                if (saveFileDialog.ShowDialog() == true)
                {
                    await cnn.SaveModel(saveFileDialog.FileName + ".json");
                }
            }
        }

        private async Task LoadCNNModelFileExecute()
        {
            cnn = new ConvolutionalNeuralNetwork();
            cnn.TrainingProgressChanged += (progress) =>
            {
                TrainingProgress2 = progress;
            };

            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                await cnn.LoadModel(openFileDialog.FileName);
            }
        }

        private async Task TraimANNModel1rExecute()
        {
            if (trainingData != null)
            {
                await Task.Factory.StartNew(() =>
                {
                    neuralNetwork = new NeuralNetwork();
                    neuralNetwork.UseGpu = UseGpu; // Apply current setting
                    neuralNetwork.TrainingProgressChanged += (e) =>
                    {
                        TrainingProgress1 = e;
                    };
                    neuralNetwork.Train(trainingData);
                }, TaskCreationOptions.LongRunning);
            }
        }

        private async Task TrainCNNModelExecute()
        {
            if (trainingData != null)
            {
                await Task.Factory.StartNew(() =>
                {
                    cnn = new ConvolutionalNeuralNetwork();
                    cnn.TrainingProgressChanged += (progress) =>
                    {
                        TrainingProgress2 = progress;
                    };
                    TrainingProgress2 = 0;
                    cnn.Train(trainingData, 5, 32);
                    TrainingProgress2 = 100;
                }, TaskCreationOptions.LongRunning);
            }
        }

        private async Task GeneratorExecute()
        {
            if (trainingData != null)
            {
                SimpleDigitGenerator simpleDigitGenerator = new SimpleDigitGenerator(trainingData);
                var resImg = simpleDigitGenerator.GenerateAverageDigit2(NumberToGenerator, Threshold);
                Application.Current.Dispatcher.Invoke(() =>
                {
                    var bitmap = BitmapSource.Create(28, 28, 96, 96, PixelFormats.Gray8, null, resImg, 28);
                    double[] doubleArray = new double[resImg.Length];

                    for (int i = 0; i < resImg.Length; i++)
                    {
                        doubleArray[i] = resImg[i];
                    }
                    Images.Add(new ImageModel()
                    {
                        ImageData = bitmap,
                        Key = NumberToGenerator,
                        Classify = new RelayCommandAsync(async () =>
                        {
                            var res = knn.Classify(doubleArray);
                            var res1 = "enpt";
                            if (simpleANN != null)
                            {
                                res1 = simpleANN.Classify(doubleArray).ToString();
                                var full = simpleANN.ClassifyWithProbabilities(doubleArray);
                                var tt = string.Join("\n", full.ToList().Select(_ => $"Key {_.Key} - value {_.Value}"));
                                //MessageBox.Show(tt);
                            }
                            var res3 = "";
                            if (cnn != null)
                            {
                                res3 = "CNN: " + cnn.Classify(doubleArray).ToString();
                            }
                            ResultNumber = $"res1: {res} res2 {res1} {res3}";
                        })
                    });
                });
            }
        }

        private async Task loadANNModelFileExecute()
        {
            simpleANN = new SimpleANN();

            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                await simpleANN.LoadModel(openFileDialog.FileName);
            }

        }

        private async Task SaveANNModelToFileExecute()
        {
            if (simpleANN != null)
            {
                SaveFileDialog saveFileDialog = new SaveFileDialog();
                if (saveFileDialog.ShowDialog() == true)
                {
                    await simpleANN.SaveModel(saveFileDialog.FileName + ".json");
                }
            }
        }

        List<Tuple<double[], int>> trainingData = new List<Tuple<double[], int>>();
        private async Task TraimANNModelExecute()
        {
            if (trainingData != null)
            {
                await Task.Factory.StartNew(() =>
                {
                    simpleANN = new SimpleANN();
                    simpleANN.TrainingProgressChanged += (e) =>
                    {
                        TrainingProgress = e;
                    };
                    simpleANN.Train(trainingData);
                }, TaskCreationOptions.LongRunning);
            }
        }

        private async Task LoadMoreExecute()
        {
            await LOadImages();
            //await Task.Factory.StartNew(() =>
            //{
            //    SaveImages(LoadTrainingDataByte(lines), "C:\\Users\\yafridman\\Documents\\NumImages");
            //}, TaskCreationOptions.LongRunning);
            // await LOadImages();
            //OpenFileDialog openFileDialog = new OpenFileDialog();
            //if (openFileDialog.ShowDialog() == true)
            //{
            //    var imageBytes = File.ReadAllBytes(openFileDialog.FileName);
            //    MLModel2.ModelInput sampleData = new MLModel2.ModelInput()
            //    {
            //        ImageSource = imageBytes,
            //    };
            //
            //    //Load model and predict output
            //    var result = MLModel2.Predict(sampleData);
            //    ResultNumber = result.PredictedLabel.ToString();
            //}
        }

        private async Task SelectFileExecute()
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {

                lines = (await System.IO.File.ReadAllLinesAsync(openFileDialog.FileName)).Skip(1).ToArray();
                FileLocation = openFileDialog.FileName;

                await Task.Factory.StartNew(() =>
                {
                    trainingData = LoadTrainingData(lines);
                    knn = new KnnDigitRecognizer(3, LoadTrainingData(lines));

                }, TaskCreationOptions.LongRunning);
                await LOadImages();

            }
        }

        private List<Tuple<double[], int>> LoadTrainingData(string[] lines)
        {
            // Assume the MNIST data is in CSV format with the label as the first column
            var data = new List<Tuple<double[], int>>();
            //bool first = true;
            foreach (var line in lines)
            {
                //if (first)
                //{
                //    first = false;
                //    continue;
                //}
                var values = line.Split(',').Select(double.Parse).ToArray();
                var label = (int)values[0];
                var features = values.Skip(1).ToArray();
                data.Add(Tuple.Create(features, label));
            }
            return data;
        }
        private List<Tuple<byte[], int>> LoadTrainingDataByte(string[] lines)
        {
            // Assume the MNIST data is in CSV format with the label as the first column
            var data = new List<Tuple<byte[], int>>();
            //bool first = true;
            foreach (var line in lines)
            {
                //if (first)
                //{
                //    first = false;
                //    continue;
                //}
                var values = line.Split(',').Select(byte.Parse).ToArray();
                var label = (int)values[0];
                var features = values.Skip(1).ToArray();
                data.Add(Tuple.Create(features, label));
            }
            return data;
        }

        public void SaveImages(List<Tuple<byte[], int>> imagesData, string baseDirectory)
        {
            foreach (var imageData in imagesData)
            {
                byte[] imageBytes = imageData.Item1;
                int key = imageData.Item2;

                string directoryPath = Path.Combine(baseDirectory, key.ToString());
                Directory.CreateDirectory(directoryPath);

                string imagePath = Path.Combine(directoryPath, $"{Guid.NewGuid()}.png");
                BitmapSource imageBitmap = BitmapSource.Create(28, 28, 96, 96, PixelFormats.Gray8, null, imageBytes, 28);

                using (var fileStream = new FileStream(imagePath, FileMode.Create))
                {
                    PngBitmapEncoder encoder = new PngBitmapEncoder();
                    encoder.Frames.Add(BitmapFrame.Create(imageBitmap));
                    encoder.Save(fileStream);
                }
                TrainingProgress1++;
            }
        }
        void G()
        {
            //Load sample data
            var imageBytes = File.ReadAllBytes(@"C:\Users\yafridman\Documents\NumImages\0\001829e9-32fe-4126-990b-8820804909e8.png");
            MLModel2.ModelInput sampleData = new MLModel2.ModelInput()
            {
                ImageSource = imageBytes,
            };

            //Load model and predict output
            var result = MLModel2.Predict(sampleData);

        }
        string MlModel2(BitmapSource bitmapSource)
        {
            var imageBytes = BitmapSourceToByteArray(bitmapSource);
            MLModel2.ModelInput sampleData = new MLModel2.ModelInput()
            {
                ImageSource = imageBytes,
            };
            var result = MLModel2.Predict(sampleData);
            return result.PredictedLabel.ToString();
        }
        public static byte[] BitmapSourceToByteArray(BitmapSource bitmapSource)
        {
            byte[] bytes;
            using (MemoryStream stream = new MemoryStream())
            {
                BitmapEncoder encoder = new PngBitmapEncoder(); // או כל אנקודר אחר בהתאם לפורמט הרצוי, כמו JpegBitmapEncoder
                encoder.Frames.Add(BitmapFrame.Create(bitmapSource));
                encoder.Save(stream);
                bytes = stream.ToArray();
            }
            return bytes;
        }

        private async Task LOadImages()
        {
            await Task.Factory.StartNew(() =>
            {
                foreach (var line in lines.Skip(Images.Count).Take(500))
                {
                    try
                    {
                        var values = line.Split(',');
                        var label = values[0];
                        var pixels = new byte[28 * 28];
                        bool first = true;
                        for (int i = 1; i < values.Length; i++)
                        {
                            if (first)
                            {
                                first = false;
                                continue;
                            }
                            pixels[i - 1] = byte.Parse(values[i]);
                        }
                        double[] doubleArray = new double[pixels.Length];

                        for (int i = 0; i < pixels.Length; i++)
                        {
                            doubleArray[i] = pixels[i];
                        }

                        Application.Current.Dispatcher.Invoke(() =>
                        {
                            var bitmap = BitmapSource.Create(28, 28, 96, 96, PixelFormats.Gray8, null, pixels, 28);
                            Images.Add(new ImageModel()
                            {
                                ImageData = bitmap,
                                Key = int.Parse(label),
                                Classify = new RelayCommandAsync(async () =>
                                {
                                    var res = knn.Classify(doubleArray);
                                    var res1 = "enpt";
                                    if (simpleANN != null)
                                    {
                                        res1 = simpleANN.Classify(doubleArray).ToString();
                                        var full = simpleANN.ClassifyWithProbabilities(doubleArray);
                                        var tt = string.Join("\n", full.ToList().Select(_ => $"Key {_.Key} - value {_.Value}"));
                                        //MessageBox.Show(tt);
                                    }
                                    var res2 = "";

                                    if (neuralNetwork != null)
                                    {
                                        res2 = "NN Model 2: " + neuralNetwork.Classify(doubleArray).ToString();
                                    }
                                    var res3 = "";
                                    if (cnn != null)
                                    {
                                        res3 = "CNN: " + cnn.Classify(doubleArray).ToString();
                                    }
                                    //var res3 = "Microsoft Model 1: " + MLModel(pixels);

                                    //var res4 = "Microsoft Model 2: " + MlModel2(bitmap);
                                    ResultNumber = $"Knn Model: {res} | NN Model 1 : {res1} {res2} {res3}";
                                })
                            });
                        });
                    }
                    catch (Exception e)
                    {
                        Debug.WriteLine(e);
                    }

                }
            }, TaskCreationOptions.LongRunning);
        }



        public ICommand ExtractPixelsCommand => new RelayCommand<InkCanvas>(ExtractPixels);
        public ICommand ExtractClearPixelsCommand => new RelayCommand<InkCanvas>(ExtractClearPixels);

        private void ExtractClearPixels(InkCanvas canvas)
        {
            canvas.Strokes.Clear();
        }

        private void ExtractPixels(InkCanvas inkCanvas)
        {

            int width = 28;// (int)inkCanvas.ActualWidth;
            int height = 28;  //(int)inkCanvas.ActualHeight;
            RenderTargetBitmap rtb = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Pbgra32);
            rtb.Render(inkCanvas);

            // 2. קריאה של הפיקסלים
            int stride = (width * rtb.Format.BitsPerPixel + 7) / 8;
            byte[] pixelData = new byte[height * stride];

            rtb.CopyPixels(pixelData, stride, 0);



            int bytesPerPixel = 4;

            Pixel[,] pixelArray2D = new Pixel[height, width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int index = (y * width + x) * bytesPerPixel;
                    pixelArray2D[y, x] = new Pixel
                    {
                        R = pixelData[index],
                        G = pixelData[index + 1],
                        B = pixelData[index + 2],
                        Alpha = pixelData[index + 3]
                    };
                }
            }

            try
            {



                if (knn != null)
                {
                    // double[] doubleArray = pixelData.Select(b => (double)b).ToArray();
                    var pixelsDouble = new double[28 * 28];
                    for (int i = 0; i < 28 * 28; i++)
                    {
                        int byteIndex = i * 4;
                        pixelsDouble[i] = pixelData[byteIndex + 2] / 255.0;
                    }

                    var bytes = ConvertToOnePixel(pixelArray2D, 0);
                    double[] doubleArray = bytes.Select(b => (double)b).ToArray();


                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        BitmapSource invertedBitmap = BitmapSource.Create(width, height, 96, 96, PixelFormats.Gray8, null, bytes, 28);
                        Images.Add(new ImageModel() { ImageData = invertedBitmap, Key = 0 });
                    });
                    var res = knn.Classify(doubleArray);
                    var resimg = knn.GenerateDigit(doubleArray);
                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        BitmapSource invertedBitmap = BitmapSource.Create(width, height, 96, 96, PixelFormats.Gray8, null, resimg, 28);
                        Images.Add(new ImageModel()
                        {
                            ImageData = invertedBitmap,
                            Key = 0,
                            Classify = new RelayCommandAsync(async () =>
                            {
                                var res = knn.Classify(doubleArray);
                                var res1 = "enpt";
                                if (simpleANN != null)
                                {
                                    res1 = simpleANN.Classify(doubleArray).ToString();
                                    var full = simpleANN.ClassifyWithProbabilities(doubleArray);
                                    var tt = string.Join("\n", full.ToList().Select(_ => $"Key {_.Key} - value {_.Value}"));
                                    //MessageBox.Show(tt);
                                }
                                var res2 = "";
                                if (neuralNetwork != null)
                                {
                                    res2 = "NN Model 2: " + neuralNetwork.Classify(doubleArray).ToString();
                                }
                                var res3 = "";
                                if (cnn != null)
                                {
                                    res3 = "CNN: " + cnn.Classify(doubleArray).ToString();
                                }
                                //var res3 = "Microsoft Model 1: " + MLModel(bytes);

                                //var res4 = "Microsoft Model 2: " + MlModel2(bitmap);
                                ResultNumber = $"Knn Model: {res} | NN Model 1 : {res1} | {res2} | {res3}";
                            })
                        });
                    });
                    var res1 = "enpt";
                    if (simpleANN != null)
                    {
                        res1 = simpleANN.Classify(doubleArray).ToString();
                        var full = simpleANN.ClassifyWithProbabilities(doubleArray);
                        var tt = string.Join("\n", full.ToList().Select(_ => $"Key {_.Key} - value {_.Value}"));
                        //MessageBox.Show(tt);
                        var resImg = simpleANN.GenerateImage(doubleArray);
                        Application.Current.Dispatcher.Invoke(() =>
                        {
                            BitmapSource invertedBitmap = BitmapSource.Create(width, height, 96, 96, PixelFormats.Gray8, null, resImg, 28);
                            Images.Add(new ImageModel()
                            {
                                ImageData = invertedBitmap,
                                Key = 0,
                                Classify = new RelayCommandAsync(async () =>
                                {
                                    var res = knn.Classify(doubleArray);
                                    var res1 = "enpt";
                                    if (simpleANN != null)
                                    {
                                        res1 = simpleANN.Classify(doubleArray).ToString();
                                        var full = simpleANN.ClassifyWithProbabilities(doubleArray);
                                        var tt = string.Join("\n", full.ToList().Select(_ => $"Key {_.Key} - value {_.Value}"));
                                        //MessageBox.Show(tt);
                                    }
                                    var res2 = "";
                                    if (neuralNetwork != null)
                                    {
                                        res2 = "NN Model 2: " + neuralNetwork.Classify(doubleArray).ToString();
                                    }
                                    var res3 = "";
                                    if (cnn != null)
                                    {
                                        res3 = "CNN: " + cnn.Classify(doubleArray).ToString();
                                    }
                                    //var res3 = "Microsoft Model 1: " + MLModel(bytes);

                                    //var res4 = "Microsoft Model 2: " + MlModel2(invertedBitmap);
                                    ResultNumber = $"Knn Model: {res} | NN Model 1 : {res1}  | {res2} | {res3} ";
                                })
                            });
                        });
                    }
                    var res2 = "";
                    BitmapSource invertedBitmap = BitmapSource.Create(width, height, 96, 96, PixelFormats.Gray8, null, resimg, 28);
                    if (neuralNetwork != null)
                    {
                        res2 = "NN Model 2: " + neuralNetwork.Classify(doubleArray).ToString();
                    }
                    var res3 = "";
                    if (cnn != null)
                    {
                        res3 = "CNN: " + cnn.Classify(doubleArray).ToString();
                    }
                    //var res3 = "Microsoft Model 1: " + MLModel(bytes);

                    //var res4 = "Microsoft Model 2: " + MlModel2(invertedBitmap);
                    ResultNumber = $"Knn Model: {res} | NN Model 1 : {res1} | {res2} | {res3} ";
                }

            }
            catch (Exception e)
            {

                MessageBox.Show(e.ToString());
            }
        }

        public bool ContainsPixel(Pixel[,] pixelArray2D, int val)
        {
            for (int y = 0; y < pixelArray2D.GetLength(0); y++)
            {
                for (int x = 0; x < pixelArray2D.GetLength(1); x++)
                {
                    if (pixelArray2D[y, x].R == val && pixelArray2D[y, x].G == val && pixelArray2D[y, x].B == val)
                    {
                        return true; // מצאנו פיקסל שחור
                    }
                }
            }
            return false; // לא מצאנו פיקסל שחור
        }

        public byte[] ConvertToOnePixel(Pixel[,] pixelArray2D, int val)
        {
            byte[] onePixel = new byte[pixelArray2D.Length];
            int i = 0;
            for (int y = 0; y < pixelArray2D.GetLength(0); y++)
            {
                for (int x = 0; x < pixelArray2D.GetLength(1); x++)
                {
                    byte grayValue = (byte)((pixelArray2D[y, x].R + pixelArray2D[y, x].G + pixelArray2D[y, x].B) / 3);
                    onePixel[i] = (byte)(255 - grayValue);

                    //  onePixel[i] = pixelArray2D[y, x].G;
                    //if (pixelArray2D[y, x].R == val && pixelArray2D[y, x].G == val && pixelArray2D[y, x].B == val)
                    //{
                    //    onePixel[y] = 0;
                    //}
                    //else
                    //{
                    //    onePixel[y] = 255;
                    //}
                    i++;
                }
            }
            return onePixel; // לא מצאנו פיקסל שחור
        }
        public bool ContainsPixel(Pixel[,] pixelArray2D, Pixel val)
        {
            for (int y = 0; y < pixelArray2D.GetLength(0); y++)
            {
                for (int x = 0; x < pixelArray2D.GetLength(1); x++)
                {
                    if (pixelArray2D[y, x].R == val.R && pixelArray2D[y, x].G == val.G && pixelArray2D[y, x].B == val.B)
                    {
                        return true; // מצאנו פיקסל שחור
                    }
                }
            }
            return false; // לא מצאנו פיקסל שחור
        }

        private string MLModel(byte[] pixesl)
        {
            MLModel1.ModelInput modelInput = new MLModel1.ModelInput();
            Type myType = modelInput.GetType();
            for (int i = 1; i < pixesl.Length; i++)
            {
                string propertyName = "Col" + (i).ToString();
                PropertyInfo propertyInfo = myType.GetProperty(propertyName);
                if (propertyInfo != null)
                {
                    propertyInfo.SetValue(modelInput, pixesl[i].ToString());
                }
            }
            var res = MLModel1.Predict(modelInput);
            return res.PredictedLabel.ToString();
        }

    }



    public class Pixel
    {
        public byte R { get; set; }
        public byte G { get; set; }
        public byte B { get; set; }
        public byte Alpha { get; set; }  // Transparency. 255 is fully opaque, 0 is fully transparent.

        public byte GetGrayScaleValue()
        {
            return (byte)((R + G + B) / 3);
        }

        public override string ToString()
        {
            return $"R {R}, G {G}, B {B}";
        }
    }

}
