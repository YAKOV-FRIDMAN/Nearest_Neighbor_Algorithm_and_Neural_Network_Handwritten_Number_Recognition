using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using TestAi.Commands;

namespace TestAi.Models
{
    internal class ImageModel
    {
        public BitmapSource ImageData { get; set; }
        public int Key { get; set; }
        public RelayCommandAsync Classify { get; set; }
        public RelayCommandAsync Generate { get; set; }
        public RelayCommandAsync ShowProperty { get; set; }
    }
}
