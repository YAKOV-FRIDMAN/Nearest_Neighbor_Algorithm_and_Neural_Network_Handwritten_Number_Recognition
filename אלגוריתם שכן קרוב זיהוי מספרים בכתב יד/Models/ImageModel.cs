using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using אלגוריתם_שכן_קרוב_זיהוי_מספרים_בכתב_יד.Commands;

namespace אלגוריתם_שכן_קרוב_זיהוי_מספרים_בכתב_יד.Models
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
