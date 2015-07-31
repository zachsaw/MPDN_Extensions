using System.Drawing;
using System.Drawing.Drawing2D;

namespace Mpdn.Extensions.PlayerExtensions
{
    public static class BitmapHelper
    {
        public static Image Resize(string path, int width, int height)
        {
            var img = Image.FromFile(path);

            var bm = new Bitmap(width, height);

            using (var g = Graphics.FromImage(bm))
            {
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.InterpolationMode = InterpolationMode.HighQualityBilinear;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.DrawImage(img, new Rectangle(0, 0, width, height));
            }

            return bm;
        }
    }
}
