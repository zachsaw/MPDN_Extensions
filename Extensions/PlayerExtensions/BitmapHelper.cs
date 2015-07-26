using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace Mpdn.Extensions.PlayerExtensions
{
    public static class BitmapHelper
    {
        /// <summary>
        /// Converts Bitmap to array. Supports only Format32bppArgb pixel format.
        /// </summary>
        /// <param name="bmp">Bitmap to convert.</param>
        /// <returns>Output array.</returns>
        public static ushort[][,] ConvertBitmapToArray(Bitmap bmp)
        {

            var array = new ushort[4][,];

            for (int i = 0; i < 4; i++)
                array[i] = new ushort[bmp.Width, bmp.Height];

            var bd = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            int nOffset = (bd.Stride - bd.Width * 4);

            unsafe
            {

                byte* p = (byte*)bd.Scan0;

                for (int y = 0; y < bd.Height; y++)
                {
                    for (int x = 0; x < bd.Width; x++)
                    {

                        array[3][x, y] = (ushort)p[3];
                        array[2][x, y] = (ushort)p[2];
                        array[1][x, y] = (ushort)p[1];
                        array[0][x, y] = (ushort)p[0];

                        p += 4;
                    }

                    p += nOffset;
                }
            }

            bmp.UnlockBits(bd);

            return array;
        }

        /// <summary>
        /// Converts array to Bitmap. Supports only Format32bppArgb pixel format.
        /// </summary>
        /// <param name="array">Array to convert.</param>
        /// <returns>Output Bitmap.</returns>
        public static Bitmap ConvertArrayToBitmap(ushort[][,] array)
        {

            int width = array[0].GetLength(0);
            int height = array[0].GetLength(1);

            var bmp = new Bitmap(width, height, PixelFormat.Format32bppArgb);

            var bd = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            int nOffset = (bd.Stride - bd.Width * 4);

            unsafe
            {

                byte* p = (byte*)bd.Scan0;

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {

                        p[3] = (byte)Math.Min(Math.Max(array[3][x, y], Byte.MinValue), Byte.MaxValue);
                        p[2] = (byte)Math.Min(Math.Max(array[2][x, y], Byte.MinValue), Byte.MaxValue);
                        p[1] = (byte)Math.Min(Math.Max(array[1][x, y], Byte.MinValue), Byte.MaxValue);
                        p[0] = (byte)Math.Min(Math.Max(array[0][x, y], Byte.MinValue), Byte.MaxValue);

                        p += 4;
                    }

                    p += nOffset;
                }
            }

            bmp.UnlockBits(bd);

            return bmp;
        }
    }
}
