using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace Mpdn.RenderScript
{
    namespace Shiandow.Deband
    {
        public class Deband : RenderChain
        {
            public int detaillevel { get; set; }
            public int maxbitdepth { get; set; }
            public float threshold { get; set; }

            public Deband()
            {
                maxbitdepth = 8;
                threshold = 0.7f;
                detaillevel = 2;
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                var bilinear = new Scaler.HwBilinear();
             
                int bits = 8;
                switch (Renderer.InputFormat)
                {
                    case FrameBufferInputFormat.P010: bits = 10; break;
                    case FrameBufferInputFormat.Y410: bits = 10; break;
                    case FrameBufferInputFormat.P016: bits = 16; break;
                    case FrameBufferInputFormat.Y416: bits = 16; break;
                    case FrameBufferInputFormat.Rgb24: return sourceFilter;
                    case FrameBufferInputFormat.Rgb32: return sourceFilter;
                }
                if (bits > maxbitdepth) return sourceFilter;

                float[] YuvConsts = new float[2];
                switch (Renderer.Colorimetric)
                {
                    case YuvColorimetric.Auto: break;
                    case YuvColorimetric.FullRangePc601: YuvConsts = new[] { 0.114f, 0.299f, 0.0f }; break;
                    case YuvColorimetric.FullRangePc709: YuvConsts = new[] { 0.0722f, 0.2126f, 0.0f }; break;
                    case YuvColorimetric.FullRangePc2020: YuvConsts = new[] { 0.0593f, 0.2627f, 0.0f }; break;
                    case YuvColorimetric.ItuBt601: YuvConsts = new[] { 0.114f, 0.299f, 1.0f }; break;
                    case YuvColorimetric.ItuBt709: YuvConsts = new[] { 0.0722f, 0.2126f, 1.0f }; break;
                    case YuvColorimetric.ItuBt2020: YuvConsts = new[] { 0.0593f, 0.2627f, 1.0f }; break;
                }

                float[] Consts = new[] {
                    (1 << bits) - 1, 
                    threshold,
                    YuvConsts[0], YuvConsts[1]
                };

                var Deband = CompileShader("Deband.hlsl").Configure(false, Consts);
                var Subtract = CompileShader("Subtract.hlsl").Configure(true, format: TextureFormat.Float16);
                var SubtractLimited = CompileShader("SubtractLimited.hlsl").Configure(true, Consts);

                IFilter input = sourceFilter.ConvertToYuv();
                var inputsize = input.OutputSize;

                var current = input;
                var downscaled = new Stack<IFilter>();
                downscaled.Push(current);

                // Generate downscaled images
                var size = inputsize;
                for (int i = 0; i < 8; i++)
                {
                    double factor = i == 0 ? (1 << detaillevel) : 2.0;
                    size = new Size((int)Math.Floor(size.Width / factor), (int)Math.Floor(size.Height / factor));
                    if (size.Width == 0 || size.Height == 0)
                        break;

                    current = new ResizeFilter(current, size, bilinear, bilinear);
                    downscaled.Push(current);
                }

                size = downscaled.ElementAt(Math.Max(0,downscaled.Count - detaillevel - 1)).OutputSize;
                var deband = downscaled.Pop();
                while (downscaled.Count > 0)
                {
                    current = downscaled.Pop();

                    if (downscaled.Count > 0)
                    {
                        current = new ShaderFilter(Deband, current, deband);
                        var diff = new ShaderFilter(Subtract, deband, current);
                        deband = new ShaderFilter(SubtractLimited, current, diff);
                    }
                    else deband = new ShaderFilter(Deband, current, deband);
                }

                return deband.ConvertToRgb();
            }
        }

        public class DebandUi : RenderChainUi<Deband, DebandConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Shiandow.Deband"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("EE3B46F7-00BB-4299-9B3F-058BCC3F591C"),
                        Name = "Deband",
                        Description = "Removes banding",
                        Copyright = "Deband by Shiandow",
                    };
                }
            }
        }
    }
}
