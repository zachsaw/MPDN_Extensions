using System;
using System.Collections.Generic;
using System.Drawing;

namespace Mpdn.RenderScript
{
    namespace Shiandow.Deband
    {
        public class Deband : RenderChain
        {
            private const float DEFAULT_THRESHOLD = 0.5f;
            private const float DEFAULT_MARGIN = 1.0f;

            public int maxbitdepth { get; set; }
            public float threshold { get; set; }
            public float margin { get; set; }
            public bool advancedMode { get; set; }
            public bool legacyMode { get; set; }

            public Deband()
            {
                maxbitdepth = 8;
                threshold = DEFAULT_THRESHOLD;
                margin = DEFAULT_MARGIN;
                advancedMode = false;
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                var gaussian = new Scaler.Custom(new GaussianBlur(0.75), ScalerTaps.Four, false);
                var bilinear = new Scaler.HwBilinear(); // Good enough (?)
             
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
                    advancedMode ? margin : DEFAULT_MARGIN,
                    YuvConsts[0], YuvConsts[1]
                };

                var LinearToGamma = CompileShader("../Common/LinearToGamma.hlsl");
                var GammaToLinear = CompileShader("../Common/GammaToLinear.hlsl");
                var Deband = CompileShader(legacyMode ? "DebandOld.hlsl" : "Deband.hlsl").Configure(true, Consts);
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
                    var factor = 2.0;
                    size = new Size((int)Math.Floor(size.Width / factor), (int)Math.Floor(size.Height / factor));
                    if (size.Width == 0 || size.Height == 0)
                        break;

                    current = new ResizeFilter(current, size, bilinear, bilinear);
                    downscaled.Push(current);
                }

                var deband = downscaled.Pop();
                while (downscaled.Count > 0)
                {
                    current = downscaled.Pop();
                    current = new ShaderFilter(Deband, current, deband);

                    if (downscaled.Count > 0)
                    {
                        var diff = new ShaderFilter(Subtract, deband, current);
                        deband = new ShaderFilter(SubtractLimited, current, diff);
                    }
                    else
                        deband = current;
                }

                return deband.ConvertToRgb();
            }

            private class GaussianBlur : ICustomLinearScaler
            {
                private double m_Sigma;

                public GaussianBlur(double sigma)
                {
                    m_Sigma = sigma;
                }

                public Guid Guid
                {
                    get { return Guid.Empty; }
                }

                public string Name
                {
                    get { return ""; }
                }

                public bool AllowDeRing
                {
                    get { return false; }
                }

                public ScalerTaps MaxTapCount
                {
                    get { return ScalerTaps.Eight; }
                }

                public float GetWeight(float n, int width)
                {
                    return (float)GaussianKernel(n);
                }

                private double GaussianKernel(double x)
                {
                    var sigma = m_Sigma;
                    return Math.Exp(-(x * x / (2 * sigma * sigma)));
                }
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
