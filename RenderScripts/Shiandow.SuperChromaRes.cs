using System;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Shiandow.SuperRes
    {
        public class SuperChromaRes : RenderChain
        {
            [YAXDontSerialize]
            public int Passes { get; set; }
            private IScaler upscaler, downscaler;
            public bool FirstPassOnly;

            public SuperChromaRes()
            {
                Passes = 2;
                FirstPassOnly = true;
                upscaler = new Scaler.Bilinear();
                downscaler = new Scaler.Bilinear();
            }

            protected override string ShaderPath
            {
                get { return "SuperRes"; }
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                IFilter yuv;

                var chromaSize = Renderer.ChromaSize;
                var targetSize = sourceFilter.OutputSize;

                var Diff = CompileShader("SuperChromaRes/Diff.hlsl");
                var CopyLuma = CompileShader("SuperChromaRes/CopyLuma.hlsl");
                var CopyChroma = CompileShader("SuperChromaRes/CopyChroma.hlsl");
                var SuperRes = CompileShader("SuperChromaRes/SuperRes.hlsl");

                var GammaToLab = CompileShader("GammaToLab.hlsl");
                var LabToGamma = CompileShader("LabToGamma.hlsl");
                var LinearToGamma = CompileShader("LinearToGamma.hlsl");
                var GammaToLinear = CompileShader("GammaToLinear.hlsl");
                var LabToLinear = CompileShader("LabToLinear.hlsl");
                var LinearToLab = CompileShader("LinearToLab.hlsl");

                // Skip if downscaling
                if (targetSize.Width <= chromaSize.Width && targetSize.Height <= chromaSize.Height)
                    return sourceFilter;

                // Original values
                var YInput = new YSourceFilter();
                var uInput = new USourceFilter();
                var vInput = new VSourceFilter();

                yuv = sourceFilter.ConvertToYuv();

                float[] YuvConsts = new float[2];
                switch (Renderer.Colorimetric)
                {
                    case YuvColorimetric.Auto : return sourceFilter;
                    case YuvColorimetric.FullRange : return sourceFilter;
                    case YuvColorimetric.FullRangePc601: YuvConsts = new[] { 0.114f, 0.299f }; break;
                    case YuvColorimetric.FullRangePc709: YuvConsts = new[] { 0.0722f, 0.2126f }; break;
                    case YuvColorimetric.FullRangePc2020: YuvConsts = new[] { 0.0593f, 0.2627f }; break;
                    case YuvColorimetric.ItuBt601: YuvConsts = new[] { 0.114f, 0.299f }; break;
                    case YuvColorimetric.ItuBt709: YuvConsts = new[] { 0.0722f, 0.2126f }; break;
                    case YuvColorimetric.ItuBt2020: YuvConsts = new[] { 0.0593f, 0.2627f }; break;
                }

                for (int i = 1; i <= Passes; i++)
                {
                    IFilter linear, res, diff;

                    // Compare to chroma
                    linear = new ShaderFilter(GammaToLinear, new RgbFilter(yuv));
                    res = new ResizeFilter(linear, chromaSize, upscaler, downscaler);
                    res = new ShaderFilter(LinearToGamma, res).ConvertToYuv();
                    diff = new ShaderFilter(Diff, YuvConsts, res, uInput, vInput);
                    if (!(upscaler is Scaler.Bilinear))
                        diff = new ResizeFilter(diff, targetSize, upscaler, downscaler); // Scale to output size

                    // Update result
                    yuv = new ShaderFilter(SuperRes, (upscaler is Scaler.Bilinear), new[]{ 0.8f, 0.25f, 0.5f, 0f, YuvConsts[0], YuvConsts[1] }, yuv, diff, uInput, vInput);

                    if (FirstPassOnly == true)
                        upscaler = new Scaler.Bilinear();
                }

                return yuv.ConvertToRgb();
            }
        }

        public class SuperChromaResUi : RenderChainUi<SuperChromaRes>
        {
            protected override RenderScriptDescriptor ScriptDescriptor
            {
                get
                {
                    return new RenderScriptDescriptor
                    {
                        Guid = new Guid("AC6F46E2-C04E-4A20-AF68-EFA8A6CA7FCD"),
                        Name = "SuperChromaRes",
                        Description = "SuperChromaRes chroma scaling",
                        Copyright = "SuperChromaRes by Shiandow",
                    };
                }
            }
        }
    }
}
