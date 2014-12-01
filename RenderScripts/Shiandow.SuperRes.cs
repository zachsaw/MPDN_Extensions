using System;
using System.Drawing;
using Mpdn.RenderScript.Scaler;
using YAXLib;

namespace Mpdn.RenderScript
{
    namespace Shiandow.Nedi
    {
        public class SuperRes : RenderChain
        {
            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public int Passes { get; set; }

            public IScaler upscaler, downscaler; // Not saved
            public Func<Size> TargetSize; // Not saved

            public SuperRes()
            {
                TargetSize = () => Renderer.TargetSize;
                Passes = 3;
                upscaler = new Scaler.Jinc(ScalerTaps.Four, false);
                downscaler = new Scaler.Bilinear();
            }

            public override IFilter CreateFilter(IFilter sourceFilter)
            {
                return CreateFilter(sourceFilter, new ResizeFilter(sourceFilter, TargetSize()));
            }

            public IFilter CreateFilter(IFilter original, IFilter initial)
            {
                IFilter lab, linear;

                var inputSize = original.OutputSize;
                var targetSize = TargetSize();

                var Diff = CompileShader("Diff.hlsl");
                var SuperRes = CompileShader("SuperRes.hlsl");
                var AntiRinging = CompileShader("AntiRinging.hlsl");

                var GammaToLab = CompileShader("GammaToLab.hlsl");
                var LabToGamma = CompileShader("LabToGamma.hlsl");
                var LinearToGamma = CompileShader("LinearToGamma.hlsl");
                var GammaToLinear = CompileShader("GammaToLinear.hlsl");
                var LabToLinear = CompileShader("LabToLinear.hlsl");
                var LinearToLab = CompileShader("LinearToLab.hlsl");

                // Skip if downscaling
                if (targetSize.Width  <= inputSize.Width && targetSize.Height <= inputSize.Height)
                    return original;

                // Initial scaling
                linear   = new ShaderFilter(GammaToLinear, initial);
                original = new ShaderFilter(GammaToLab, original);

                for (int i = 1; i <= Passes; i++)
                {
                    IFilter res, diff;                  

                    // Calculate difference
                    res = new ResizeFilter(linear, inputSize, upscaler, downscaler); // Downscale result
                    diff = new ShaderFilter(Diff, res, original);               // Compare with original
                    diff = new ResizeFilter(diff, targetSize, upscaler, downscaler);  // Scale to output size

                    // Update result
                    lab = new ShaderFilter(LinearToLab, linear);
                    linear = new ShaderFilter(SuperRes, lab, diff, original);
                }

                return new ShaderFilter(LinearToGamma, linear);
            }
        }

        public class SuperNEDIRes : RenderChain
        {
            private IScaler m_ShiftedScaler;

            public SuperNEDIRes()
            {
                m_ShiftedScaler = new Scaler.Custom(new ShiftedScaler(0.5f), ScalerTaps.Six, false);
            }

            protected override string ShaderPath
            {
                get { return "SuperRes"; }
            }

            public override IFilter CreateFilter(IFilter sourceFilter)
            {
                var inputSize = sourceFilter.OutputSize;
                var targetSize = Renderer.TargetSize;
                var nedi = new Nedi{ AlwaysDoubleImage = false }.CreateFilter(sourceFilter);

                // Skip if downscaling
                if (targetSize.Width <= inputSize.Width && targetSize.Height <= inputSize.Height)
                    return sourceFilter;

                IScaler upscaler;
                if (targetSize.Width < 1.7 * inputSize.Width)
                    upscaler = new Scaler.Bilinear();
                else
                    upscaler = new Scaler.Jinc(ScalerTaps.Four, false);

                var shifted = new ResizeFilter(nedi, Renderer.TargetSize, m_ShiftedScaler, m_ShiftedScaler);
                var superres =  new SuperRes{ Passes = 2, upscaler = upscaler }.CreateFilter(sourceFilter, shifted);

                // Skip if downscaling
                if (targetSize.Width <= inputSize.Width && targetSize.Height <= inputSize.Height)
                    return sourceFilter;
                else
                    return superres;
            }

            private class ShiftedScaler : ICustomLinearScaler
            {
                private float m_Offset;

                public ShiftedScaler(float offset)
                {
                    m_Offset = offset;
                }

                public Guid Guid
                {
                    get { return new Guid(); }
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
                    get { return ScalerTaps.Six; }
                } 

                public float GetWeight(float n, int width)
                {
                    return (float)Kernel(n + m_Offset, width);
                }

                private static double Kernel(double x, double radius)
                {
                    return Math.Exp(-x * x);
                    /*x = Math.Abs(x);
                    var B = 1.0;
                    var C = 0.0;

                    if (x > 2.0) 
                        return 0;
                    else if (x <= 1.0)
                        return ((2-1.5*B-C)*x + (-3+2*B+C))*x*x + (1-B/3.0);
                    else
                        return (((-B / 6.0 - C) * x + (B + 5 * C)) * x + (-2 * B - 8 * C)) * x + ((4.0 / 3.0) * B + 4 * C);*/
                }
            }
        }

        public class SuperChromaRes : RenderChain
        {
            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public int passes { get; set; }
            private IScaler upscaler, downscaler;

            public SuperChromaRes()
            {
                passes = 2;
                upscaler = new Scaler.Bilinear();
                downscaler = new Scaler.Bilinear();
            }

            protected override string ShaderPath
            {
                get { return "SuperRes"; }
            }

            public override IFilter CreateFilter(IFilter sourceFilter)
            {
                IFilter linear, original;

                var chromaSize = Renderer.ChromaSize;
                var targetSize = sourceFilter.OutputSize;

                var Diff = CompileShader("SuperChromaRes/Diff.hlsl");
                var CopyLuma = CompileShader("SuperChromaRes/CopyLuma.hlsl");
                var CopyChroma = CompileShader("SuperChromaRes/CopyChroma.hlsl");
                var SuperRes = CompileShader("SuperChromaRes/SuperRes.hlsl");
                var AntiRinging = CompileShader("SuperChromaRes/AntiRinging.hlsl");

                var GammaToLab = CompileShader("GammaToLab.hlsl");
                var LabToGamma = CompileShader("LabToGamma.hlsl");
                var LinearToGamma = CompileShader("LinearToGamma.hlsl");
                var GammaToLinear = CompileShader("GammaToLinear.hlsl");
                var LabToLinear = CompileShader("LabToLinear.hlsl");
                var LinearToLab = CompileShader("LinearToLab.hlsl");

                // Initial scaling
                original = sourceFilter.ConvertToYuv();
                linear = new ShaderFilter(GammaToLinear, new RgbFilter(original));

                // Original values
                var uInput = new USourceFilter();
                var vInput = new VSourceFilter();

                for (int i = 1; i <= passes; i++)
                {
                    IFilter lab, yuv, rgb, res, diff;

                    // Compare to chroma
                    res = new ResizeFilter(linear, chromaSize, upscaler, downscaler);
                    yuv = new ShaderFilter(LinearToGamma, res).ConvertToYuv();
                    rgb = new ShaderFilter(CopyChroma, yuv, uInput, vInput).ConvertToRgb();
                    diff = new ShaderFilter(Diff, res, rgb);
                    diff = new ResizeFilter(diff, targetSize, upscaler, downscaler);

                    // Update result
                    lab = new ShaderFilter(LinearToLab, linear);
                    yuv = new ShaderFilter(SuperRes, lab, diff).ConvertToYuv();

                    // Anti ringing
                    rgb = new ShaderFilter(AntiRinging, yuv, original, uInput, vInput).ConvertToRgb();
                    linear = new ShaderFilter(GammaToLinear, rgb);
                }

                return new ShaderFilter(LinearToGamma, linear);
            }
        }

        public class SuperResUi : RenderChainUi<SuperRes>
        {
            protected override RenderScriptDescriptor ScriptDescriptor
            {
                get
                {
                    return new RenderScriptDescriptor
                    {
                        Guid = new Guid("3E7C670C-EFFB-41EB-AC19-207E650DEBD0"),
                        Name = "SuperRes",
                        Description = "SuperRes image scaling",
                        Copyright = "SuperRes by Shiandow",
                    };
                }
            }
        }

        public class SuperNEDIResUi : RenderChainUi<SuperNEDIRes>
        {
            protected override RenderScriptDescriptor ScriptDescriptor
            {
                get
                {
                    return new RenderScriptDescriptor
                    {
                        Guid = new Guid("24E09DA6-EBDF-4980-A360-0F62B083BAA2"),
                        Name = "SuperNEDIRes",
                        Description = "Combines NEDI with SuperRes",
                        Copyright = "SuperNEDIRes by Shiandow",
                    };
                }
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
