using System;
using System.Drawing;
using Mpdn.RenderScript.Scaler;
using SharpDX;
using YAXLib;
using Mpdn.RenderScript.Shiandow.Nedi;

namespace Mpdn.RenderScript
{
    namespace Shiandow.SuperRes
    {
        public class SuperRes : RenderChain
        {
            #region Settings

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public int Passes { get; set; }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public float Strength { get; set; }
            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public float Sharpness { get; set; }
            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public float AntiAliasing { get; set; }
            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public float AntiRinging { get; set; }

            [YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public bool UseNEDI { get; set; }
            //[YAXErrorIfMissed(YAXExceptionTypes.Ignore)]
            public bool FirstPassOnly;// { get; set; }

            #endregion

            public Func<Size> TargetSize; // Not saved
            private IScaler downscaler, upscaler;

            public SuperRes()
            {
                TargetSize = () => Renderer.TargetSize;
                m_ShiftedScaler = new Scaler.Custom(new ShiftedScaler(0.5f), ScalerTaps.Six, false);

                Passes = 2;

                Strength = 0.8f;
                Sharpness = 0.5f;
                AntiAliasing = 1.0f;
                AntiRinging = 0.8f;

                UseNEDI = false;
                FirstPassOnly = false;
                upscaler = new Scaler.Bicubic(2.0f/3.0f, false);
                downscaler = new Scaler.Bilinear();
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                var inputSize = sourceFilter.OutputSize;
                var targetSize = TargetSize();

                IFilter initial;
                if (UseNEDI)
                {
                    initial = new Shiandow.Nedi.Nedi { AlwaysDoubleImage = false, Centered = false }.CreateFilter(sourceFilter);
                    initial = new ResizeFilter(initial, targetSize, m_ShiftedScaler, m_ShiftedScaler);
                }
                else
                {
                    initial = new ResizeFilter(sourceFilter, targetSize);
                }

                // Skip if downscaling
                if (targetSize.Width <= inputSize.Width && targetSize.Height <= inputSize.Height)
                    return sourceFilter;
                else
                    return CreateFilter(sourceFilter, initial);
            }

            public IFilter CreateFilter(IFilter original, IFilter initial)
            {
                IFilter lab, linear;

                var inputSize = original.OutputSize;
                var targetSize = TargetSize();

                var Diff = CompileShader("Diff.hlsl");
                var SuperRes = CompileShader("SuperRes.hlsl");
                var ARShader = CompileShader("AntiRinging.hlsl");

                var GammaToLab = CompileShader("GammaToLab.hlsl");
                var LabToGamma = CompileShader("LabToGamma.hlsl");
                var LinearToGamma = CompileShader("LinearToGamma.hlsl");
                var GammaToLinear = CompileShader("GammaToLinear.hlsl");
                var LabToLinear = CompileShader("LabToLinear.hlsl");
                var LinearToLab = CompileShader("LinearToLab.hlsl");

                // Initial scaling
                linear   = new ShaderFilter(GammaToLinear, initial);
                original = new ShaderFilter(GammaToLab, original);

                float AA = AntiAliasing;
                for (int i = 1; i <= Passes; i++)
                {
                    IFilter res, diff;
                    bool useBilinear = (upscaler is Scaler.Bilinear) || (FirstPassOnly && !(i == 1));

                    // Calculate difference
                    res = new ResizeFilter(linear, inputSize, upscaler, downscaler); // Downscale result
                    diff = new ShaderFilter(Diff, res, original);                    // Compare with original
                    if (!useBilinear)
                        diff = new ResizeFilter(diff, targetSize, upscaler, downscaler); // Scale to output size

                    // Update result
                    lab = new ShaderFilter(LinearToLab, linear);
                    linear = new ShaderFilter(SuperRes, useBilinear, new[] { Strength, Sharpness, AA, AntiRinging }, lab, diff, original);

                    AA *= 0.5f;
                }

                return new ShaderFilter(LinearToGamma, linear);
            }

            private IScaler m_ShiftedScaler;

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
                    x = Math.Abs(x);
                    var B = 1.0/3.0;
                    var C = 1.0/3.0;

                    if (x > 2.0)
                        return 0;
                    else if (x <= 1.0)
                        return ((2 - 1.5 * B - C) * x + (-3 + 2 * B + C)) * x * x + (1 - B / 3.0);
                    else
                        return (((-B / 6.0 - C) * x + (B + 5 * C)) * x + (-2 * B - 8 * C)) * x + ((4.0 / 3.0) * B + 4 * C);
                }
            }
        }

        public class SuperResUi : ConfigurableRenderChainUi<SuperRes, SuperResConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Shiandow.SuperRes"; }
            }

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
    }
}
