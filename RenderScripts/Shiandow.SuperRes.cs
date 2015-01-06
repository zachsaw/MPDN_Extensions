using System;
using System.Drawing;
using Mpdn.RenderScript.Scaler;
using SharpDX;
using YAXLib;
using Mpdn.RenderScript.Shiandow.Nedi;
using Mpdn.CustomLinearScaler.Example;

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
                Sharpness = 0.85f;
                AntiAliasing = 0.65f;
                AntiRinging = 0.8f;

                UseNEDI = false;
                FirstPassOnly = false;
                //upscaler = new Scaler.Bicubic(2.0f/3.0f, false);
                //upscaler = new Scaler.Custom(new Gaussian(), ScalerTaps.Four, false);
                upscaler = new Scaler.Jinc(ScalerTaps.Four, false);
                downscaler = new Scaler.Bilinear();
            }

            public override IFilter CreateFilter(IResizeableFilter sourceFilter)
            {
                var inputSize = sourceFilter.OutputSize;
                var targetSize = TargetSize();

                IResizeableFilter initial;
                if (UseNEDI)
                {
                    var nedi = new Shiandow.Nedi.Nedi { AlwaysDoubleImage = false, Centered = false }.CreateFilter(sourceFilter);
                    initial = new ResizeFilter(nedi, targetSize, m_ShiftedScaler, m_ShiftedScaler);
                }
                else
                {
                    initial = new ResizeFilter(sourceFilter, inputSize); // SetSize must *not* affect original. 
                }

                // Skip if downscaling
                if (targetSize.Width <= inputSize.Width && targetSize.Height <= inputSize.Height)
                    return sourceFilter;
                else
                    return CreateFilter(sourceFilter, initial);
            }

            public IFilter CreateFilter(IFilter original, IResizeableFilter initial)
            {
                IFilter lab, linear, result = initial;

                var inputSize = original.OutputSize;
                var currentSize = initial.OutputSize;
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
                lab = new ShaderFilter(GammaToLab, initial);
                original = new ShaderFilter(GammaToLab, original);

                float AA = AntiAliasing;
                for (int i = 1; i <= Passes; i++)
                {
                    IFilter res, diff;

                    // Initial calculations
                    bool useBilinear = (upscaler is Scaler.Bilinear) || (FirstPassOnly && !(i == 1));
                    /*
                    if (i == Passes) currentSize = targetSize;
                    else currentSize = CalculateSize(currentSize, targetSize, i);
                     */
                    currentSize = targetSize;
                    
                    // Resize and Convert
                    lab = new ResizeFilter(lab, currentSize);
                    linear = new ShaderFilter(LabToLinear, lab);

                    // Calculate difference
                    res = new ResizeFilter(linear, inputSize, upscaler, downscaler); // Downscale result
                    diff = new ShaderFilter(Diff, res, original);                    // Compare with original
                    if (!useBilinear)
                        diff = new ResizeFilter(diff, currentSize, upscaler, downscaler); // Scale to output size
                    
                    // Update result
                    lab = new ShaderFilter(SuperRes, useBilinear, new[] { Strength, Sharpness, AA, AntiRinging }, lab, diff, original);
                    result = new ShaderFilter(LabToGamma, lab);

                    AA *= 0.65f;
                }

                return result;
            }

            private Size CalculateSize(Size sizeA, Size sizeB, int k)
            {
                int minW = sizeA.Width; int minH = sizeA.Height;
                int maxW = sizeB.Width; int maxH = sizeB.Height;

                int steps = (int)Math.Floor(Math.Log((double)(maxH * maxW) / (double)(minH * minW)) / (2 * Math.Log(1.5)));
                if (steps < 1) steps = 1;

                double w = minW * Math.Pow((double)maxW / (double)minW, (double)Math.Min(k, steps) / (double)steps);
                double h = minW * Math.Pow((double)maxH / (double)minH, (double)Math.Min(k, steps) / (double)steps);

                return new Size(Math.Min(maxW, Math.Max(minW, (int)Math.Round(w))),
                                Math.Min(maxH, Math.Max(minH, (int)Math.Round(h))));
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
