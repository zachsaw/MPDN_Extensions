// This file is a part of MPDN Extensions.
// https://github.com/zachsaw/MPDN_Extensions
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library.

//#define USE_LIBRETRO_APPROX_JINC

using System;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript;
using TransformFunc = System.Func<System.Drawing.Size, System.Drawing.Size>;
using WeightFilter = Mpdn.Extensions.Framework.RenderChain.TextureSourceFilter<Mpdn.ISourceTexture>;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.Jinc2D
    {
        public class Jinc2D : RenderChain
        {
            private ISourceTexture[] m_Weights;

            #region Settings

            public ScalerTaps TapCount { get; set; }
            public bool AntiRingingEnabled { get; set; }
            public float AntiRingingStrength { get; set; }

            public Jinc2D()
            {
                TapCount = ScalerTaps.Four;
                AntiRingingEnabled = false;
                AntiRingingStrength = 0.85f;
            }

            #endregion

            protected override string ShaderPath
            {
                get { return "Jinc2D"; }
            }

            public override IFilter CreateFilter(IFilter input)
            {
                var sourceSize = input.OutputSize;
                if (!IsUpscalingFrom(sourceSize))
                    return input;

                CreateWeights();

                var shader = CompileShader("Jinc2D.hlsl",
                    macroDefinitions:
                        string.Format("AR = {0}; AR_STRENGTH = {1}", AntiRingingEnabled ? 1 : 0, AntiRingingStrength))
                    .Configure(linearSampling: false, transform: size => Renderer.TargetSize);

                return new ShaderFilter(shader, input,
                    new WeightFilter(m_Weights[0]),
                    new WeightFilter(m_Weights[1]),
                    new WeightFilter(m_Weights[2]),
                    new WeightFilter(m_Weights[3]));
            }

            public override void Reset()
            {
                DiscardTextures();

                base.Reset();
            }

            private void DiscardTextures()
            {
                if (m_Weights == null)
                    return;

                foreach (var w in m_Weights)
                {
                    DisposeHelper.Dispose(w);
                }
                m_Weights = null;
            }

#if USE_LIBRETRO_APPROX_JINC
            private double GetApproxJincWeight(double dist)
            {
                const double jinc2WindowSinc = 0.44;
                const double jinc2Sinc = 0.82;
                const double wa = jinc2WindowSinc*Math.PI;
                const double wb = jinc2Sinc*Math.PI;

                return dist < 1e-8 ? wa*wb : Math.Sin(dist*wa)*Math.Sin(dist*wb)/(dist*dist);
            }
#endif

            private float GetWeight(double dist)
            {
                var lobes = TapCount.ToInt()/2;

#if USE_LIBRETRO_APPROX_JINC
                if (lobes == 2)
                {
                    return (float) GetApproxJincWeight(dist);
                }
#endif
                return (float) Jinc.GetWeight(dist, lobes);
            }

            private static double GetDistance(double point1, double point2)
            {
                return Math.Sqrt(point1*point1 + point2*point2);
            }

            private void CreateWeights()
            {
                if (m_Weights != null)
                    return;

                const int dataPoints = 24; // ~2.5% error in high contrast edges
                // Note: 
                //    Increase by steps of 4 if you want better quality (less error vs mathematical model)
                //    At 32 data points, you'll reduce the error to ~2% but it's probably not worth the GPU load overhead

                // For 2 lobed Jinc, we need 4 2D LUTs
                int tapCount = TapCount.ToInt();
                m_Weights = new ISourceTexture[tapCount];
                var data = new float[dataPoints, dataPoints*4];
                for (int z = 0; z < tapCount; z++)
                {
                    for (int y = 0; y < dataPoints; y++)
                    {
                        for (int x = 0; x < dataPoints; x++)
                        {
                            var offsetX = x/(float) dataPoints;
                            var offsetY = y/(float) dataPoints;
                            int topLeft = -tapCount/2 + 1;
                            data[y, x*4 + 0] = GetWeight(GetDistance(topLeft + 0 - offsetX, topLeft + z - offsetY));
                            data[y, x*4 + 1] = GetWeight(GetDistance(topLeft + 1 - offsetX, topLeft + z - offsetY));
                            data[y, x*4 + 2] = GetWeight(GetDistance(topLeft + 2 - offsetX, topLeft + z - offsetY));
                            data[y, x*4 + 3] = GetWeight(GetDistance(topLeft + 3 - offsetX, topLeft + z - offsetY));
                        }
                    }
                    m_Weights[z] = Renderer.CreateTexture(dataPoints, dataPoints, TextureFormat.Float32);
                    Renderer.UpdateTexture(m_Weights[z], data);
                }
            }
        }

        public class Jinc
        {
            // see https://github.com/AviSynth/jinc-resize/blob/master/JincResize/JincFilter.cpp

            private const double JINC_PI = 3.141592653589793238462643;

            private static readonly double[] s_JincZeros =
            {
                1.2196698912665045,
                2.2331305943815286,
                3.2383154841662362,
                4.2410628637960699,
                5.2427643768701817,
                6.2439216898644877,
                7.2447598687199570,
                8.2453949139520427,
                9.2458926849494673,
                10.246293348754916,
                11.246622794877883,
                12.246898461138105,
                13.247132522181061,
                14.247333735806849,
                15.247508563037300,
                16.247661874700962
            };

            private static double J1(double x)
            {
                double[]
                    p1 =
                    {
                        0.581199354001606143928050809e+21,
                        -0.6672106568924916298020941484e+20,
                        0.2316433580634002297931815435e+19,
                        -0.3588817569910106050743641413e+17,
                        0.2908795263834775409737601689e+15,
                        -0.1322983480332126453125473247e+13,
                        0.3413234182301700539091292655e+10,
                        -0.4695753530642995859767162166e+7,
                        0.270112271089232341485679099e+4
                    },
                    q1 =
                    {
                        0.11623987080032122878585294e+22,
                        0.1185770712190320999837113348e+20,
                        0.6092061398917521746105196863e+17,
                        0.2081661221307607351240184229e+15,
                        0.5243710262167649715406728642e+12,
                        0.1013863514358673989967045588e+10,
                        0.1501793594998585505921097578e+7,
                        0.1606931573481487801970916749e+4,
                        0.1e+1
                    };

                double p = p1[8];
                double q = q1[8];

                for (int i = 7; i >= 0; i--)
                {
                    p = p * x * x + p1[i];
                    q = q * x * x + q1[i];
                }

                return p / q;
            }

            private static double P1(double x)
            {
                double[]
                    p1 =
                    {
                        0.352246649133679798341724373e+5,
                        0.62758845247161281269005675e+5,
                        0.313539631109159574238669888e+5,
                        0.49854832060594338434500455e+4,
                        0.2111529182853962382105718e+3,
                        0.12571716929145341558495e+1
                    },
                    q1 =
                    {
                        0.352246649133679798068390431e+5,
                        0.626943469593560511888833731e+5,
                        0.312404063819041039923015703e+5,
                        0.4930396490181088979386097e+4,
                        0.2030775189134759322293574e+3,
                        0.1e+1
                    };

                double p = p1[5];
                double q = q1[5];

                for (int i = 4; i >= 0; i--)
                {
                    p = p * (8.0 / x) * (8.0 / x) + p1[i];
                    q = q * (8.0 / x) * (8.0 / x) + q1[i];
                }

                return p / q;
            }

            private static double Q1(double x)
            {
                double[]
                    p1 =
                    {
                        0.3511751914303552822533318e+3,
                        0.7210391804904475039280863e+3,
                        0.4259873011654442389886993e+3,
                        0.831898957673850827325226e+2,
                        0.45681716295512267064405e+1,
                        0.3532840052740123642735e-1
                    },
                    q1 =
                    {
                        0.74917374171809127714519505e+4,
                        0.154141773392650970499848051e+5,
                        0.91522317015169922705904727e+4,
                        0.18111867005523513506724158e+4,
                        0.1038187585462133728776636e+3,
                        0.1e+1
                    };

                double p = p1[5];
                double q = q1[5];

                for (int i = 4; i >= 0; i--)
                {
                    p = p * (8.0 / x) * (8.0 / x) + p1[i];
                    q = q * (8.0 / x) * (8.0 / x) + q1[i];
                }

                return p / q;
            }

            private static double BesselOrderOne(double x)
            {
                if (x < 1e-8)
                    return 0.0;

                double p = x;

                if (x < 0.0)
                {
                    x = -x;
                }

                if (x < 8.0)
                    return p * J1(x);

                double q = (Math.Sqrt(2.0 / (JINC_PI * x)) *
                            (P1(x) * (1.0 / Math.Sqrt(2.0) * (Math.Sin(x) - Math.Cos(x))) -
                             8.0 / x * Q1(x) * (-1.0 / Math.Sqrt(2.0) * (Math.Sin(x) + Math.Cos(x)))));

                if (p < 0.0)
                {
                    q = -q;
                }

                return q;
            }

            private static double Jinc1(double x)
            {
                return x < 1e-8 ? Math.PI / 2.0f : BesselOrderOne(Math.PI * x) / x;
            }

            private static double JincWindowFactor(int lobes)
            {
                return s_JincZeros[0] / JincSupportFactor(lobes);
            }

            private static double JincSupportFactor(int lobes)
            {
                return s_JincZeros[Math.Min(16, lobes) - 1];
            }

            public static double GetWeight(double dist, int lobes)
            {
                if (dist > JincSupportFactor(lobes))
                    return 0;

                return Jinc1(dist) * Jinc1(dist * JincWindowFactor(lobes));
            }
        }

        public class ChromaScaler : RenderChainUi<Jinc2D, Jinc2DConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.Jinc2D"; }
            }

            public override string Category
            {
                get { return "Upscaling"; }
            }

            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("317F3D2F-8A0C-4FA3-AF71-41BD4DDC6FC8"),
                        Name = "Jinc2D",
                        Description = "Jinc (cylindrical) image upscaler"
                    };
                }
            }
        }
    }
}
