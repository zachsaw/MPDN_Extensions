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

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Mpdn.Extensions.CustomLinearScalers.Functions;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript;
using SharpDX;
using WeightFilter = Mpdn.Extensions.Framework.RenderChain.TextureSourceFilter<Mpdn.ISourceTexture>;

namespace Mpdn.Extensions.RenderScripts
{
    namespace Mpdn.EwaScaler
    {
        public class EwaScaler : RenderChain
        {
            private const int BASE_DATA_POINTS = 8;

            private ISourceTexture[] m_Weights;
            private ICustomLinearScaler m_Scaler;

            #region Settings

            public ScalerTaps TapCount { get; set; }
            public bool AntiRingingEnabled { get; set; }
            public float AntiRingingStrength { get; set; }

            public ICustomLinearScaler Scaler
            {
                get { return m_Scaler; }
                set
                {
                    if (value == null)
                        return;

                    m_Scaler = value; 
                }
            }

            public EwaScaler()
            {
                TapCount = ScalerTaps.Four;
                AntiRingingEnabled = false;
                AntiRingingStrength = 0.85f;
                Scaler = new JincScaler();
            }

            #endregion

            public override string Active()
            {
                return string.Format("EWA {0}", m_Scaler.Name + TapCount.ToInt() + (AntiRingingEnabled ? "AR" : ""));
            }

            protected override string ShaderPath
            {
                get { return "EwaScaler"; }
            }

            protected override IFilter CreateFilter(IFilter input)
            {
                DiscardTextures();

                var sourceSize = input.OutputSize;
                if (!IsUpscalingFrom(sourceSize))
                    return input;

                var targetSize = Renderer.TargetSize;
                CreateWeights((Size) sourceSize, targetSize);

                int lobes = TapCount.ToInt()/2;
                var shader = CompileShader("EwaScaler.hlsl",
                    macroDefinitions:
                        string.Format("LOBES = {0}; AR = {1}",
                            lobes, AntiRingingEnabled ? 1 : 0))
                    .Configure(
                        transform: size => targetSize,
                        arguments: new[] {AntiRingingStrength},
                        linearSampling: true
                    );

                return GetEwaFilter(shader, new[] {input});
            }

            private static double GetScaleFactor(int dest, int source)
            {
                return Math.Log(dest/(double)source, 2);
            }

            protected IFilter GetEwaFilter(ShaderFilterSettings<IShader> shader, IFilter[] inputs)
            {
                var filters = m_Weights.Select(w => new WeightFilter(w));
                return new ShaderFilter(shader, inputs.Concat((IEnumerable<IFilter<IBaseTexture>>) filters).ToArray());
            }

            public override void Reset()
            {
                DiscardTextures();

                base.Reset();
            }

            protected void DiscardTextures()
            {
                DiscardWeights(ref m_Weights);
            }

            private static void DiscardWeights(ref ISourceTexture[] weights)
            {
                if (weights == null) 
                    return;

                foreach (var w in weights)
                {
                    DisposeHelper.Dispose(w);
                }
                weights = null;
            }

            private static double GetDistance(double point1, double point2)
            {
                return Math.Sqrt(point1*point1 + point2*point2);
            }

            protected void CreateWeights(TextureSize sourceSize, TextureSize targetSize)
            {
                if (m_Weights != null)
                    return;

                double scaleFactorX = GetScaleFactor(targetSize.Width, sourceSize.Width);
                double scaleFactorY = GetScaleFactor(targetSize.Height, sourceSize.Height);

                var tapCount = TapCount.ToInt();
                int lobes = tapCount / 2;
                m_Weights = new ISourceTexture[lobes];
                var dataPointsX = GetDataPointCount(scaleFactorX);
                var dataPointsY = GetDataPointCount(scaleFactorY);
                var channels = lobes == 2 ? 2 : 4;
                var data = new Half[dataPointsY, dataPointsX * channels];
                for (int z = 0; z < lobes; z++)
                {
                    for (int y = 0; y < dataPointsY; y++)
                    {
                        for (int x = 0; x < dataPointsX; x++)
                        {
                            var offsetX = x/(double) dataPointsX;
                            var offsetY = y/(double) dataPointsX;

                            for (int i = 0; i < lobes; i++)
                            {
                                var distance = GetDistance(i + offsetX, z + offsetY);
                                data[y, x*channels + i] = GetWeight(distance, tapCount);
                            }
                        }
                    }
                    m_Weights[z] = Renderer.CreateTexture(dataPointsX, dataPointsY,
                        channels == 2 ? TextureFormat.Float16_RG : TextureFormat.Float16);
                    Renderer.UpdateTexture(m_Weights[z], data);
                }
            }

            private float GetWeight(double distance, int width)
            {
                return Scaler.GetWeight((float) distance, width);
            }

            private static int GetDataPointCount(double scaleFactorX)
            {
                return Math.Max(BASE_DATA_POINTS / 2, (int)(BASE_DATA_POINTS * scaleFactorX));
            }

            #region Jinc CustomLinearScaler

            public class JincScaler : ICustomLinearScaler
            {
                public float GetWeight(float n, int width)
                {
                    return (float) (Jinc.CalculateJinc(n)*Jinc.CalculateWindow(n, width/2));
                }

                public Guid Guid { get { return new Guid("55263FDE-A57D-4695-BF5A-80CE74994BED"); } }
                public string Name { get { return "Jinc"; } }
                public bool AllowDeRing { get { return true; } }
                public ScalerTaps MinTapCount { get { return ScalerTaps.Four; } }
                public ScalerTaps MaxTapCount { get { return ScalerTaps.Sixteen; } }
            }

            #endregion
        }

        public class EwaScalerScaler : RenderChainUi<EwaScaler, EwaScalerConfigDialog>
        {
            protected override string ConfigFileName
            {
                get { return "Mpdn.EwaScaler"; }
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
                        Name = "EWA Scaler",
                        Description = "Elliptical weighted average (EWA) image upscaler"
                    };
                }
            }
        }
    }
}
