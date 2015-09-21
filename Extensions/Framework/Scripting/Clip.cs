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
// 

using System;
using System.Drawing;
using Mpdn.RenderScript;
using SharpDX;
using Point = System.Drawing.Point;

namespace Mpdn.Extensions.Framework.Scripting
{
    public abstract class Clip
    {
        public abstract Size InputSize { get; }

        #region Renderer Properties

        public virtual string FileName
        {
            get { return Renderer.VideoFileName; }
        }

        public virtual bool Interlaced
        {
            get { return Renderer.InterlaceFlags.HasFlag(InterlaceFlags.IsInterlaced); }
        }

        public virtual Size TargetSize
        {
            get { return Renderer.TargetSize; }
        }

        public virtual Size SourceSize
        {
            get { return Renderer.VideoSize; }
        }

        public virtual Size LumaSize
        {
            get { return Renderer.LumaSize; }
        }

        public virtual Size ChromaSize
        {
            get { return Renderer.ChromaSize; }
        }

        public virtual Vector2 ChromaOffset
        {
            get { return Renderer.ChromaOffset; }
        }

        public virtual Point AspectRatio
        {
            get { return Renderer.AspectRatio; }
        }

        public virtual YuvColorimetric Colorimetric
        {
            get { return Renderer.Colorimetric; }
        }

        public virtual FrameBufferInputFormat InputFormat
        {
            get { return Renderer.InputFormat; }
        }

        public virtual double FrameRateHz
        {
            get { return Renderer.FrameRateHz; }
        }

        #endregion

        #region Automatic Properties

        private readonly RenderChain.RenderChain m_Chain = RenderChain.RenderChain.Identity;

        public bool NeedsUpscaling
        {
            get { return m_Chain.IsUpscalingFrom(InputSize); }
        }

        public bool NeedsDownscaling
        {
            get { return m_Chain.IsDownscalingFrom(InputSize); }
        }

        public double ScalingFactor
        {
            get { return Math.Sqrt(ScalingFactorX * ScalingFactorY); }
        }

        public double ScalingFactorX
        {
            get { return TargetSize.Width / (double)InputSize.Width; }
        }

        public double ScalingFactorY
        {
            get { return TargetSize.Height / (double)InputSize.Height; }
        }

        public int SourceBitDepth
        {
            get { return InputFormat.GetBitDepth(); }
        }

        public bool SourceRgb
        {
            get { return InputFormat.IsRgb(); }
        }

        public bool SourceYuv
        {
            get { return InputFormat.IsYuv(); }
        }

        #endregion
    }
}
