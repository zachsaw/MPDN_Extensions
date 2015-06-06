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

using System.Drawing;
using SharpDX;
using Point = System.Drawing.Point;

namespace Mpdn.Extensions.Framework.Scripting
{
    public interface IClip
    {
        string FileName { get; }
        bool Interlaced { get; }
        bool NeedsUpscaling { get; }
        bool NeedsDownscaling { get; }
        Size TargetSize { get; }
        Size SourceSize { get; }
        Size LumaSize { get; }
        Size ChromaSize { get; }
        Vector2 ChromaOffset { get; }
        Point AspectRatio { get; }
        YuvColorimetric Colorimetric { get; }
        FrameBufferInputFormat InputFormat { get; }
        double FrameRateHz { get; }
    }
}
