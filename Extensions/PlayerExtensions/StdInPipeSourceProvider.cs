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
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using DirectShowLib;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class StdInPipeSourceProvider : PlayerExtension
    {
        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("F00F32AA-81BD-44EE-9543-E5EB9E53F16E"),
                    Name = "StdIn Pipe Source Provider",
                    Description = "Uses LAV Splitter Source to read source from std::in pipe"
                };
            }
        }

        public override void Initialize()
        {
            Media.Loading += OnMediaLoading;
        }

        public override void Destroy()
        {
            Media.Loading -= OnMediaLoading;
        }

        private static void OnMediaLoading(object sender, MediaLoadingEventArgs e)
        {
            if (e.Filename.Trim() == "-")
            {
                // Note: This assumes MPDN uses LAV Splitter Source
                e.Filename = "pipe:0";
            }
        }
    }
}
