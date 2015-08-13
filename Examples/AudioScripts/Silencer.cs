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
using Mpdn.Extensions.Framework;

namespace Mpdn.Examples.AudioScripts
{
    public class Silencer : Extensions.Framework.AudioScript
    {
        private const int CHANNEL_TO_SILENT = 0;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("93C32135-7F75-44EE-91C9-6BDF905D76BB"),
                    Name = "Silencer",
                    Description = "Simple audio silencer example"
                };
            }
        }

        protected override bool SupportBitStreaming
        {
            get { return false; }
        }

        protected override void Process(float[,] samples)
        {
            var length = samples.GetLength(1);
            for (int i = 0; i < length; i++)
            {
                samples[CHANNEL_TO_SILENT, i] = 0;
            }
        }
    }
}
