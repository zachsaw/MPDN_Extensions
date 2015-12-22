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
using Cudafy;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.AudioChain;

namespace Mpdn.Examples.AudioScripts
{
    namespace Example
    {
        public class Gain : AudioFilter
        {
            protected override void OnLoadAudioKernel()
            {
                Gpu.LoadAudioKernel(typeof (Gain));
            }

            protected override void Process(float[,] samples, short channels, int sampleCount)
            {
                const int threadCount = 512;

                // Note: samples resides on OpenCL device memory
                Gpu.Launch(threadCount, 1).Amplify(samples, 2.0f);
            }

            [Cudafy]
            public static void Amplify(GThread thread, float[,] samples, float amplication)
            {
                var channels = samples.GetLength(0);
                var sampleCount = samples.GetLength(1);

                int tid = thread.blockIdx.x;
                while (tid < sampleCount)
                {
                    for (int i = 0; i < channels; i++)
                    {
                        // The framework will clip anything that is overamplified
                        // so quality won't be the best but this is just an example
                        var s = samples[i, tid];
                        s *= amplication;
                        samples[i, tid] = s;
                    }
                    tid += thread.gridDim.x;
                }
            }
        }

        public class GainUi : AudioChainUi<StaticAudioChain<Gain>>
        {
            public override ExtensionUiDescriptor Descriptor
            {
                get
                {
                    return new ExtensionUiDescriptor
                    {
                        Guid = new Guid("02C441FE-1592-41A2-9901-5FC41DEAA3D2"),
                        Name = "Gain",
                        Description = "Amplifies audio by 2x with OpenCL"
                    };
                }
            }

            public override string Category
            {
                get { return "Volume"; }
            }
        }
    }
}