using System;

namespace Mpdn.Extensions.Framework.RenderChain.Shaders
{
    public interface IShaderParameters
    {
        bool LinearSampling { get; }
        bool[] PerTextureLinearSampling { get; }
        Func<TextureSize, TextureSize> Transform { get; }
        TextureFormat Format { get; }
        int SizeIndex { get; }
        ArgumentList Arguments { get; }
        ArgumentList.Entry this[string identifier] { get; }
    }

    public interface IComputeShaderParameters : IShaderParameters
    {
        int ThreadGroupX { get; }
        int ThreadGroupY { get; }
        int ThreadGroupZ { get; }
    }

    public interface IClKernelParameters : IShaderParameters
    {
        int[] GlobalWorkSizes { get; }
        int[] LocalWorkSizes { get; }
    }
}