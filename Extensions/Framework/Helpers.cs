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
using System.Collections;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using Mpdn.Extensions.Framework.Chain;
using Mpdn.Extensions.Framework.Config;
using Mpdn.Extensions.Framework.RenderChain;
using Mpdn.RenderScript.Scaler;

namespace Mpdn.Extensions.Framework
{
    public static class DisposeHelper
    {
        public static void Dispose(object obj)
        {
            var disposable = obj as IDisposable;
            if (disposable == null) return;
            SafeDispose(disposable);
        }

        public static void DisposeElements<T>(ref T enumerable) where T : IEnumerable
        {
            DisposeElements(enumerable);
            enumerable = default(T);
        }

        public static void DisposeElements<T>(T enumerable) where T : IEnumerable
        {
            if (enumerable == null) return;
            foreach (var obj in enumerable.OfType<IDisposable>())
            {
                SafeDispose(obj);
            }
        }

        public static void Dispose<T>(ref T obj)
        {
            var disposable = obj as IDisposable;
            if (disposable == null) return;
            SafeDispose(disposable);
            obj = default(T);
        }

        private static void SafeDispose(IDisposable disposable)
        {
            try
            {
                disposable.Dispose();
            }
            catch (Exception ex)
            {
                // Ignore dispose exceptions (some third party libs can throw exception in Dispose)
                if (Debugger.IsAttached)
                {
                    throw;
                }
                Trace.WriteLine(ex);
            }
        }
    }

    public static class PathHelper
    {
        public static string GetDirectoryName(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException("path");
            }

            return Path.GetDirectoryName(path) ?? Path.GetPathRoot(path);
        }

        public static string GetExtension(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException("path");
            }

            var result = Path.GetExtension(path);
            if (result == null)
                throw new ArgumentException();

            return result;
        }

        public static string ExtensionsPath
        {
            get
            {
                return Path.Combine(GetDirectoryName(Assembly.GetAssembly(typeof(IPlayerExtension)).Location),
                    "Extensions");
            }
        }
    }

    public static class ConfigHelper
    {
        public static string SaveToString<TSettings>(TSettings settings)
            where TSettings : class, new()
        {
            string result;

            var config = new MemConfig<TSettings>(settings);
            return config.SaveToString(out result) ? result : null;
        }

        public static TSettings LoadFromString<TSettings>(string input)
            where TSettings : class, new()
        {
            var config = new MemConfig<TSettings>();
            return config.LoadFromString(input) ? config.Config : null;
        }
    }

    public static class PresetHelper
    {
        public static Preset<T, TScript> MakeNewPreset<T, TScript>(this IChainUi<T, TScript> renderScript, string name = null)
            where TScript : class, IScript
        {
            return renderScript.CreateNew().ToPreset();
        }

        public static Preset<T, TScript> ToPreset<T, TScript>(this IChainUi<T, TScript> renderScript, string name = null)
            where TScript : class, IScript
        {
            return new Preset<T, TScript> { Name = name ?? renderScript.Descriptor.Name, Script = renderScript };
        }
    }

    public static class EnumHelpers
    {
        public static string ToDescription(this Enum en)
        {
            var type = en.GetType();
            var enumString = en.ToString();

            var memInfo = type.GetMember(enumString);

            if (memInfo.Length > 0)
            {
                var attrs = memInfo[0].GetCustomAttributes(typeof(DescriptionAttribute), false);

                if (attrs.Length > 0)
                {
                    return ((DescriptionAttribute)attrs[0]).Description;
                }
            }

            return enumString;
        }

        public static string[] GetDescriptions<T>()
        {
            return Enum.GetValues(typeof(T)).Cast<Enum>().Select(val => val.ToDescription()).ToArray();
        }
    }

    public static class RendererHelpers
    {
        public static TextureFormat GetTextureFormat(this RenderQuality quality)
        {
            switch (quality)
            {
                case RenderQuality.MaxPerformance:
                    return TextureFormat.Unorm8;
                case RenderQuality.Performance:
                    return TextureFormat.Float16;
                case RenderQuality.Quality:
                    return TextureFormat.Unorm16;
                case RenderQuality.MaxQuality:
                    return TextureFormat.Float32;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static bool PerformanceMode(this RenderQuality quality)
        {
            switch (quality)
            {
                case RenderQuality.MaxPerformance:
                case RenderQuality.Performance:
                    return true;
                case RenderQuality.Quality:
                case RenderQuality.MaxQuality:
                    return false;
                default:
                    throw new ArgumentOutOfRangeException("quality");
            }
        }

        public static bool QualityMode(this RenderQuality quality)
        {
            return !PerformanceMode(quality);
        }

        public static bool IsFullRange(this YuvColorimetric colorimetric)
        {
            return !IsLimitedRange(colorimetric);
        }

        public static bool IsLimitedRange(this YuvColorimetric colorimetric)
        {
            switch (colorimetric)
            {
                case YuvColorimetric.FullRangePc601:
                case YuvColorimetric.FullRangePc709:
                case YuvColorimetric.FullRangePc2020:
                    return false;
                case YuvColorimetric.ItuBt601:
                case YuvColorimetric.ItuBt709:
                case YuvColorimetric.ItuBt2020:
                    return true;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static bool IsYuv(this FrameBufferInputFormat format)
        {
            return !IsRgb(format);
        }

        public static bool IsRgb(this FrameBufferInputFormat format)
        {
            switch (format)
            {
                case FrameBufferInputFormat.Rgb24:
                case FrameBufferInputFormat.Rgb32:
                case FrameBufferInputFormat.Rgb48:
                    return true;
            }

            return false;
        }

        public static int GetBitDepth(this FrameBufferInputFormat format)
        {
            switch (format)
            {
                case FrameBufferInputFormat.Nv12:
                case FrameBufferInputFormat.Yv12:
                case FrameBufferInputFormat.Yuy2:
                case FrameBufferInputFormat.Uyvy:
                case FrameBufferInputFormat.Yv24:
                case FrameBufferInputFormat.Ayuv:
                case FrameBufferInputFormat.Rgb24:
                case FrameBufferInputFormat.Rgb32:
                    return 8;
                case FrameBufferInputFormat.P010:
                case FrameBufferInputFormat.P210:
                case FrameBufferInputFormat.Y410:
                    return 10;
                case FrameBufferInputFormat.P016:
                case FrameBufferInputFormat.P216:
                case FrameBufferInputFormat.Y416:
                case FrameBufferInputFormat.Rgb48:
                    return 16;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static int ToInt(this ScalerTaps scalerTaps)
        {
            switch (scalerTaps)
            {
                case ScalerTaps.Two:
                    return 2;
                case ScalerTaps.Four:
                    return 4;
                case ScalerTaps.Six:
                    return 6;
                case ScalerTaps.Eight:
                    return 8;
                case ScalerTaps.Twelve:
                    return 12;
                case ScalerTaps.Sixteen:
                    return 16;
                default:
                    throw new ArgumentOutOfRangeException("scalerTaps");
            }
        }
        
        public static float[] GetYuvConsts(this YuvColorimetric colorimetric) 
        {
            switch (colorimetric)
            {
                case YuvColorimetric.FullRangePc601:
                case YuvColorimetric.ItuBt601:
                    return new[] {0.114f, 0.299f};
                case YuvColorimetric.FullRangePc709:
                case YuvColorimetric.ItuBt709:
                    return new[] {0.0722f, 0.2126f};
                case YuvColorimetric.FullRangePc2020:
                case YuvColorimetric.ItuBt2020:
                    return new[] {0.0593f, 0.2627f};
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }

    public static class StringHelpers
    {
        public static string SubstringIdx(this string self, int startIndex, int endIndex)
        {
            var length = endIndex - startIndex;
            if (length < 0)
            {
                length = 0;
            }
            return self.Substring(startIndex, length);
        }
    }

    public static class EventHelpers
    {
        public static void Handle<T>(this T self, Action<T> action) where T : class
        {
            if (self == null)
                return;

            action(self);
        }
    }

    public static class StatusHelpers
    {
        public static string FlattenStatus(this string status)
        {
            return status.Replace(';', ',');
        }

        public static string ToSubStatus(this string status)
        {
            return string.IsNullOrEmpty(status)
                ? ""
                : string.Format("({0})", FlattenStatus(status));
        }

        public static string AppendSubStatus(this string first, string status)
        {
            return string.IsNullOrEmpty(status)
                ? first
                : first + " " + status.ToSubStatus();
        }

        public static Func<string> AppendSubStatus(this Func<string> first, Func<string> status)
        {
            return () => first().AppendSubStatus(status());
        }

        public static string AppendStatus(this string first, string status)
        {
            return string.Join("; ",
                (new[] { first, status })
                .Where(str => !string.IsNullOrEmpty(str))
                .ToArray());
        }

        public static string PrependToStatus(this string status, string prefix)
        {
            return (string.IsNullOrEmpty(status))
                ? status
                : prefix + status;
        }

        public static Func<string> Append(this Func<string> first, Func<string> status)
        {
            return () => first().AppendStatus(status());
        }

        public static string ScaleDescription(TextureSize inputSize, TextureSize outputSize, IScaler upscaler, IScaler downscaler, IScaler convolver = null)
        {
            var xDesc = ScaleDescription(inputSize.Width, outputSize.Width, upscaler, downscaler, convolver);
            var yDesc = ScaleDescription(inputSize.Height, outputSize.Height, upscaler, downscaler, convolver);

            if (xDesc == yDesc)
                return xDesc;

            xDesc = xDesc.PrependToStatus("X:");
            yDesc = yDesc.PrependToStatus("Y:");

            return xDesc.AppendStatus(yDesc);
        }

        public static string ScaleDescription(int inputDimension, int outputDimension, IScaler upscaler, IScaler downscaler, IScaler convolver = null)
        {
            if (outputDimension > inputDimension)
                return "↑" + upscaler.GetDescription();
            if (outputDimension < inputDimension)
                return "↓" + downscaler.GetDescription(true);
            if (convolver != null)
                return "⇄ " + convolver.GetDescription(true);
            return "";
        }

        public static string GetDescription(this IScaler scaler, bool useDownscalerName = false)
        {
            if (useDownscalerName)
            {
                switch (scaler.ScalerType)
                {
                    case ImageScaler.NearestNeighbour:
                        return "Box";
                    case ImageScaler.Bilinear:
                        return "Triangle";
                }
            }

            var result = scaler.GetType().Name;
            switch (scaler.ScalerType)
            {
                case ImageScaler.NearestNeighbour:
                case ImageScaler.Bilinear:
                case ImageScaler.Bicubic:
                case ImageScaler.Softcubic:
                    return result + GetScalerAntiRingingDescription(scaler);
                case ImageScaler.Custom:
                    return GetCustomLinearScalerDescription((Custom) scaler);
            }
            return result + scaler.KernelTaps + GetScalerAntiRingingDescription(scaler);
        }

        private static string GetCustomLinearScalerDescription(Custom scaler)
        {
            var fields = scaler.GetType()
                .GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            var customLinearScaler =
                fields.Select(f => f.GetValue(scaler)).OfType<ICustomLinearScaler>().FirstOrDefault();
            return (customLinearScaler != null ? customLinearScaler.Name : "Custom") + scaler.KernelTaps +
                   GetScalerAntiRingingDescription(scaler);
        }

        private static string GetScalerAntiRingingDescription(IScaler scaler)
        {
            switch (scaler.ScalerType)
            {
                case ImageScaler.NearestNeighbour:
                case ImageScaler.Bilinear:
                case ImageScaler.Softcubic:
                    return "";
                case ImageScaler.Bicubic:
                    return ((Bicubic) scaler).AntiRingingEnabled ? "AR" : "";
                case ImageScaler.Lanczos:
                    return ((Lanczos) scaler).AntiRingingEnabled ? "AR" : "";
                case ImageScaler.Spline:
                    return ((Spline) scaler).AntiRingingEnabled ? "AR" : "";
                case ImageScaler.Jinc:
                    return ((Jinc) scaler).AntiRingingEnabled ? "AR" : "";
                case ImageScaler.Custom:
                    return ((Custom) scaler).AntiRingingEnabled ? "AR" : "";
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }
}
