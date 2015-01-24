using System;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace Mpdn.RenderScript
{
    public static class RenderScript
    {
        public static IRenderScriptUi Empty = new NullRenderScriptUi();

        public static bool IsEmpty(this IRenderScriptUi script)
        {
            return script is NullRenderScriptUi;
        }

        private class NullRenderScriptUi : IRenderScriptUi
        {
            public ScriptDescriptor Descriptor
            {
                get
                {
                    return new ScriptDescriptor
                    {
                        Guid = Guid.Empty,
                        Name = "None",
                        Description = "Do not use render script",
                        HasConfigDialog = false
                    };
                }
            }

            public IRenderScript CreateRenderScript()
            {
                return null;
            }

            public void Initialize()
            {
            }

            public void Destroy()
            {
            }

            public bool ShowConfigDialog(IWin32Window owner)
            {
                return false;
            }
        }
    }

    public static class Common
    {
        public static void Dispose(object obj)
        {
            var disposable = obj as IDisposable;
            if (disposable != null)
            {
                disposable.Dispose();
            }
        }

        public static void Dispose<T>(ref T obj) where T : class, IDisposable
        {
            if (obj != null)
            {
                obj.Dispose();
                obj = default(T);
            }
        }

        public static string GetDirectoryName(string path)
        {
            if (path == null)
            {
                throw new ArgumentNullException("path");
            }

            return Path.GetDirectoryName(path) ?? Path.GetPathRoot(path);
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
                var attrs = memInfo[0].GetCustomAttributes(typeof (DescriptionAttribute), false);

                if (attrs.Length > 0)
                {
                    return ((DescriptionAttribute) attrs[0]).Description;
                }
            }

            return enumString;
        }

        public static string[] GetDescriptions<T>()
        {
            return Enum.GetValues(typeof (T)).Cast<Enum>().Select(val => val.ToDescription()).ToArray();
        }
    }
}