using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Security;
using DirectShowLib;

namespace Mpdn.Extensions.PlayerExtensions.Interfaces
{

    #region MFType

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public class MFSize
    {
        public int cx;
        public int cy;

        public MFSize()
        {
            cx = 0;
            cy = 0;
        }

        public MFSize(int iWidth, int iHeight)
        {
            cx = iWidth;
            cy = iHeight;
        }

        public int Width
        {
            get { return cx; }
            set { cx = value; }
        }

        public int Height
        {
            get { return cy; }
            set { cy = value; }
        }
    }

    /// <summary>
    ///     MFRect is a managed representation of the Win32 RECT structure.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public class MFRect
    {
        public int left;
        public int top;
        public int right;
        public int bottom;

        /// <summary>
        ///     Empty contructor. Initialize all fields to 0
        /// </summary>
        public MFRect()
        {
        }

        /// <summary>
        ///     A parametred constructor. Initialize fields with given values.
        /// </summary>
        /// <param name="left">the left value</param>
        /// <param name="top">the top value</param>
        /// <param name="right">the right value</param>
        /// <param name="bottom">the bottom value</param>
        public MFRect(int l, int t, int r, int b)
        {
            left = l;
            top = t;
            right = r;
            bottom = b;
        }

        /// <summary>
        ///     A parametred constructor. Initialize fields with a given <see cref="System.Drawing.Rectangle" />.
        /// </summary>
        /// <param name="rectangle">A <see cref="System.Drawing.Rectangle" /></param>
        /// <remarks>
        ///     Warning, MFRect define a rectangle by defining two of his corners and <see cref="System.Drawing.Rectangle" />
        ///     define a rectangle with his upper/left corner, his width and his height.
        /// </remarks>
        public MFRect(Rectangle rectangle)
        {
            left = rectangle.Left;
            top = rectangle.Top;
            right = rectangle.Right;
            bottom = rectangle.Bottom;
        }

        /// <summary>
        ///     Provide de string representation of this MFRect instance
        /// </summary>
        /// <returns>A string formated like this : [left, top - right, bottom]</returns>
        public override string ToString()
        {
            return string.Format("[{0}, {1}] - [{2}, {3}]", left, top, right, bottom);
        }

        public override int GetHashCode()
        {
            return left.GetHashCode() |
                   top.GetHashCode() |
                   right.GetHashCode() |
                   bottom.GetHashCode();
        }

        public override bool Equals(object obj)
        {
            if (obj is MFRect)
            {
                MFRect cmp = (MFRect) obj;

                return right == cmp.right && bottom == cmp.bottom && left == cmp.left && top == cmp.top;
            }

            if (obj is Rectangle)
            {
                Rectangle cmp = (Rectangle) obj;

                return right == cmp.Right && bottom == cmp.Bottom && left == cmp.Left && top == cmp.Top;
            }

            return false;
        }

        /// <summary>
        ///     Checks to see if the rectangle is empty
        /// </summary>
        /// <returns>Returns true if the rectangle is empty</returns>
        public bool IsEmpty()
        {
            return (right <= left || bottom <= top);
        }

        /// <summary>
        ///     Define implicit cast between MFRect and System.Drawing.Rectangle for languages supporting this feature.
        ///     VB.Net doesn't support implicit cast. <see cref="MFRect.ToRectangle" /> for similar functionality.
        ///     <code>
        ///    // Define a new Rectangle instance
        ///    Rectangle r = new Rectangle(0, 0, 100, 100);
        ///    // Do implicit cast between Rectangle and MFRect
        ///    MFRect mfR = r;
        /// 
        ///    Console.WriteLine(mfR.ToString());
        ///  </code>
        /// </summary>
        /// <param name="r">a MFRect to be cast</param>
        /// <returns>A casted System.Drawing.Rectangle</returns>
        public static implicit operator Rectangle(MFRect r)
        {
            return r.ToRectangle();
        }

        /// <summary>
        ///     Define implicit cast between System.Drawing.Rectangle and MFRect for languages supporting this feature.
        ///     VB.Net doesn't support implicit cast. <see cref="MFRect.FromRectangle" /> for similar functionality.
        ///     <code>
        ///    // Define a new MFRect instance
        ///    MFRect mfR = new MFRect(0, 0, 100, 100);
        ///    // Do implicit cast between MFRect and Rectangle
        ///    Rectangle r = mfR;
        /// 
        ///    Console.WriteLine(r.ToString());
        ///  </code>
        /// </summary>
        /// <param name="r">A System.Drawing.Rectangle to be cast</param>
        /// <returns>A casted MFRect</returns>
        public static implicit operator MFRect(Rectangle r)
        {
            return new MFRect(r);
        }

        /// <summary>
        ///     Get the System.Drawing.Rectangle equivalent to this MFRect instance.
        /// </summary>
        /// <returns>A System.Drawing.Rectangle</returns>
        public Rectangle ToRectangle()
        {
            return new Rectangle(left, top, (right - left), (bottom - top));
        }

        /// <summary>
        ///     Get a new MFRect instance for a given <see cref="System.Drawing.Rectangle" />
        /// </summary>
        /// <param name="r">The <see cref="System.Drawing.Rectangle" /> used to initialize this new MFGuid</param>
        /// <returns>A new instance of MFGuid</returns>
        public static MFRect FromRectangle(Rectangle r)
        {
            return new MFRect(r);
        }

        /// <summary>
        ///     Copy the members from an MFRect into this object
        /// </summary>
        /// <param name="from">The rectangle from which to copy the values.</param>
        public void CopyFrom(MFRect from)
        {
            left = from.left;
            top = from.top;
            right = from.right;
            bottom = from.bottom;
        }
    }

    #endregion

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public class LOGFONT
    {
        public int lfHeight;
        public int lfWidth;
        public int lfEscapement;
        public int lfOrientation;
        public int lfWeight;
        public byte lfItalic;
        public byte lfUnderline;
        public byte lfStrikeOut;
        public byte lfCharSet;
        public byte lfOutPrecision;
        public byte lfClipPrecision;
        public byte lfQuality;
        public byte lfPitchAndFamily;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string lfFaceName = string.Empty;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct NORMALIZEDRECT
    {
        public float left;
        public float top;
        public float right;
        public float bottom;
    }

    [ComVisible(true), ComImport, SuppressUnmanagedCodeSecurity,
     Guid("EBE1FB08-3957-47ca-AF13-5827E5442E56"),
     InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IDirectVobSub : IBaseFilter
    {
        [PreserveSig]
        int get_FileName( /*[MarshalAs(UnmanagedType.LPWStr, SizeConst = 260)]ref StringBuilder*/ IntPtr fn);

        [PreserveSig]
        int put_FileName([MarshalAs(UnmanagedType.LPWStr)] string fn);

        [PreserveSig]
        int get_LanguageCount(out int nLangs);

        [PreserveSig]
        int get_LanguageName(int iLanguage, [MarshalAs(UnmanagedType.LPWStr)] out string ppName);

        [PreserveSig]
        int get_SelectedLanguage(ref int iSelected);

        [PreserveSig]
        int put_SelectedLanguage(int iSelected);

        [PreserveSig]
        int get_HideSubtitles(ref bool fHideSubtitles);

        [PreserveSig]
        int put_HideSubtitles([MarshalAs(UnmanagedType.I1)] bool fHideSubtitles);

        [PreserveSig]
        int get_PreBuffering(ref bool fDoPreBuffering);

        [PreserveSig]
        int put_PreBuffering([MarshalAs(UnmanagedType.I1)] bool fDoPreBuffering);

        [PreserveSig]
        int get_Placement(ref bool fOverridePlacement, ref int xperc, ref int yperc);

        [PreserveSig]
        int put_Placement([MarshalAs(UnmanagedType.I1)] bool fOverridePlacement, int xperc, int yperc);

        [PreserveSig]
        int get_VobSubSettings(ref bool fBuffer, ref bool fOnlyShowForcedSubs, ref bool fPolygonize);

        [PreserveSig]
        int put_VobSubSettings([MarshalAs(UnmanagedType.I1)] bool fBuffer,
            [MarshalAs(UnmanagedType.I1)] bool fOnlyShowForcedSubs, [MarshalAs(UnmanagedType.I1)] bool fPolygonize);

        [PreserveSig]
        int get_TextSettings(LOGFONT lf, int lflen, ref uint color, ref bool fShadow, ref bool fOutline,
            ref bool fAdvancedRenderer);

        [PreserveSig]
        int put_TextSettings(LOGFONT lf, int lflen, uint color, bool fShadow, bool fOutline, bool fAdvancedRenderer);

        [PreserveSig]
        int get_Flip(ref bool fPicture, ref bool fSubtitles);

        [PreserveSig]
        int put_Flip([MarshalAs(UnmanagedType.I1)] bool fPicture, [MarshalAs(UnmanagedType.I1)] bool fSubtitles);

        [PreserveSig]
        int get_OSD(ref bool fOSD);

        [PreserveSig]
        int put_OSD([MarshalAs(UnmanagedType.I1)] bool fOSD);

        [PreserveSig]
        int get_SubtitleTiming(
            [Out] out int delay,
            [Out] out int speedmul,
            [Out] out int speeddiv);

        [PreserveSig]
        int put_SubtitleTiming(
            [In] int delay,
            [In] int speedmul,
            [In] int speeddiv);
    }

    [ComVisible(true), ComImport, SuppressUnmanagedCodeSecurity,
     Guid("85E5D6F9-BEFB-4E01-B047-758359CDF9AB"),
     InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IDirectVobSubXy : IBaseFilter
    {
        [PreserveSig]
        int XyGetBool(int field, ref bool value);

        [PreserveSig]
        int XyGetInt(int field, ref int value);

        [PreserveSig]
        int XyGetSize(int field, ref MFSize value);

        [PreserveSig]
        int XyGetRect(int field, ref MFRect value);

        [PreserveSig]
        int XyGetUlonglong(int field, ref long value);

        [PreserveSig]
        int XyGetDouble(int field, ref double value);

        [PreserveSig]
        int XyGetString(int field, ref string value, ref int chars);

        [PreserveSig]
        int XyGetBin(int field, ref IntPtr value, ref int size);

        [PreserveSig]
        int XySetBool(int field, bool value);

        [PreserveSig]
        int XySetInt(int field, int value);

        [PreserveSig]
        int XySetSize(int field, MFSize value);

        [PreserveSig]
        int XySetRect(int field, MFRect value);

        [PreserveSig]
        int XySetUlonglong(int field, long value);

        [PreserveSig]
        int XySetDouble(int field, double value);

        [PreserveSig]
        int XySetString(int field, string value, int chars);

        [PreserveSig]
        int XySetBin(int field, IntPtr value, int size);
    }

    [ComVisible(true), ComImport, SuppressUnmanagedCodeSecurity,
     Guid("AB52FC9C-2415-4dca-BC1C-8DCC2EAE8151"),
     InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IDirectVobSub3 : IBaseFilter
    {
        [PreserveSig]
        int OpenSubtitles([MarshalAs(UnmanagedType.LPWStr)] string fn);

        [PreserveSig]
        int SkipAutoloadCheck(int boolval);
    }
}