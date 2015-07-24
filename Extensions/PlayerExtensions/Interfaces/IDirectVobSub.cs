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
//Source: https://picprojects.googlecode.com/svn/trunk/MediaServiceHost/MediaHost/IDirectVobSub.cs

using System;
using System.Runtime.InteropServices;
using DirectShowLib;

namespace Mpdn.Extensions.PlayerExtensions.Interfaces
{
    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown),
     Guid("EBE1FB08-3957-47ca-AF13-5827E5442E56")]
    public interface IDirectVobSub
    {
        [PreserveSig]
        int get_FileName([In, Out, MarshalAs(UnmanagedType.BStr)] ref string fn);

        [PreserveSig]
        int put_FileName([In, MarshalAs(UnmanagedType.BStr)] string fn);

        [PreserveSig]
        int get_LanguageCount([Out] out int nLangs);

        [PreserveSig]
        int get_LanguageName([In] int iLanguage, [Out, MarshalAs(UnmanagedType.LPWStr)] out string fn);

        [PreserveSig]
        int get_SelectedLanguage([Out] out int iSelected);

        [PreserveSig]
        int put_SelectedLanguage([In] int iSelected);

        [PreserveSig]
        int get_HideSubtitles([Out, MarshalAs(UnmanagedType.VariantBool)] out bool fHideSubtitles);

        [PreserveSig]
        int put_HideSubtitles([In, MarshalAs(UnmanagedType.VariantBool)] bool fHideSubtitles);

        [PreserveSig]
        int get_PreBuffering([Out] out bool fDoPreBuffering);

        [PreserveSig]
        int put_PreBuffering([In] bool fDoPreBuffering);

        [PreserveSig]
        int get_Placement([Out] out bool fOverridePlacement,
            [Out] out int xprec,
            [Out] out int yprec);

        [PreserveSig]
        int put_Placement([In] bool fOverridePlacement,
            [In] int xprec, [In] int yprec);

        [PreserveSig]
        int get_VobSubSettings([Out] out bool fBuffer,
            [Out, MarshalAs(UnmanagedType.VariantBool)] out bool fOnlyShowForcedSubs,
            [Out, MarshalAs(UnmanagedType.VariantBool)] out bool fPolygonize);

        [PreserveSig]
        int put_VobSubSettings([In] bool fBuffer,
            [In, MarshalAs(UnmanagedType.VariantBool)] bool fOnlyShowForcedSubs,
            [In, MarshalAs(UnmanagedType.VariantBool)] bool fPolygonize);

        [PreserveSig]
        int get_TextSettings(
            [In] IntPtr lf,
            [In] int lflen,
            [Out] out int color,
            [Out, MarshalAs(UnmanagedType.VariantBool)] out bool fShadow,
            [Out, MarshalAs(UnmanagedType.VariantBool)] out bool fOutline,
            [Out, MarshalAs(UnmanagedType.VariantBool)] out bool fAdvancedRenderer);

        [PreserveSig]
        int put_TextSettings(
            [In] IntPtr lf,
            [In] int lflen,
            [In] int color,
            [In, MarshalAs(UnmanagedType.VariantBool)] bool fShadow,
            [In, MarshalAs(UnmanagedType.VariantBool)] bool fOutline,
            [In, MarshalAs(UnmanagedType.VariantBool)] bool fAdvancedRenderer);

        [PreserveSig]
        int get_Flip(
            [Out, MarshalAs(UnmanagedType.VariantBool)] out bool fPicture,
            [Out, MarshalAs(UnmanagedType.VariantBool)] out bool fSubtitles);

        [PreserveSig]
        int put_Flip(
            [In, MarshalAs(UnmanagedType.VariantBool)] bool fPicture,
            [In, MarshalAs(UnmanagedType.VariantBool)] bool fSubtitles);

        [PreserveSig]
        int get_OSD([Out, MarshalAs(UnmanagedType.VariantBool)] out bool fOSD);

        [PreserveSig]
        int put_OSD([In, MarshalAs(UnmanagedType.VariantBool)] bool fOSD);

        [PreserveSig]
        int get_SaveFullPath([Out, MarshalAs(UnmanagedType.VariantBool)] out bool fSaveFullPath);

        [PreserveSig]
        int put_SaveFullPath([In, MarshalAs(UnmanagedType.VariantBool)] bool fSaveFullPath);

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

        [PreserveSig]
        int get_MediaFPS(
            [Out, MarshalAs(UnmanagedType.VariantBool)] out bool fEnabled,
            [Out] out double fps);

        [PreserveSig]
        int put_MediaFPS(
            [In, MarshalAs(UnmanagedType.VariantBool)] bool fEnabled,
            [In] double fps);

        //// no longer supported
        [PreserveSig]
        int get_ColorFormat([Out] out int iPosition);

        [PreserveSig]
        int put_ColorFormat([In] int iPosition);

        [PreserveSig]
        int get_ZoomRect([Out] out NormalizedRect rect);

        [PreserveSig]
        int put_ZoomRect([Out, In] ref NormalizedRect rect);

        [PreserveSig]
        int UpdateRegistry();

        [PreserveSig]
        int HasConfigDialog([In] int iSelected);

        [PreserveSig]
        int ShowConfigDialog([In] int iSelected, [In] IntPtr hwndParent);

        ////
        [PreserveSig]
        int IsSubtitleReloaderLocked([Out, MarshalAs(UnmanagedType.VariantBool)] out bool fLocked);

        [PreserveSig]
        int LockSubtitleReloader([In, MarshalAs(UnmanagedType.VariantBool)] bool fLocked);

        [PreserveSig]
        int get_SubtitleReloader([Out, MarshalAs(UnmanagedType.VariantBool)] out bool fDisabled);

        [PreserveSig]
        int put_SubtitleReloader([In, MarshalAs(UnmanagedType.VariantBool)] bool fDisabled);

        ////
        [PreserveSig]
        int get_ExtendPicture(
            [Out] out int horizontal, // 0 - disabled, 1 - mod32 extension (width = (width+31)&~31)
            [Out] out int vertical,
            // 0 - disabled, 1 - 16:9, 2 - 4:3, 0x80 - crop (use crop together with 16:9 or 4:3, eg 0x81 will crop to 16:9 if the picture was taller)
            [Out] out int resx2, // 0 - disabled, 1 - enabled, 2 - depends on the original resolution
            [Out] out int resx2minw,
            // resolution doubler will be used if width*height <= resx2minw*resx2minh (resx2minw*resx2minh equals to 384*288 by default)
            [Out] out int resx2minh);

        [PreserveSig]
        int put_ExtendPicture(
            [In] int horizontal, // 0 - disabled, 1 - mod32 extension (width = (width+31)&~31)
            [In] int vertical,
            // 0 - disabled, 1 - 16:9, 2 - 4:3, 0x80 - crop (use crop together with 16:9 or 4:3, eg 0x81 will crop to 16:9 if the picture was taller)
            [In] int resx2, // 0 - disabled, 1 - enabled, 2 - depends on the original resolution
            [In] int resx2minw,
            // resolution doubler will be used if width*height <= resx2minw*resx2minh (resx2minw*resx2minh equals to 384*288 by default)
            [In] int resx2minh);

        [PreserveSig]
        int get_LoadSettings(
            [Out] out int level,
            [Out, MarshalAs(UnmanagedType.VariantBool)] out bool fExternalLoad,
            [Out, MarshalAs(UnmanagedType.VariantBool)] out bool fWebLoad,
            [Out, MarshalAs(UnmanagedType.VariantBool)] out bool fEmbeddedLoad);

        [PreserveSig]
        int put_LoadSettings(
            [In] int level,
            [In, MarshalAs(UnmanagedType.VariantBool)] bool fExternalLoad,
            [In, MarshalAs(UnmanagedType.VariantBool)] bool fWebLoad,
            [In, MarshalAs(UnmanagedType.VariantBool)] bool fEmbeddedLoad);
    }
}