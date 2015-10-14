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
using System.Runtime.InteropServices;
using System.Security;

namespace Mpdn.Extensions.PlayerExtensions.Interfaces
{
    // Codecs supported in the LAV Audio configuration
    // Codecs not listed here cannot be turned off. You can request codecs to be added to this list, if you wish.
    public enum LAVAudioCodec
    {
        Codec_AAC,
        Codec_AC3,
        Codec_EAC3,
        Codec_DTS,
        Codec_MP2,
        Codec_MP3,
        Codec_TRUEHD,
        Codec_FLAC,
        Codec_VORBIS,
        Codec_LPCM,
        Codec_PCM,
        Codec_WAVPACK,
        Codec_TTA,
        Codec_WMA2,
        Codec_WMAPRO,
        Codec_Cook,
        Codec_RealAudio,
        Codec_WMALL,
        Codec_ALAC,
        Codec_Opus,
        Codec_AMR,
        Codec_Nellymoser,
        Codec_MSPCM,
        Codec_Truespeech,
        Codec_TAK,
        Codec_ATRAC,

        Codec_AudioNB // Number of entries (do not use when dynamically linking)
    };

    // Bitstreaming Codecs supported in LAV Audio
    public enum LAVBitstreamCodec
    {
        Bitstream_AC3,
        Bitstream_EAC3,
        Bitstream_TRUEHD,
        Bitstream_DTS,
        Bitstream_DTSHD,

        Bitstream_NB // Number of entries (do not use when dynamically linking)
    };


    // Supported Sample Formats in LAV Audio
    public enum LAVAudioSampleFormat
    {
        SampleFormat_None = -1,
        SampleFormat_16,
        SampleFormat_24,
        SampleFormat_32,
        SampleFormat_U8,
        SampleFormat_FP32,
        SampleFormat_Bitstream,

        SampleFormat_NB // Number of entries (do not use when dynamically linking)
    };

    public enum LAVAudioMixingMode
    {
        MatrixEncoding_None,
        MatrixEncoding_Dolby,
        MatrixEncoding_DPLII,

        MatrixEncoding_NB
    };

    [Flags]
    public enum LAVMixingFlag
    {
        UntouchedStereo = 1,
        NormalizeMatrix = 2,
        ClipProtection = 4
    }

    [ComImport, Guid("4158A22B-6553-45D0-8069-24716F8FF171")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    [SuppressUnmanagedCodeSecurity]
    public interface ILAVAudioSettings
    {
        // Switch to Runtime Config mode. This will reset all settings to default, and no changes to the settings will be saved
        // You can use this to programmatically configure LAV Audio without interfering with the users settings in the registry.
        // Subsequent calls to this function will reset all settings back to defaults, even if the mode does not change.
        //
        // Note that calling this function during playback is not supported and may exhibit undocumented behaviour. 
        // For smooth operations, it must be called before LAV Audio is connected to other filters.
        void SetRuntimeConfig(bool bRuntimeConfig);

        // Dynamic Range Compression
        // pbDRCEnabled: The state of DRC
        // piDRCLevel:   The DRC strength (0-100, 100 is maximum)
        void GetDRC(out bool pbDRCEnabled, out int piDRCLevel);
        void SetDRC(bool bDRCEnabled, int iDRCLevel);

        // Configure which codecs are enabled
        // If aCodec is invalid (possibly a version difference), Get will return FALSE, and Set E_FAIL.
        bool GetFormatConfiguration(LAVAudioCodec aCodec);
        void SetFormatConfiguration(LAVAudioCodec aCodec, bool bEnabled);

        // Control Bitstreaming
        // If bsCodec is invalid (possibly a version difference), Get will return FALSE, and Set E_FAIL.
        bool GetBitstreamConfig(LAVBitstreamCodec bsCodec);
        void SetBitstreamConfig(LAVBitstreamCodec bsCodec, bool bEnabled);

        // Should "normal" DTS frames be encapsulated in DTS-HD frames when bitstreaming?
        bool GetDTSHDFraming();
        void SetDTSHDFraming(bool bHDFraming);

        // Control Auto A/V syncing
        bool GetAutoAVSync();
        void SetAutoAVSync(bool bAutoSync);

        // Convert all Channel Layouts to standard layouts
        // Standard are: Mono, Stereo, 5.1, 6.1, 7.1
        bool GetOutputStandardLayout();
        void SetOutputStandardLayout(bool bStdLayout);

        // Expand Mono to Stereo by simply doubling the audio
        bool GetExpandMono();
        void SetExpandMono(bool bExpandMono);

        // Expand 6.1 to 7.1 by doubling the back center
        bool GetExpand61();
        void SetExpand61(bool bExpand61);

        // Allow Raw PCM and SPDIF encoded input
        bool GetAllowRawSPDIFInput();
        void SetAllowRawSPDIFInput(bool bAllow);

        // Configure which sample formats are enabled
        // Note: SampleFormat_Bitstream cannot be controlled by this
        bool GetSampleFormat(LAVAudioSampleFormat format);
        void SetSampleFormat(LAVAudioSampleFormat format, bool bEnabled);

        // Configure a delay for the audio
        void GetAudioDelay(out bool pbEnabled, out int pDelay);
        void SetAudioDelay(bool bEnabled, int delay);

        // Enable/Disable Mixing
        void SetMixingEnabled(bool bEnabled);
        bool GetMixingEnabled();

        // Control Mixing Layout
        void SetMixingLayout(int dwLayout);
        int GetMixingLayout();

        // Set Mixing Flags
        void SetMixingFlags(int dwFlags);
        int GetMixingFlags();

        // Set Mixing Mode
        void SetMixingMode(LAVAudioMixingMode mixingMode);
        LAVAudioMixingMode GetMixingMode();

        // Set Mixing Levels
        void SetMixingLevels(int dwCenterLevel, int dwSurroundLevel, int dwLFELevel);
        void GetMixingLevels(out int dwCenterLevel, out int dwSurroundLevel, out int dwLFELevel);

        // Toggle Tray Icon
        void SetTrayIcon(bool bEnabled);
        bool GetTrayIcon();

        // Toggle Dithering for sample format conversion
        void SetSampleConvertDithering(bool bEnabled);
        bool GetSampleConvertDithering();

        // Suppress sample format changes. This will allow channel count to increase, but not to reduce, instead adding empty channels
        // This option is NOT persistent
        void SetSuppressFormatChanges(bool bEnabled);
        bool GetSuppressFormatChanges();
    }
}
