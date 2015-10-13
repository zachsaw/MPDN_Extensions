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
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.PlayerExtensions.DisplayChangerNativeMethods;
using Mpdn.Extensions.PlayerExtensions.GitHub;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class FullScreenDisplaySelector : PlayerExtension<FullScreenDisplaySelectorSettings, FullScreenDisplaySelectorConfigDialog>
    {
        private Rectangle m_RestoreBounds = Rectangle.Empty;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("F2C4B202-DE21-4AC6-8DCA-44F8716F10A8"),
                    Name = "Full Screen Display Selector",
                    Description = "Selects a display for MPDN to go full screen"
                };
            }
        }

        protected override string ConfigFileName
        {
            get { return "FullScreenDisplaySelector"; }
        }

        public override void Initialize()
        {
            base.Initialize();

            Player.FullScreenMode.Entering += FullScreenModeOnEntering;
            Player.FullScreenMode.Exited += FullScreenModeOnExited;
        }

        public override void Destroy()
        {
            base.Destroy();

            Player.FullScreenMode.Entering -= FullScreenModeOnEntering;
            Player.FullScreenMode.Exited -= FullScreenModeOnExited;
        }

        private void FullScreenModeOnEntering(object sender, EventArgs eventArgs)
        {
            if (!Settings.Enabled)
                return;

            if (Settings.Monitor >= Displays.AllDisplays.Length)
                return;

            m_RestoreBounds = Player.ActiveForm.Bounds;

            var newBounds = Displays.AllDisplays[Settings.Monitor].Bounds;
            Player.ActiveForm.Top = newBounds.Top;
            Player.ActiveForm.Left = newBounds.Left;
        }

        private void FullScreenModeOnExited(object sender, EventArgs eventArgs)
        {
            var bounds = m_RestoreBounds;
            m_RestoreBounds = Rectangle.Empty;

            if (bounds == Rectangle.Empty)
                return;

            Player.ActiveForm.Bounds = bounds;
        }
    }

    public class FullScreenDisplaySelectorSettings
    {
        public FullScreenDisplaySelectorSettings()
        {
            Monitor = 0;
            Enabled = false;
        }

        public int Monitor { get; set; }
        public bool Enabled { get; set; }
    }

    public class Displays
    {
        [DllImport("user32.dll")]
        private static extern bool EnumDisplayDevices(string lpDevice, uint iDevNum,
            ref DisplayDevice lpDisplayDevice, uint dwFlags);

        private static Display[] s_AllDisplays;

        public static Display[] AllDisplays
        {
            get { return GetAllDisplays(); }
        }

        private static Display[] GetAllDisplays()
        {
            if (s_AllDisplays != null)
            {
                return s_AllDisplays;
            }

            var result = new List<Display>();

            var device = new DisplayDevice();
            device.cb = Marshal.SizeOf(device);
            try
            {
                for (uint i = 0; EnumDisplayDevices(null, i, ref device, 1); i++)
                {
                    if (device.StateFlags.HasFlag(DisplayDeviceStateFlags.AttachedToDesktop))
                    {
                        device.cb = Marshal.SizeOf(device);
                        EnumDisplayDevices(device.DeviceName, 0, ref device, 0);
                        if (string.IsNullOrWhiteSpace(device.DeviceName))
                            continue; // no monitor attached (VGA display)
                        result.Add(new Display(device));
                    }
                    device.cb = Marshal.SizeOf(device);
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine("{0}", ex);
            }

            s_AllDisplays = result.ToArray();
            return s_AllDisplays;
        }
    }

    public class Display
    {
        private readonly DisplayDevice m_Device;
        private readonly Rectangle m_Bounds;

        public string DeviceName { get { return m_Device.DeviceString; } }
        public Rectangle Bounds { get { return m_Bounds; } }

        public Display(DisplayDevice device)
        {
            m_Device = device;
            m_Bounds = GetBounds(device);
        }

        private static Rectangle GetBounds(DisplayDevice device)
        {
            var devName = device.DeviceName;
            devName = devName.Substring(0, devName.LastIndexOf('\\'));
            var dm = NativeMethods.CreateDevmode(devName);
            if (NativeMethods.EnumDisplaySettings(devName, NativeMethods.ENUM_CURRENT_SETTINGS, ref dm) == 0)
                throw new Win32Exception();

            int width = 0, height = 0, posX = 0, posY = 0;
            if ((dm.dmFields & (int)DM.PelsWidth) > 0)
            {
                width = dm.dmPelsWidth;
            }
            if ((dm.dmFields & (int)DM.PelsHeight) > 0)
            {
                height = dm.dmPelsHeight;
            }
            if ((dm.dmFields & (int)DM.Position) > 0)
            {
                posX = dm.dmPositionX;
                posY = dm.dmPositionY;
            }

            if (width != 0 && height != 0)
            {
                return new Rectangle(posX, posY, width, height);
            }

            throw new Exception("Not expected: width or height of display = 0!");
        }
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct DisplayDevice
    {
        [MarshalAs(UnmanagedType.U4)]
        public int cb;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
        public string DeviceName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string DeviceString;
        [MarshalAs(UnmanagedType.U4)]
        public DisplayDeviceStateFlags StateFlags;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string DeviceID;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string DeviceKey;
    }

    [Flags]
    public enum DisplayDeviceStateFlags
    {
        /// <summary>The device is part of the desktop.</summary>
        AttachedToDesktop = 0x1,
        MultiDriver = 0x2,

        /// <summary>The device is part of the desktop.</summary>
        PrimaryDevice = 0x4,

        /// <summary>Represents a pseudo device used to mirror application drawing for remoting or other purposes.</summary>
        MirroringDriver = 0x8,

        /// <summary>The device is VGA compatible.</summary>
        VgaCompatible = 0x10,

        /// <summary>The device is removable; it cannot be the primary display.</summary>
        Removable = 0x20,

        /// <summary>The device has more display modes than its output devices support.</summary>
        ModesPruned = 0x8000000,
        Remote = 0x4000000,
        Disconnect = 0x2000000
    }
}
