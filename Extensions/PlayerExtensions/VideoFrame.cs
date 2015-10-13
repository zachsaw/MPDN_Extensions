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
using Mpdn.Config;
using Mpdn.Extensions.Framework;
using Mpdn.RenderScript;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class VideoFrame : PlayerExtension
    {
        private const Category CATEGORY = Category.View;
        private const string SUBCATEGORY = "Video Frame";

        private readonly PlayerMenuItem m_ApplyLetterBoxingMenu = new PlayerMenuItem(initiallyDisabled: true);
        private readonly PlayerMenuItem m_Zoom75LetterBoxingMenu = new PlayerMenuItem(initiallyDisabled: true);
        private readonly PlayerMenuItem m_Zoom50LetterBoxingMenu = new PlayerMenuItem(initiallyDisabled: true);
        private readonly PlayerMenuItem m_Zoom25LetterBoxingMenu = new PlayerMenuItem(initiallyDisabled: true);
        private readonly PlayerMenuItem m_ZoomNoLetterBoxingMenu = new PlayerMenuItem(initiallyDisabled: true);
        private readonly PlayerMenuItem m_FillMenu = new PlayerMenuItem(initiallyDisabled: true);
        private readonly PlayerMenuItem m_ToggleLetterBoxingMenu = new PlayerMenuItem(initiallyDisabled: true);

        private bool m_Enabled;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("A8055BB5-25BE-4ECE-8C01-18ED36B8E6C5"),
                    Name = "Video Frame",
                    Description = "Controls the video frame"
                };
            }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(CATEGORY, SUBCATEGORY, "Apply letter boxing (no clipping)", PlayerMenuItemType.Radio, () => DoAction(SelectApplyLetterBoxing), m_ApplyLetterBoxingMenu),
                    new Verb(CATEGORY, SUBCATEGORY, "75% letter boxing (clip video)", PlayerMenuItemType.Radio, () => DoAction(SelectZoom75LetterBoxing), m_Zoom75LetterBoxingMenu),
                    new Verb(CATEGORY, SUBCATEGORY, "50% letter boxing (clip video)", PlayerMenuItemType.Radio, () => DoAction(SelectZoom50LetterBoxing), m_Zoom50LetterBoxingMenu),
                    new Verb(CATEGORY, SUBCATEGORY, "25% letter boxing (clip video)", PlayerMenuItemType.Radio, () => DoAction(SelectZoom25LetterBoxing), m_Zoom25LetterBoxingMenu),
                    new Verb(CATEGORY, SUBCATEGORY, "No letter boxing (clip video)", PlayerMenuItemType.Radio, () => DoAction(SelectZoomNoLetterBoxing), m_ZoomNoLetterBoxingMenu),
                    new Verb(CATEGORY, SUBCATEGORY, "Fill (stretch to fit)", PlayerMenuItemType.Radio, () => DoAction(SelectFill), m_FillMenu),
                    new Verb(CATEGORY, SUBCATEGORY, "Toggle letter boxing", "Ctrl+Shift+V", string.Empty, () => DoAction(ToggleLetterBoxing), m_ToggleLetterBoxingMenu)
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();
            Player.StateChanged += OnPlayerStateChanged;
            Player.Config.Changed += OnSettingsChanged;
        }

        public override void Destroy()
        {
            Player.Config.Changed -= OnSettingsChanged;
            Player.StateChanged -= OnPlayerStateChanged;
            base.Destroy();
        }

        protected static VideoRendererSettings RendererSettings
        {
            get { return Player.Config.Settings.VideoRendererSettings; }
        }

        private void DoAction(Action action)
        {
            if (m_Enabled)
            {
                action();
            }
        }

        private void OnPlayerStateChanged(object sender, PlayerStateEventArgs args)
        {
            var enabled = (args.NewState == PlayerState.Playing || args.NewState == PlayerState.Paused) &&
                          Renderer.InputFormat.IsYuv();

            m_Enabled = enabled;

            m_ApplyLetterBoxingMenu.Enabled = enabled;
            m_Zoom75LetterBoxingMenu.Enabled = enabled;
            m_Zoom50LetterBoxingMenu.Enabled = enabled;
            m_Zoom25LetterBoxingMenu.Enabled = enabled;
            m_ZoomNoLetterBoxingMenu.Enabled = enabled;
            m_FillMenu.Enabled = enabled;
            m_ToggleLetterBoxingMenu.Enabled = enabled;

            if (args.NewState != PlayerState.Closed && args.OldState != PlayerState.Closed)
                return;

            UpdateControls();
        }

        private void OnSettingsChanged(object sender, EventArgs e)
        {
            UpdateControls();
        }

        private void ClearSelections()
        {
            m_ApplyLetterBoxingMenu.Checked = false;
            m_Zoom75LetterBoxingMenu.Checked = false;
            m_Zoom50LetterBoxingMenu.Checked = false;
            m_Zoom25LetterBoxingMenu.Checked = false;
            m_ZoomNoLetterBoxingMenu.Checked = false;
            m_FillMenu.Checked = false;
        }

        private void SelectLetterBoxingMode(FillMode mode)
        {
            RendererSettings.LetterBoxing = mode;
            ApplyLetterBoxing();
        }

        private void SelectApplyLetterBoxing()
        {
            SelectLetterBoxingMode(FillMode.ApplyLetterBoxing);
        }

        private void SelectZoom75LetterBoxing()
        {
            SelectLetterBoxingMode(FillMode.Zoomed75PcLetterBoxing);
        }

        private void SelectZoom50LetterBoxing()
        {
            SelectLetterBoxingMode(FillMode.Zoomed50PcLetterBoxing);
        }

        private void SelectZoom25LetterBoxing()
        {
            SelectLetterBoxingMode(FillMode.Zoomed25PcLetterBoxing);
        }

        private void SelectZoomNoLetterBoxing()
        {
            SelectLetterBoxingMode(FillMode.ZoomedNoLetterBoxing);
        }

        private void SelectFill()
        {
            SelectLetterBoxingMode(FillMode.Fill);
        }

        private void ApplyLetterBoxing()
        {
            Player.OsdText.Show(RendererSettings.LetterBoxing.ToDescription());
            Player.Config.Refresh();
        }

        private void UpdateControls()
        {
            ClearSelections();

            switch (RendererSettings.LetterBoxing)
            {
                case FillMode.ApplyLetterBoxing:
                    m_ApplyLetterBoxingMenu.Checked = true;
                    break;
                case FillMode.Zoomed75PcLetterBoxing:
                    m_Zoom75LetterBoxingMenu.Checked = true;
                    break;
                case FillMode.Zoomed50PcLetterBoxing:
                    m_Zoom50LetterBoxingMenu.Checked = true;
                    break;
                case FillMode.Zoomed25PcLetterBoxing:
                    m_Zoom25LetterBoxingMenu.Checked = true;
                    break;
                case FillMode.ZoomedNoLetterBoxing:
                    m_ZoomNoLetterBoxingMenu.Checked = true;
                    break;
                case FillMode.Fill:
                    m_FillMenu.Checked = true;
                    break;
            }
        }

        private void ToggleLetterBoxing()
        {
            switch (RendererSettings.LetterBoxing)
            {
                case FillMode.ApplyLetterBoxing:
                    RendererSettings.LetterBoxing = FillMode.Zoomed75PcLetterBoxing;
                    break;
                case FillMode.Zoomed75PcLetterBoxing:
                    RendererSettings.LetterBoxing = FillMode.Zoomed50PcLetterBoxing;
                    break;
                case FillMode.Zoomed50PcLetterBoxing:
                    RendererSettings.LetterBoxing = FillMode.Zoomed25PcLetterBoxing;
                    break;
                case FillMode.Zoomed25PcLetterBoxing:
                    RendererSettings.LetterBoxing = FillMode.ZoomedNoLetterBoxing;
                    break;
                case FillMode.ZoomedNoLetterBoxing:
                    RendererSettings.LetterBoxing = FillMode.Fill;
                    break;
                case FillMode.Fill:
                    RendererSettings.LetterBoxing = FillMode.ApplyLetterBoxing;
                    break;
            }

            ApplyLetterBoxing();
        }
    }
}