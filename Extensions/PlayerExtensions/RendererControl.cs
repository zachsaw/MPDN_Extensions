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
    public class RenderControl : PlayerExtension
    {
        private const Category CATEGORY = Category.Play;
        private const string SUBCATEGORY = "Renderer";

        private readonly PlayerMenuItem m_LevelsMenu;
        private readonly PlayerMenuItem m_MatrixMenu;
        private readonly PlayerMenuItem m_ToggleLevelsMenu = new PlayerMenuItem(initiallyDisabled: true);
        private readonly PlayerMenuItem m_ToggleMatrixMenu = new PlayerMenuItem(initiallyDisabled: true);

        private PlayerMenuItem m_TvRangeMenu;
        private PlayerMenuItem m_PcRangeMenu;
        private PlayerMenuItem m_Bt601Menu;
        private PlayerMenuItem m_Bt709Menu;
        private PlayerMenuItem m_Bt2020Menu;

        private bool m_Enabled;

        public RenderControl()
        {
            m_LevelsMenu = new PlayerMenuItem(onAddedToPlayer: PopulateLevels, initiallyDisabled: true);
            m_MatrixMenu = new PlayerMenuItem(onAddedToPlayer: PopulateMatrix, initiallyDisabled: true);
        }

        public override void Initialize()
        {
            base.Initialize();
            PlayerControl.PlayerStateChanged += OnPlayerStateChanged;
            PlayerControl.SettingsChanged += OnSettingsChanged;
        }

        public override void Destroy()
        {
            PlayerControl.SettingsChanged -= OnSettingsChanged;
            PlayerControl.PlayerStateChanged -= OnPlayerStateChanged;
            base.Destroy();
        }

        protected static VideoRendererSettings RendererSettings
        {
            get { return PlayerControl.PlayerSettings.VideoRendererSettings;  }
        }

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("7563EB69-C62C-42FC-9054-0E33ABEE1D5F"),
                    Name = "Renderer Control",
                    Description = "Controls the video renderer"
                };
            }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(CATEGORY, SUBCATEGORY, "YUV levels", m_LevelsMenu), 
                    new Verb(CATEGORY, SUBCATEGORY, "YUV matrix", m_MatrixMenu), 
                    new Verb(CATEGORY, SUBCATEGORY, "Toggle YUV levels", "Ctrl+Shift+L", string.Empty, () => DoAction(ToggleLevels), m_ToggleLevelsMenu),
                    new Verb(CATEGORY, SUBCATEGORY, "Toggle YUV matrix", "Ctrl+Shift+M", string.Empty, () => DoAction(ToggleYuv), m_ToggleMatrixMenu)
                };
            }
        }

        private void PopulateLevels(PlayerMenuItem item)
        {
            // item == m_LevelsMenuParent
            m_TvRangeMenu = item.AddChild(new Verb("TV range", PlayerMenuItemType.Radio, () => DoAction(SelectTvRange)));
            m_PcRangeMenu = item.AddChild(new Verb("PC range", PlayerMenuItemType.Radio, () => DoAction(SelectPcRange)));
        }

        private void PopulateMatrix(PlayerMenuItem item)
        {
            // item == m_MatrixMenuParent
            m_Bt601Menu = item.AddChild(new Verb("BT.601", PlayerMenuItemType.Radio, () => DoAction(SelectBt601)));
            m_Bt709Menu = item.AddChild(new Verb("BT.709", PlayerMenuItemType.Radio, () => DoAction(SelectBt709)));
            m_Bt2020Menu = item.AddChild(new Verb("BT.2020", PlayerMenuItemType.Radio, () => DoAction(SelectBt2020)));
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
                          !(Renderer.InputFormat == FrameBufferInputFormat.Rgb24 ||
                            Renderer.InputFormat == FrameBufferInputFormat.Rgb32 ||
                            Renderer.InputFormat == FrameBufferInputFormat.Rgb48);

            m_Enabled = enabled;

            m_LevelsMenu.Enabled = enabled;
            m_MatrixMenu.Enabled = enabled;
            m_ToggleLevelsMenu.Enabled = enabled;
            m_ToggleMatrixMenu.Enabled = enabled;

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
            foreach (var child in m_LevelsMenu.Children)
            {
                child.Checked = false;
            }
            foreach (var child in m_MatrixMenu.Children)
            {
                child.Checked = false;
            }
        }

        private void SelectTvRange()
        {
            var settings = RendererSettings.OutputLevels;
            switch (Renderer.Colorimetric)
            {
                case YuvColorimetric.FullRangePc601: RendererSettings.OutputLevels = YuvColorimetric.ItuBt601; break;
                case YuvColorimetric.FullRangePc709: RendererSettings.OutputLevels = YuvColorimetric.ItuBt709; break;
                case YuvColorimetric.FullRangePc2020: RendererSettings.OutputLevels = YuvColorimetric.ItuBt2020; break;
            }
            Apply();
            RendererSettings.OutputLevels = settings;
        }

        private void SelectPcRange()
        {
            var settings = RendererSettings.OutputLevels;
            switch (Renderer.Colorimetric)
            {
                case YuvColorimetric.ItuBt601: RendererSettings.OutputLevels = YuvColorimetric.FullRangePc601; break;
                case YuvColorimetric.ItuBt709: RendererSettings.OutputLevels = YuvColorimetric.FullRangePc709; break;
                case YuvColorimetric.ItuBt2020: RendererSettings.OutputLevels = YuvColorimetric.FullRangePc2020; break;
            }
            Apply();
            RendererSettings.OutputLevels = settings;
        }

        private void SelectBt601()
        {
            var settings = RendererSettings.OutputLevels;
            switch (Renderer.Colorimetric)
            {
                case YuvColorimetric.FullRangePc709: 
                case YuvColorimetric.FullRangePc2020: 
                    RendererSettings.OutputLevels = YuvColorimetric.FullRangePc601;
                    break;
                case YuvColorimetric.ItuBt709:
                case YuvColorimetric.ItuBt2020: 
                    RendererSettings.OutputLevels = YuvColorimetric.ItuBt601; 
                    break;
            }
            Apply();
            RendererSettings.OutputLevels = settings;
        }

        private void SelectBt709()
        {
            var settings = RendererSettings.OutputLevels;
            switch (Renderer.Colorimetric)
            {
                case YuvColorimetric.FullRangePc601:
                case YuvColorimetric.FullRangePc2020:
                    RendererSettings.OutputLevels = YuvColorimetric.FullRangePc709;
                    break;
                case YuvColorimetric.ItuBt601:
                case YuvColorimetric.ItuBt2020:
                    RendererSettings.OutputLevels = YuvColorimetric.ItuBt709;
                    break;
            }
            Apply();
            RendererSettings.OutputLevels = settings;
        }

        private void SelectBt2020()
        {
            var settings = RendererSettings.OutputLevels;
            switch (Renderer.Colorimetric)
            {
                case YuvColorimetric.FullRangePc601:
                case YuvColorimetric.FullRangePc709:
                    RendererSettings.OutputLevels = YuvColorimetric.FullRangePc2020;
                    break;
                case YuvColorimetric.ItuBt601:
                case YuvColorimetric.ItuBt709:
                    RendererSettings.OutputLevels = YuvColorimetric.ItuBt2020;
                    break;
            }
            Apply();
            RendererSettings.OutputLevels = settings;
        }

        private void ToggleLevels()
        {
            var settings = RendererSettings.OutputLevels;
            switch (Renderer.Colorimetric)
            {
                case YuvColorimetric.FullRangePc601: RendererSettings.OutputLevels = YuvColorimetric.ItuBt601; break;
                case YuvColorimetric.FullRangePc709: RendererSettings.OutputLevels = YuvColorimetric.ItuBt709; break;
                case YuvColorimetric.FullRangePc2020: RendererSettings.OutputLevels = YuvColorimetric.ItuBt2020; break;
                case YuvColorimetric.ItuBt601: RendererSettings.OutputLevels = YuvColorimetric.FullRangePc601; break;
                case YuvColorimetric.ItuBt709: RendererSettings.OutputLevels = YuvColorimetric.FullRangePc709; break;
                case YuvColorimetric.ItuBt2020: RendererSettings.OutputLevels = YuvColorimetric.FullRangePc2020; break;
            }
            Apply();
            RendererSettings.OutputLevels = settings;
        }

        private void ToggleYuv()
        {
            var settings = RendererSettings.OutputLevels;
            switch (Renderer.Colorimetric)
            {
                case YuvColorimetric.FullRangePc601: RendererSettings.OutputLevels = YuvColorimetric.FullRangePc709; break;
                case YuvColorimetric.FullRangePc709: RendererSettings.OutputLevels = YuvColorimetric.FullRangePc2020; break;
                case YuvColorimetric.FullRangePc2020: RendererSettings.OutputLevels = YuvColorimetric.FullRangePc601; break;
                case YuvColorimetric.ItuBt601: RendererSettings.OutputLevels = YuvColorimetric.ItuBt709; break;
                case YuvColorimetric.ItuBt709: RendererSettings.OutputLevels = YuvColorimetric.ItuBt2020; break;
                case YuvColorimetric.ItuBt2020: RendererSettings.OutputLevels = YuvColorimetric.ItuBt601; break;
            }
            Apply();
            RendererSettings.OutputLevels = settings;
        }

        private void Apply()
        {
            PlayerControl.ShowOsdText("Colour space: " + RendererSettings.OutputLevels.ToDescription());
            PlayerControl.RefreshSettings();
        }

        private void UpdateControls()
        {
            ClearSelections();
            if (PlayerControl.PlayerState != PlayerState.Playing && PlayerControl.PlayerState != PlayerState.Paused)
                return;

            switch (Renderer.Colorimetric)
            {
                case YuvColorimetric.FullRangePc601:
                    m_PcRangeMenu.Checked = true;
                    m_Bt601Menu.Checked = true;
                    break;
                case YuvColorimetric.FullRangePc709:
                    m_PcRangeMenu.Checked = true;
                    m_Bt709Menu.Checked = true;
                    break;
                case YuvColorimetric.FullRangePc2020:
                    m_PcRangeMenu.Checked = true;
                    m_Bt2020Menu.Checked = true;
                    break;
                case YuvColorimetric.ItuBt601:
                    m_TvRangeMenu.Checked = true;
                    m_Bt601Menu.Checked = true;
                    break;
                case YuvColorimetric.ItuBt709:
                    m_TvRangeMenu.Checked = true;
                    m_Bt709Menu.Checked = true;
                    break;
                case YuvColorimetric.ItuBt2020:
                    m_TvRangeMenu.Checked = true;
                    m_Bt2020Menu.Checked = true;
                    break;
            }
        }
    }
}