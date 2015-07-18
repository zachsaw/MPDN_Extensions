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
using System.Linq;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.PlayerExtensions.Interfaces;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class AudioSettings : PlayerExtension
    {
        private const Category CATEGORY = Category.Play;
        private const string SUBCATEGORY = "Audio Delay";

        private static readonly Guid s_ClsIdLavAudioDecoder = new Guid("E8E73B6B-4CB3-44A4-BE99-4F7BCB96E491");

        private readonly PlayerMenuItem m_AddDelayMenu = new PlayerMenuItem(initiallyDisabled: true);
        private readonly PlayerMenuItem m_MinusDelayMenu = new PlayerMenuItem(initiallyDisabled: true);
        private ILAVAudioSettings m_LavAudioSettings;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("83400FF2-623C-4E35-BDB7-E7EDEC968286"),
                    Name = "Audio Settings",
                    Description = "Changes LAV Audio Decoder settings"
                };
            }
        }

        protected override string ConfigFileName
        {
            get { return "AudioSettings"; }
        }

        public override void Initialize()
        {
            base.Initialize();

            Media.Loaded += MediaLoaded;
            Player.StateChanged += PlayerStateChanged;
        }

        public override void Destroy()
        {
            base.Destroy();

            Media.Loaded -= MediaLoaded;
            Player.StateChanged -= PlayerStateChanged;
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(CATEGORY, SUBCATEGORY, "Add 5ms", "+", string.Empty, () => AddDelay(5), m_AddDelayMenu),
                    new Verb(CATEGORY, SUBCATEGORY, "Minus 5ms", "-", string.Empty, () => AddDelay(-5), m_MinusDelayMenu)
                };
            }
        }

        private void MediaLoaded(object sender, EventArgs e)
        {
            var audioDecoder = Player.Filters.Audio.FirstOrDefault(f => f.ClsId == s_ClsIdLavAudioDecoder);
            if (audioDecoder == null)
                return;

            ComThread.Do(() =>
            {
                var settings = (ILAVAudioSettings) audioDecoder.Base;
                m_LavAudioSettings = settings;
            });

            m_AddDelayMenu.Enabled = true;
            m_MinusDelayMenu.Enabled = true;
        }

        private void PlayerStateChanged(object sender, PlayerStateEventArgs e)
        {
            if (e.NewState != PlayerState.Closed)
                return;

            m_LavAudioSettings = null;
            m_AddDelayMenu.Enabled = false;
            m_MinusDelayMenu.Enabled = false;
        }

        private void AddDelay(int delayMs)
        {
            if (Player.State == PlayerState.Closed)
                return;

            int delay = 0;
            ComThread.Do(() =>
            {
                bool enabled;
                m_LavAudioSettings.GetAudioDelay(out enabled, out delay);

                if (!enabled)
                {
                    delay = 0;
                }

                delay += delayMs;
                m_LavAudioSettings.SetAudioDelay(true, delay);
            });

            Player.OsdText.Show(string.Format("Audio Delay: {0}ms", delay));
        }
    }
}
