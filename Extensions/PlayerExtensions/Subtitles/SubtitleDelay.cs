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
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions.Subtitles
{
    public class SubtitleDelay : PlayerExtension
    {
        private const Category CATEGORY = Category.Play;
        private const string SUBCATEGORY = "Subtitle Delay";
        private readonly PlayerMenuItem m_AddDelayMenu = new PlayerMenuItem(initiallyDisabled: true);
        private readonly PlayerMenuItem m_MinusDelayMenu = new PlayerMenuItem(initiallyDisabled: true);
        private readonly PlayerMenuItem m_ResetDelayMenu = new PlayerMenuItem(initiallyDisabled: true);
        private SubtitleManager.SubtitleTiming currentTiming;
        private SubtitleManager.SubtitleTiming starTiming;

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("A82960C0-FE47-41B3-BD36-4E4DE86F0A25"),
                    Name = "Subtitle Delay",
                    Description = "Changes the subtitle delay"
                };
            }
        }

        protected override string ConfigFileName
        {
            get { return "SubtitleDelay"; }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(CATEGORY, SUBCATEGORY, "Add 250ms", "F1", string.Empty, () => AddDelay(250), m_AddDelayMenu),
                    new Verb(CATEGORY, SUBCATEGORY, "Minus 250ms", "F2", string.Empty, () => AddDelay(-250),
                        m_MinusDelayMenu),
                    new Verb(CATEGORY, SUBCATEGORY, "Reset", ResetDelay, m_ResetDelayMenu)
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();

            Player.StateChanged += PlayerStateChanged;
        }

        public override void Destroy()
        {
            base.Destroy();

            Player.StateChanged -= PlayerStateChanged;
        }

        private void PlayerStateChanged(object sender, PlayerStateEventArgs e)
        {
            if (e.OldState == PlayerState.Closed)
            {
                starTiming = SubtitleManager.GetTiming();
                if (starTiming == null)
                    return;

                SetDefaultCurrentTiming();

                m_AddDelayMenu.Enabled = true;
                m_MinusDelayMenu.Enabled = true;
                m_ResetDelayMenu.Enabled = true;
            }
            else if (e.NewState == PlayerState.Closed)
            {
                m_AddDelayMenu.Enabled = false;
                m_MinusDelayMenu.Enabled = false;
                m_ResetDelayMenu.Enabled = false;
            }
        }

        private void SetDefaultCurrentTiming()
        {
            currentTiming = new SubtitleManager.SubtitleTiming(starTiming.Delay, starTiming.SpeedMultiplier,
                starTiming.SpeedDivisor);
        }

        private void ResetDelay()
        {
            if (starTiming == null)
                return;

            SubtitleManager.SetTiming(starTiming);
            SetDefaultCurrentTiming();

            ShowDelayText(0);
        }

        private static void ShowDelayText(int delay)
        {
            Player.OsdText.Show(string.Format("Subtitle Delay: {0}ms", delay));
        }

        private void AddDelay(int delayMs)
        {
            if (starTiming == null)
                return;

            currentTiming.Delay += delayMs;
            SubtitleManager.SetTiming(currentTiming);

            ShowDelayText(currentTiming.Delay);
        }
    }
}