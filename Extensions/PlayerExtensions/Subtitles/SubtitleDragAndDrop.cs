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
using System.IO;
using System.Linq;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions.Subtitles
{
    public class SubtitleDragAndDrop : PlayerExtension
    {
        private string m_SubtitleFile;
        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("A2B9AF1B-C1A7-4338-9D75-012B4690C865"),
                    Name = "Subtitle Drag and Drop",
                    Description = "Add Drag and Drop support for subtitle"
                };
            }
        }

        protected override string ConfigFileName
        {
            get { return "SubtitleDragAndDrop"; }
        }

        public override void Initialize()
        {
            base.Initialize();
            Player.DragDrop += PlayerOnDragDrop;
            Player.StateChanged +=PlayerOnStateChanged;
        }

        private void PlayerOnStateChanged(object sender, PlayerStateEventArgs playerStateEventArgs)
        {
            if (playerStateEventArgs.OldState == PlayerState.Closed && m_SubtitleFile != null)
            {
                LoadSubtitleFile(m_SubtitleFile);
                m_SubtitleFile = null;
            } 
        }

        private void PlayerOnDragDrop(object sender, PlayerControlEventArgs<DragEventArgs> playerControlEventArgs)
        {
            var files = (string[]) playerControlEventArgs.InputArgs.Data.GetData(DataFormats.FileDrop);

            if (files != null)
            foreach (var file in files.Where(SubtitleManager.IsSubtitleFile))
            {
                playerControlEventArgs.Handled = true;
                LoadSubtitleFile(file);
                break;
            }
        }

        private void LoadSubtitleFile(string file)
        {
            if (Player.State == PlayerState.Closed)
            {
                m_SubtitleFile = file;
                //Player.OsdText.Show("Subtitle will be loaded with media", 3000); //Doesn't seem to work
                return;
            }

            Media.Pause(false);
            var subtitleLoaded = string.Format("Subtitle Loaded: {0}", Path.GetFileName(file));
            Player.OsdText.Show(SubtitleManager.LoadFile(file)
                ? subtitleLoaded
                : "Impossible to load Subtitle file.");
            Media.Play(false);
        }

        public override void Destroy()
        {
            base.Destroy();
            Player.DragDrop -= PlayerOnDragDrop;
            Player.StateChanged -= PlayerOnStateChanged;
        }
    }
}