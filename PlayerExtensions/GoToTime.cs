using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace Mpdn.PlayerExtensions.Example
{
    public class GoToTime : PlayerExtension
    {
        public override ExtensionDescriptor Descriptor
        {
            get
            {
                return new ExtensionDescriptor
                {
                    Guid = new Guid("7C3BA1E2-EE7B-47D2-B174-6AE76D65EC04"),
                    Name = "Go To Time",
                    Description = "Jump to a specified timecode in media",
                    Copyright = "Copyright Example © 2014. All rights reserved."
                };
            }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.Play, string.Empty, "Go To...", "Ctrl+G", string.Empty, GotoPosition)
                };
            }
        }

        private void GotoPosition()
        {
            using (var form = new GoToTimeForm())
            {
                if (form.ShowDialog(PlayerControl.Form) != DialogResult.OK)
                    return;

                if (PlayerControl.PlayerState == PlayerState.Closed)
                    return;

                if (PlayerControl.PlayerState == PlayerState.Stopped)
                {
                    PlayerControl.PauseMedia(false);
                }

                PlayerControl.SeekMedia(form.Position * 1000);
            }
        }
    }
}