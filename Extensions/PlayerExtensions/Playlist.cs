using System;
using System.Collections.Generic;
using System.IO;

namespace Mpdn.PlayerExtensions.Playlist
{
    public class Playlist : PlayerExtension
    {
        private const string Subcategory = "Playlist";

        private readonly PlaylistForm form = new PlaylistForm();

        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("A1997E34-D67B-43BB-8FE6-55A71AE7184B"),
                    Name = "Playlist",
                    Description = "Adds playlist support with advanced capabilities",
                    Copyright = "Enhanced by Garteal"
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();
            form.Setup();

            PlayerControl.CommandLineFileOpen += OnCommandLineFileOpen;
        }

        public override void Destroy()
        {
            base.Destroy();
            form.Dispose();
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.File, string.Empty, "Open Playlist", "Ctrl+Alt+O", string.Empty, OpenPlaylist),
                    new Verb(Category.View, string.Empty, "Playlist", "Ctrl+Alt+P", string.Empty, ViewPlaylist),
                    new Verb(Category.Play, Subcategory, "Next", "Ctrl+Alt+N", string.Empty, () => form.PlayNext()),
                    new Verb(Category.Play, Subcategory, "Previous", "Ctrl+Alt+B", string.Empty, () => form.PlayPrevious())
                };
            }
        }

        private void OpenPlaylist()
        {
            form.Show(PlayerControl.Form);
            form.OpenPlaylist();
        }

        private void ViewPlaylist()
        {
            form.Show(PlayerControl.Form);
        }

        private void OnCommandLineFileOpen(object sender, CommandLineFileOpenEventArgs e)
        {
            if (!IsPlaylistFile(e.Filename)) return;
            e.Handled = true;
            form.OpenPlaylist(e.Filename);
        }

        private static bool IsPlaylistFile(string filename)
        {
            var extension = Path.GetExtension(filename);
            return extension != null && extension.ToLower() == ".mpl";
        }
    }
}
