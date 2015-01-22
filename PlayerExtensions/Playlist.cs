using System;
using System.Collections.Generic;
using System.IO;
using System.Windows.Forms;

namespace Mpdn.PlayerExtensions.GitHub
{
    public class Playlist : PlayerExtension
    {
        private const string SUBCATEGORY = "Playlist";

        private readonly PlaylistForm m_Form = new PlaylistForm();

        protected override PlayerExtensionDescriptor ScriptDescriptor
        {
            get
            {
                return new PlayerExtensionDescriptor
                {
                    Guid = new Guid("A1997E34-D67B-43BB-8FE6-55A71AE7184B"),
                    Name = "Playlist",
                    Description = "Playlist Support"
                };
            }
        }

        public override void Initialize()
        {
            base.Initialize();

            m_Form.SetPlayerControl(PlayerControl);

            PlayerControl.DragEnter += OnDragEnter;
            PlayerControl.DragDrop += OnDragDrop;
            PlayerControl.CommandLineFileOpen += OnCommandLineFileOpen;
        }

        public override void Destroy()
        {
            base.Destroy();

            m_Form.Dispose();
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.File, string.Empty, "Open Playlist", "Ctrl+Alt+O", string.Empty, OpenPlaylist),
                    new Verb(Category.View, string.Empty, "Playlist", "Ctrl+Alt+P", string.Empty, ViewPlaylist),
                    new Verb(Category.Play, SUBCATEGORY, "Next", "Ctrl+Alt+N", string.Empty, () => m_Form.PlayNext()),
                    new Verb(Category.Play, SUBCATEGORY, "Previous", "Ctrl+Alt+B", string.Empty, () => m_Form.PlayPrevious())
                };
            }
        }

        private void OpenPlaylist()
        {
            m_Form.Show(PlayerControl.Form);
            m_Form.OpenPlaylist();
        }

        private void ViewPlaylist()
        {
            m_Form.Show(PlayerControl.Form);
        }

        private void OnDragEnter(object sender, PlayerControlEventArgs<DragEventArgs> e)
        {
            e.Handled = true;
            e.InputArgs.Effect = DragDropEffects.Copy;
        }

        private void OnDragDrop(object sender, PlayerControlEventArgs<DragEventArgs> e)
        {
            var files = (string[]) e.InputArgs.Data.GetData(DataFormats.FileDrop);
            if (files.Length > 1)
            {
                e.Handled = true;
                // Add multiple files to playlist
                m_Form.AddFiles(files);
            }
            else
            {
                var filename = files[0];
                if (IsPlaylistFile(filename))
                {
                    // Playlist file
                    m_Form.OpenPlaylist(filename);
                    e.Handled = true;
                }
            }
        }

        private void OnCommandLineFileOpen(object sender, CommandLineFileOpenEventArgs e)
        {
            if (!IsPlaylistFile(e.Filename)) 
                return;

            e.Handled = true;
            m_Form.OpenPlaylist(e.Filename);
        }

        private static bool IsPlaylistFile(string filename)
        {
            var extension = Path.GetExtension(filename);
            return extension != null && extension.ToLower() == ".mpl";
        }
    }
}
