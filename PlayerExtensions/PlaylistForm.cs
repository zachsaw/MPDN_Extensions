using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Windows.Forms.Design;

namespace Mpdn.PlayerExtensions.Example
{
    public partial class PlaylistForm : FormEx
    {
        private const double MAX_OPACITY = 1.0;
        private const double MIN_OPACITY = 0.7;

        private const string ACTIVE_INDICATOR_STR = "[*]";
        private const string INACTIVE_INDICATOR_STR = "[ ]";

        private bool m_FirstShow = true;
        private IPlayerControl m_PlayerControl;
        private int m_CurrentIndex = -1;

        public static int PlaylistCount { get; set; }

        public PlaylistForm()
        {
            InitializeComponent();

            openFileDialog.Filter = "Media files (all types) (*.avi; *.mp4; *.mkv; ...) |*.mkv;*.mp4;*.m4v;*.mp4v;*.3g2;*.3gp2;*.3gp;*.3gpp;*.mov;*.m2ts;*.ts;*.asf;*.wma;*.wmv;*.wm;*.asx,*.wax,*.wvx,*.wmx;*.wpl;*.dvr-ms;*.avi;*.mpg;*.mpeg;*.m1v;*.mp2;*.mp3;*.mpa;*.mpe;*.m3u;*.wav;*.mid;*.midi;*.rmi|All files (*.*)|*.*";
            Opacity = MIN_OPACITY;
        }

        public void SetPlayerControl(IPlayerControl playerControl)
        {
            m_PlayerControl = playerControl;
            Icon = m_PlayerControl.Form.Icon;
            m_PlayerControl.PlaybackCompleted += PlaybackCompleted;
        }

        public void Show(Form owner)
        {
            if (Visible)
                return;

            SetLocation(owner);
            timer.Enabled = true;
            listBox.Focus();
            base.Show(owner);
            PlaylistCount = listBox.Items.Count;
        }

        public void OpenPlaylist(string filename)
        {
            var playlist = File.ReadAllLines(filename);
            m_CurrentIndex = -1;
            listBox.Items.Clear();
            AddFiles(playlist.Select(s => s.Remove(0, ACTIVE_INDICATOR_STR.Length)), false);
            for (int i = 0; i < playlist.Length; i++)
            {
                var s = playlist[i];
                if (!s.StartsWith(ACTIVE_INDICATOR_STR)) 
                    continue;

                m_CurrentIndex = i-1;
                PlayNext();
                PlaylistCount = listBox.Items.Count;
                break;
            }
        }

        public void SavePlaylist(string filename)
        {
            var playlist =
                listBox.Items.Cast<PlaylistItem>()
                    .Select(
                        item =>
                            string.Format("{0}{1}", item.Active ? ACTIVE_INDICATOR_STR : INACTIVE_INDICATOR_STR,
                                item.FilePath));
            File.WriteAllLines(filename, playlist, Encoding.UTF8);
        }

        public void OpenPlaylist()
        {
            openPlaylistDialog.FileName = savePlaylistDialog.FileName;
            if (openPlaylistDialog.ShowDialog(m_PlayerControl.Form) != DialogResult.OK)
                return;

            OpenPlaylist(openPlaylistDialog.FileName);
            PlaylistCount = listBox.Items.Count;
        }

        public void SavePlaylist()
        {
            if (listBox.Items.Count == 0)
            {
                throw new InvalidOperationException("Nothing to save");
            }

            savePlaylistDialog.FileName = openPlaylistDialog.FileName;
            if (savePlaylistDialog.ShowDialog(m_PlayerControl.Form) != DialogResult.OK)
                return;

            SavePlaylist(savePlaylistDialog.FileName);
        }

        public void PlayNext()
        {
            m_CurrentIndex++;
            OpenMedia();
            if (m_CurrentIndex >= listBox.Items.Count)
            {
                m_CurrentIndex = listBox.Items.Count - 1;
            }
        }

        public void PlayPrevious()
        {
            m_CurrentIndex--;
            OpenMedia();
            if (m_CurrentIndex < 0)
            {
                m_CurrentIndex = 0;
            }
        }

        public void AddFiles(IEnumerable<string> files)
        {
            var startPlaying = listBox.Items.Count == 0;
            AddFiles(files, startPlaying);
            PlaylistCount = listBox.Items.Count;
        }

        public void AddFiles(IEnumerable<string> files, bool startPlaying)
        {
            foreach (var file in files)
            {
                listBox.Items.Add(new PlaylistItem(file));
            }

            if (!startPlaying || listBox.Items.Count == 0)
                return;

            if (m_PlayerControl.PlayerState != PlayerState.Closed &&
                m_PlayerControl.PlayerState != PlayerState.Stopped)
                return;

            m_CurrentIndex = -1;
            PlayNext();
            PlaylistCount = listBox.Items.Count;
        }

        #region Implementation

        private void PlaybackCompleted(object sender, EventArgs e)
        {
            if (m_PlayerControl.PlayerState == PlayerState.Closed)
                return;

            if (m_PlayerControl.MediaPosition == m_PlayerControl.MediaDuration)
            {
                PlayNext();
            }
        }

        private void SetLocation(Form owner)
        {
            if (!m_FirstShow) 
                return;

            Left = owner.Right - Width - (int) (5*ScaleFactor.Width);
            Top = owner.Bottom - Height - (int) (5*ScaleFactor.Height);
            m_FirstShow = false;
        }

        private void OpenMedia()
        {
            if (m_CurrentIndex < 0 || m_CurrentIndex >= listBox.Items.Count) 
                return;

            ResetActive();
            try
            {
                var item = ((PlaylistItem)listBox.Items[m_CurrentIndex]);
                m_PlayerControl.OpenMedia(item.FilePath);
                item.Active = true;
            }
            catch (Exception ex)
            {
                m_PlayerControl.HandleException(ex);
                PlayNext();
            }
            listBox.Invalidate();
        }

        private void ResetActive()
        {
            foreach (PlaylistItem item in listBox.Items)
            {
                item.Active = false;
            }
        }

        private void TimerTick(object sender, EventArgs e)
        {
            var pos = MousePosition;
            bool inForm = pos.X >= Left && pos.Y >= Top && pos.X < Right && pos.Y < Bottom;
            if (inForm)
            {
                if (Opacity < MAX_OPACITY)
                {
                    Opacity += 0.1;
                }
            }
            else if (Opacity > MIN_OPACITY)
            {
                Opacity -= 0.1;
            }
        }

        private void ButtonAddClick(object sender, EventArgs e)
        {
            if (openFileDialog.ShowDialog(this) != DialogResult.OK)
                return;

            var files =
                openFileDialog.FileNames
                    .Except(listBox.Items.Cast<PlaylistItem>().Select(item => item.FilePath))
                    .ToArray();

            AddFiles(files);
        }

        private void ButtonDelClick(object sender, EventArgs e)
        {
            RemoveSelectedItems();
        }

        private void RemoveSelectedItems()
        {
            var count = listBox.SelectedIndices.Count;
            if (count == 0)
                return;

            for (int i = count - 1; i >= 0; i--)
            {
                var index = listBox.SelectedIndices[i];
                listBox.Items.RemoveAt(index);
                if (m_CurrentIndex >= index)
                {
                    m_CurrentIndex--;
                }
            }
        }

        private void ButtonOpenClick(object sender, EventArgs e)
        {
            OpenPlaylist();
        }

        private void ButtonSaveClick(object sender, EventArgs e)
        {
            SavePlaylist();
        }

        private void PlaylistFormClosing(object sender, FormClosingEventArgs e)
        {
            if (e.CloseReason != CloseReason.UserClosing)
                return;

            e.Cancel = true;
            Hide();
            timer.Enabled = false;
        }

        private void FormKeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyData == Keys.Escape)
            {
                Hide();
            }
        }

        private void ListBoxDoubleClick(object sender, EventArgs e)
        {
            m_CurrentIndex = listBox.SelectedIndex - 1;
            PlayNext();
        }

        private void ListBoxDrawItem(object sender, DrawItemEventArgs e)
        {
            e.DrawBackground();
            if (e.Index >= 0)
            {
                Font toBeDisposed = null;
                var font = listBox.Font;
                var item = listBox.Items[e.Index];
                if (((PlaylistItem) item).Active)
                {
                    font = toBeDisposed = new Font(font.FontFamily, font.Size, FontStyle.Bold);
                }
                using (toBeDisposed)
                {
                    var format = new StringFormat
                    {
                        Trimming = StringTrimming.EllipsisCharacter,
                        Alignment = StringAlignment.Near,
                        LineAlignment = StringAlignment.Near
                    };
                    e.Graphics.DrawString(item.ToString(), font, SystemBrushes.WindowText, e.Bounds, format);
                }
            }
            e.DrawFocusRectangle();
        }

        private void ButtonLeftClick(object sender, EventArgs e)
        {
            PlayPrevious();
        }

        private void ButtonRightClick(object sender, EventArgs e)
        {
            PlayNext();
        }

        private void ListBoxSizeChanged(object sender, EventArgs e)
        {
            listBox.Invalidate();
        }

        private void SelectAllClick(object sender, EventArgs e)
        {
            for (int i = 0; i < listBox.Items.Count; i++)
            {
                listBox.SelectedIndices.Add(i);
            }
        }

        private void ListBoxDragEnter(object sender, DragEventArgs e)
        {
            e.Effect = DragDropEffects.Copy;
        }

        private void ListBoxDragDrop(object sender, DragEventArgs e)
        {
            if (!e.Data.GetDataPresent(DataFormats.FileDrop))
                return;

            var files = (string[]) e.Data.GetData(DataFormats.FileDrop);
            if (files == null) 
                return;

            HandleExternalDrop(files);
        }

        private void HandleExternalDrop(IList<string> files)
        {
            if (files.Count == 1)
            {
                var extension = Path.GetExtension(files[0]);
                if (extension != null && extension.ToLower() == ".mpl")
                {
                    // Playlist file
                    OpenPlaylist(files[0]);
                    return;
                }
            }
            // Enqueue files
            AddFiles(files);
        }

        private void ListBoxDropped(object sender, DroppedEventArgs e)
        {
            for (int i = 0; i < listBox.Items.Count; i++)
            {
                var item = (PlaylistItem) listBox.Items[i];
                if (!item.Active) 
                    continue;

                m_CurrentIndex = i;
                break;
            }
        }

        #endregion
    }

    public class PlaylistItem
    {
        public string FilePath { get; set; }
        public bool Active { get; set; }

        public PlaylistItem(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
            {
                throw new ArgumentNullException("filePath");
            }

            FilePath = filePath;
        }

        public override string ToString()
        {
            return Path.GetFileName(FilePath) ?? "???";
        }
    }

    [ToolStripItemDesignerAvailability(ToolStripItemDesignerAvailability.StatusStrip)]
    public class ButtonStripItem : ToolStripControlHostProxy
    {
        public ButtonStripItem()
            : base(new Button())
        {
        }
    }

    public class ToolStripControlHostProxy : ToolStripControlHost
    {
        public ToolStripControlHostProxy()
            : base(new Control())
        {
        }

        public ToolStripControlHostProxy(Control c)
            : base(c)
        {
        }
    }

    public class FormEx : Form
    {
        private float m_ScaleFactorHeight = -1f;
        private float m_ScaleFactorWidth = -1f;

        protected SizeF ScaleFactor { get; private set; }

        protected override void ScaleControl(SizeF factor, BoundsSpecified specified)
        {
            base.ScaleControl(factor, specified);

            if (!(m_ScaleFactorWidth < 0 || m_ScaleFactorHeight < 0))
                return;

            if (m_ScaleFactorWidth < 0 && specified.HasFlag(BoundsSpecified.Width))
            {
                m_ScaleFactorWidth = factor.Width;
            }
            if (m_ScaleFactorHeight < 0 && specified.HasFlag(BoundsSpecified.Height))
            {
                m_ScaleFactorHeight = factor.Height;
            }

            if (m_ScaleFactorWidth < 0 || m_ScaleFactorHeight < 0)
                return;

            ScaleFactor = new SizeF(m_ScaleFactorWidth, m_ScaleFactorHeight);
        }
    }
}
