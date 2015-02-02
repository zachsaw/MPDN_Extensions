using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Windows.Forms.Design;

namespace Mpdn.PlayerExtensions.Playlist
{
    public partial class PlaylistForm : FormEx
    {
        public static int PlaylistCount { get; set; }

        private const double MaxOpacity = 1.0;
        private const double MinOpacity = 0.7;
        private const string ActiveIndicator = "[*]";
        private const string InactiveIndicator = "[ ]";

        private bool firstShow = true;
        private bool wasShowing;

        private List<PlaylistItem> playList;
        private PlaylistItem currentPlayItem;

        private int currentPlayIndex = -1;
        private long previousChapterPosition;

        public PlaylistForm()
        {
            InitializeComponent();
            Opacity = MinOpacity;
        }

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();

                if (playList != null)
                {
                    PlayerControl.PlayerStateChanged -= PlayerStateChanged;
                    PlayerControl.PlaybackCompleted -= PlaybackCompleted;
                    PlayerControl.FrameDecoded -= FrameDecoded;
                    PlayerControl.FramePresented -= FramePresented;
                    PlayerControl.EnteringFullScreenMode -= EnteringFullScreenMode;
                    PlayerControl.ExitedFullScreenMode -= ExitedFullScreenMode;
                }
            }
            base.Dispose(disposing);
        }

        public void Show(Control owner)
        {
            if (PlayerControl.InFullScreenMode)
                return;

            Hide();
            SetLocation(owner);
            timer.Enabled = true;
            dgv_PlayList.Focus();
            base.Show(owner);
        }

        public void Setup()
        {
            if (playList != null)
                return;

            Icon = PlayerControl.ApplicationIcon;

            dgv_PlayList.CellFormatting += dgv_PlayList_CellFormatting;
            dgv_PlayList.CellPainting += dgv_PlayList_CellPainting;
            dgv_PlayList.CellDoubleClick += dgv_PlayList_CellDoubleClick;
            dgv_PlayList.CellEndEdit += dgv_PlayList_CellEndEdit;
            dgv_PlayList.EditingControlShowing += dgv_PlayList_EditingControlShowing;

            PlayerControl.PlayerStateChanged += PlayerStateChanged;
            PlayerControl.PlaybackCompleted += PlaybackCompleted;
            PlayerControl.FrameDecoded += FrameDecoded;
            PlayerControl.FramePresented += FramePresented;
            PlayerControl.EnteringFullScreenMode += EnteringFullScreenMode;
            PlayerControl.ExitedFullScreenMode += ExitedFullScreenMode;

            playList = new List<PlaylistItem>();
        }

        public void ClearPlaylist()
        {
            playList.Clear();
            currentPlayIndex = -1;
        }

        public void PopulatePlaylist()
        {
            dgv_PlayList.Rows.Clear();
            if (playList.Count == 0) return;

            foreach (var i in playList)
            {
                if (i.SkipChapters != null)
                {
                    if (i.EndChapter != -1)
                    {
                        dgv_PlayList.Rows.Add(new Bitmap(25, 25), i.FilePath, String.Join(",", i.SkipChapters), i.EndChapter);
                    }
                    else
                    {
                        dgv_PlayList.Rows.Add(new Bitmap(25, 25), i.FilePath, String.Join(",", i.SkipChapters));
                    }
                }
                else
                {
                    dgv_PlayList.Rows.Add(new Bitmap(25, 25), i.FilePath);
                }
            }

            if (PlayerControl.MediaFilePath != "" && playList.Count > 0)
            {
                currentPlayIndex = (playList.FindIndex(i => i.Active) > -1) ? playList.FindIndex(i => i.Active) : 0;
            }

            PlaylistCount = playList.Count;
        }

        public void NewPlaylist()
        {
            ClearPlaylist();
            PopulatePlaylist();
            CloseMedia();
        }

        public void OpenPlaylist()
        {
            openPlaylistDialog.FileName = savePlaylistDialog.FileName;
            if (openPlaylistDialog.ShowDialog(PlayerControl.Form) != DialogResult.OK) return;

            OpenPlaylist(openPlaylistDialog.FileName);
        }

        public void OpenPlaylist(string fileName)
        {
            ClearPlaylist();

            try
            {
                using (var sr = new StreamReader(fileName))
                {
                    string line;

                    while ((line = sr.ReadLine()) != null)
                    {
                        if (line.Contains("|"))
                        {
                            ParseWithChapters(line);
                        }
                        else
                        {
                            ParseWithoutChapters(line);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Invalid or corrupt playlist file.\nAdditional info: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            PopulatePlaylist();
            PlayActive();
        }

        public void SavePlaylist()
        {
            if (playList.Count == 0) return;

            savePlaylistDialog.FileName = openPlaylistDialog.FileName;
            if (savePlaylistDialog.ShowDialog(PlayerControl.Form) != DialogResult.OK) return;

            SavePlaylist(savePlaylistDialog.FileName);
        }

        public void SavePlaylist(string filename)
        {
            IEnumerable<string> playlist;
            bool containsChapter = false;

            foreach (var item in playList)
            {
                if (item.HasChapter)
                {
                    containsChapter = true;
                }
            }

            if (containsChapter)
            {
                playlist =
                    playList
                        .Select(
                            item =>
                                string.Format("{0}{1} | SkipChapter: {2} | EndChapter: {3}",
                                    item.Active ? ActiveIndicator : InactiveIndicator,
                                    item.FilePath, item.HasChapter ? String.Join(",", item.SkipChapters) : "0",
                                    item.EndChapter > -1 ? item.EndChapter : 0));
            }
            else
            {
                playlist =
                    playList
                        .Select(
                            item =>
                                string.Format("{0}{1}", item.Active ? ActiveIndicator : InactiveIndicator,
                                    item.FilePath));
            }

            File.WriteAllLines(filename, playlist, Encoding.UTF8);
        }

        public void PlayActive()
        {
            currentPlayIndex = -1;

            foreach (var item in playList)
            {
                currentPlayIndex++;
                if (!item.Active) continue;
                OpenMedia();
                return;
            }

            currentPlayIndex = 0;
            OpenMedia();
        }

        public void PlayNext()
        {
            currentPlayIndex++;
            OpenMedia();

            if (currentPlayIndex < playList.Count) return;
            currentPlayIndex = playList.Count - 1;
        }

        public void PlayPrevious()
        {
            currentPlayIndex--;
            OpenMedia();

            if (currentPlayIndex >= 0) return;
            currentPlayIndex = 0;
        }

        public void AddFiles(string[] fileNames)
        {
            var startPlaying = playList.Count == 0 && PlayerControl.PlayerState == PlayerState.Closed;

            var files = fileNames.Except(playList.Select(item => item.FilePath)).ToArray();

            foreach (var item in files.Select(s => new PlaylistItem(s, false) { EndChapter = -1 }))
            {
                playList.Add(item);
            }

            PopulatePlaylist();

            if (!startPlaying) return;

            currentPlayIndex = 0;
            OpenMedia();
        }

        public void CloseMedia()
        {
            try
            {
                currentPlayIndex = -1;
                currentPlayItem = null;
                Text = "Playlist";
                PlayerControl.CloseMedia();
            }
            catch (Exception ex)
            {
                PlayerControl.HandleException(ex);
            }
        }

        private void SetLocation(Control owner)
        {
            if (!firstShow) return;
            var screen = Screen.FromControl(owner);
            var screenBounds = screen.WorkingArea;
            var p = owner.PointToScreen(new Point(owner.Right, owner.Bottom));
            var left = p.X - Width / (int)(5 * ScaleFactor.Width);
            var top = p.Y - Height / (int)(5 * ScaleFactor.Height);
            Left = left + Width > screenBounds.Right ? screenBounds.Right - Width : left;
            Top = top + Height > screenBounds.Bottom ? screenBounds.Bottom - Height : top;
            firstShow = false;
        }

        private void TimerTick(object sender, EventArgs e)
        {
            HandleOpacity();
        }

        private void HandleOpacity()
        {
            var pos = MousePosition;
            bool inForm = pos.X >= Left && pos.Y >= Top && pos.X < Right && pos.Y < Bottom;

            if (inForm)
            {
                if (Opacity < MaxOpacity)
                {
                    Opacity += 0.1;
                }
            }
            else if (Opacity > MinOpacity)
            {
                Opacity -= 0.1;
            }
        }

        void dgv_PlayList_CellFormatting(object sender, DataGridViewCellFormattingEventArgs e)
        {
            var skipChapterCell = dgv_PlayList.Rows[e.RowIndex].Cells[2];
            var endChapterCell = dgv_PlayList.Rows[e.RowIndex].Cells[3];

            if (skipChapterCell.IsInEditMode || endChapterCell.IsInEditMode)
            {
                e.CellStyle.ForeColor = Color.Black;
            }
        }

        void dgv_PlayList_CellPainting(object sender, DataGridViewCellPaintingEventArgs e)
        {
            e.Paint(e.ClipBounds, DataGridViewPaintParts.All);

            bool paintPlayRow = currentPlayItem != null && e.RowIndex == currentPlayIndex;
            if (!paintPlayRow) return;

            var brush = new SolidBrush(Color.FromArgb(42, 127, 183));

            if (e.ColumnIndex == 0)
            {
                var rect = new Rectangle(e.CellBounds.X + 15, e.CellBounds.Y + 4, e.CellBounds.Width, e.CellBounds.Height - 9);
                var playIcon = (Bitmap) PlayButton.BackgroundImage;
                var offset = new Point(e.CellBounds.X, e.CellBounds.Y + 2);
                e.Graphics.FillRectangle(brush, rect);
                e.Graphics.DrawImage(playIcon, offset);
            }
            else
            {
                var rect = new Rectangle(e.CellBounds.X, e.CellBounds.Y + 4, e.CellBounds.Width, e.CellBounds.Height - 9);
                e.Graphics.FillRectangle(brush, rect);
            }

            e.Paint(e.ClipBounds, DataGridViewPaintParts.ContentForeground);
            e.Handled = true;
        }

        private void dgv_PlayList_CellDoubleClick(object sender, DataGridViewCellEventArgs e)
        {
            if (dgv_PlayList.Rows.Count <= 0) return;
            if (e.ColumnIndex > 1) return;
            currentPlayIndex = e.RowIndex;
            OpenMedia();
        }

        private void dgv_PlayList_CellEndEdit(object sender, DataGridViewCellEventArgs e)
        {
            ParseChapterInput();
        }

        void dgv_PlayList_EditingControlShowing(object sender, DataGridViewEditingControlShowingEventArgs e)
        {
            e.Control.KeyPress -= dgv_PlayList_HandleInput;
            if (dgv_PlayList.CurrentCell.ColumnIndex <= 1) return;

            var tb = e.Control as TextBox;
            if (tb != null)
            {
                tb.KeyPress += dgv_PlayList_HandleInput;
            }
        }

        private void dgv_PlayList_HandleInput(object sender, KeyPressEventArgs e)
        {
            if (!char.IsControl(e.KeyChar) && !char.IsDigit(e.KeyChar) && e.KeyChar != ',' && e.KeyChar != ' ' && dgv_PlayList.CurrentCell.ColumnIndex == 2)
            {
                var toolTip = new ToolTip();
                var cell = dgv_PlayList.CurrentCell;
                var cellDisplayRect = dgv_PlayList.GetCellDisplayRectangle(cell.ColumnIndex, cell.RowIndex, false);
                toolTip.Show("Only numbers are allowed. You may separate them with a comma or a space.", dgv_PlayList,
                              cellDisplayRect.X + cell.Size.Width / 2,
                              cellDisplayRect.Y + cell.Size.Height / 2,
                              2000);

                e.Handled = true;
            }

            if (!char.IsControl(e.KeyChar) && !char.IsDigit(e.KeyChar) && dgv_PlayList.CurrentCell.ColumnIndex == 3)
            {
                var toolTip = new ToolTip();
                var cell = dgv_PlayList.CurrentCell;
                var cellDisplayRect = dgv_PlayList.GetCellDisplayRectangle(cell.ColumnIndex, cell.RowIndex, false);
                toolTip.Show("Only numbers are allowed.", dgv_PlayList,
                              cellDisplayRect.X + cell.Size.Width / 2,
                              cellDisplayRect.Y + cell.Size.Height / 2,
                              2000);

                e.Handled = true;
            }
        }

        private void PlayerStateChanged(object sender, PlayerStateEventArgs e)
        {
            if (currentPlayItem == null) return;
            Text = PlayerControl.PlayerState + " - " + currentPlayItem.FilePath;
        }

        private void PlaybackCompleted(object sender, EventArgs e)
        {
            if (PlayerControl.PlayerState == PlayerState.Closed) return;
            if (PlayerControl.MediaPosition == PlayerControl.MediaDuration)
            {
                PlayNext();
            }
        }

        private void EnteringFullScreenMode(object sender, EventArgs e)
        {
            wasShowing = Visible;
            Hide();
        }

        private void ExitedFullScreenMode(object sender, EventArgs e)
        {
            if (wasShowing)
            {
                Show(PlayerControl.VideoPanel);
            }
        }

        private void PlaylistFormClosing(object sender, FormClosingEventArgs e)
        {
            if (e.CloseReason != CloseReason.UserClosing) return;
            e.Cancel = true;
            Hide();
            timer.Enabled = false;
        }

        private void FrameDecoded(object sender, FrameEventArgs e)
        {
            if (PlayerControl.MediaFilePath != "" && PlayerControl.Chapters.Count != 0 && currentPlayItem != null && currentPlayItem.HasChapter)
            {
                previousChapterPosition = GetChapters().Aggregate((prev, next) => e.SampleTime >= prev.Position && e.SampleTime <= next.Position ? prev : next).Position;
            }
        }

        private void FramePresented(object sender, FrameEventArgs e)
        {
            if (PlayerControl.MediaFilePath != "" && PlayerControl.Chapters.Count != 0 && currentPlayItem != null && currentPlayItem.HasChapter)
            {
                if (e.SampleTime >= previousChapterPosition)
                {
                    int currentChapterIndex = GetChapterIndexByPosition(previousChapterPosition);

                    if (currentPlayItem.SkipChapters.Contains(currentChapterIndex) && currentChapterIndex < PlayerControl.Chapters.Count)
                    {
                        SelectChapter(currentChapterIndex);
                    }
                    else if (currentChapterIndex == currentPlayItem.EndChapter)
                    {
                        PlayNext();
                    }
                }
            }
        }

        private void ParseChapterInput()
        {
            try
            {
                for (int i = 0; i < dgv_PlayList.Rows.Count; i++)
                {
                    var skipChapterCell = dgv_PlayList.Rows[i].Cells[2];
                    var endChapterCell = dgv_PlayList.Rows[i].Cells[3];

                    if (skipChapterCell.Value != null && skipChapterCell.Value.ToString() != "")
                    {
                        var formattedValue = System.Text.RegularExpressions.Regex.Replace(skipChapterCell.Value.ToString(), @"[^0-9,\s]*", "");
                        var numbers = formattedValue.Trim().Replace(" ", ",").Split(',');
                        var sortedNumbers = numbers.Distinct().Except(new[] { "" }).Select(int.Parse).OrderBy(x => x).ToList();

                        if (currentPlayItem != null && i == currentPlayIndex)
                        {
                            if (sortedNumbers.Any(num => num >= PlayerControl.Chapters.Count))
                            {
                                var toolTip = new ToolTip();
                                var cellDisplayRect = dgv_PlayList.GetCellDisplayRectangle(skipChapterCell.ColumnIndex,
                                    skipChapterCell.RowIndex, false);
                                toolTip.Show("Only numbers < " + PlayerControl.Chapters.Count + " are allowed",
                                    dgv_PlayList,
                                    cellDisplayRect.X + skipChapterCell.Size.Width / 2,
                                    cellDisplayRect.Y + skipChapterCell.Size.Height / 2,
                                    2000);
                                sortedNumbers.RemoveAll(num => num >= PlayerControl.Chapters.Count);
                            }
                            if (PlayerControl.Chapters.Count == 0)
                            {
                                sortedNumbers.Clear();
                            }
                        }

                        formattedValue = String.Join(",", sortedNumbers);
                        skipChapterCell.Value = formattedValue;
                    }

                    if (endChapterCell.Value != null && endChapterCell.Value.ToString() != "")
                    {
                        var value = new String(endChapterCell.Value.ToString().Where(Char.IsDigit).ToArray());

                        if (currentPlayItem != null && i == currentPlayIndex)
                        {
                            if (value.Length > 0 && int.Parse(value) > PlayerControl.Chapters.Count)
                            {
                                var toolTip = new ToolTip();
                                var cellDisplayRect = dgv_PlayList.GetCellDisplayRectangle(endChapterCell.ColumnIndex,
                                    endChapterCell.RowIndex, false);
                                toolTip.Show("Only numbers <= " + PlayerControl.Chapters.Count + " are allowed",
                                    dgv_PlayList,
                                    cellDisplayRect.X + endChapterCell.Size.Width / 2,
                                    cellDisplayRect.Y + endChapterCell.Size.Height / 2,
                                    2000);

                                value = PlayerControl.Chapters.Count.ToString();
                            }
                            if (PlayerControl.Chapters.Count == 0)
                            {
                                value = "";
                            }
                        }

                        endChapterCell.Value = value;
                    }
                }

                UpdatePlaylist();
            }
            catch (Exception ex)
            {
                PlayerControl.HandleException(ex);
            }
        }

        private void UpdatePlaylist()
        {
            try
            {
                for (int i = 0; i < playList.Count; i++)
                {
                    var skipChapters = new List<int>();
                    int endChapter = -1;

                    var skipChapterCell = dgv_PlayList.Rows[i].Cells[2];
                    var endChapterCell = dgv_PlayList.Rows[i].Cells[3];

                    if (skipChapterCell.Value != null && skipChapterCell.Value.ToString() != "")
                    {
                        skipChapters = skipChapterCell.Value.ToString().Split(',').Select(int.Parse).ToList();
                        playList.ElementAt(i).HasChapter = true;
                    }

                    if (endChapterCell.Value != null && endChapterCell.Value.ToString() != "")
                    {
                        endChapter = int.Parse(endChapterCell.Value.ToString());
                    }

                    playList.ElementAt(i).SkipChapters = skipChapters;
                    playList.ElementAt(i).EndChapter = endChapter;
                }
            }
            catch (Exception ex)
            {
                PlayerControl.HandleException(ex);
            }
        }

        private void ParseWithoutChapters(string line)
        {
            string title = "";
            bool isActive = false;

            if (line.StartsWith(ActiveIndicator))
            {
                title = line.Substring(ActiveIndicator.Length).Trim();
                isActive = true;
            }
            else if (line.StartsWith(InactiveIndicator))
            {
                title = line.Substring(InactiveIndicator.Length).Trim();
            }
            else
            {
                throw new FileLoadException();
            }
            playList.Add(new PlaylistItem(title, isActive));
        }

        private void ParseWithChapters(string line)
        {
            var splitLine = line.Split('|');
            string title = "";
            bool isActive = false;
            var skipChapters = new List<int>();

            if (splitLine[0].StartsWith(ActiveIndicator))
            {
                title = splitLine[0].Substring(ActiveIndicator.Length).Trim();
                isActive = true;
            }
            else if (line.StartsWith(InactiveIndicator))
            {
                title = splitLine[0].Substring(InactiveIndicator.Length).Trim();
            }
            else
            {
                throw new FileLoadException();
            }

            if (splitLine[1].Length > 0)
            {
                splitLine[1] = splitLine[1].Substring(splitLine[1].IndexOf(':') + 1).Trim();
                skipChapters = new List<int>(splitLine[1].Split(',').Select(int.Parse));
            }

            var endChapter = int.Parse(splitLine[2].Substring(splitLine[2].IndexOf(':') + 1).Trim());
            playList.Add(new PlaylistItem(title, skipChapters, endChapter, isActive));
        }

        private void OpenMedia()
        {
            if (currentPlayIndex < 0 || currentPlayIndex >= playList.Count) return;

            bool playerWasFullScreen = PlayerControl.InFullScreenMode;
            ResetActive();

            try
            {
                var item = playList[currentPlayIndex];
                SetPlayStyling();
                dgv_PlayList.CurrentCell = dgv_PlayList.Rows[currentPlayIndex].Cells[1];

                if (File.Exists(item.FilePath))
                {
                    PlayerControl.OpenMedia(item.FilePath);
                }
                else
                {
                    PlayNext();
                }

                if (playerWasFullScreen)
                {
                    PlayerControl.GoFullScreen();
                }

                item.Active = true;
                currentPlayItem = item;
                previousChapterPosition = 0;

                Text = PlayerControl.PlayerState + " ─ " + currentPlayItem.FilePath;
                ParseChapterInput();
            }
            catch (Exception ex)
            {
                PlayerControl.HandleException(ex);
                PlayNext();
            }

            dgv_PlayList.Invalidate();
        }

        private void SortPlayList(bool ascending)
        {
            if (ascending)
            {
                playList.Sort();
            }
            else
            {
                playList.Sort();
                playList.Reverse();
            }

            PopulatePlaylist();
        }

        private void SetPlayStyling()
        {
            foreach (DataGridViewRow r in dgv_PlayList.Rows)
            {
                r.DefaultCellStyle.ForeColor = Color.Black;
                r.Selected = false;
            }

            dgv_PlayList.Rows[currentPlayIndex].DefaultCellStyle.ForeColor = Color.White;
            dgv_PlayList.Rows[currentPlayIndex].Selected = true;
        }

        private void SelectChapter(int chapterNum)
        {
            if (PlayerControl.PlayerState == PlayerState.Closed) return;

            var chapters = GetChapters();

            if (chapters.ElementAt(chapterNum) == null) return;
            PlayerControl.SeekMedia(chapters.ElementAt(chapterNum).Position);
            PlayerControl.ShowOsdText(chapters.ElementAt(chapterNum).Name);
        }

        private int GetChapterIndexByPosition(long position)
        {
            int currentChapterIndex = 0;

            foreach (var c in GetChapters().Where(c => c != null))
            {
                currentChapterIndex++;
                if (c.Position != position) continue;
                return currentChapterIndex;
            }

            return 0;
        }

        private IEnumerable<Chapter> GetChapters()
        {
            return PlayerControl.Chapters.OrderBy(chapter => chapter.Position);
        }

        private void ResetActive()
        {
            foreach (var item in playList)
            {
                item.Active = false;
            }
        }

        private void AddFilesToPlaylist()
        {
            if (openFileDialog.ShowDialog(this) != DialogResult.OK)
                return;

            var fileNames = openFileDialog.FileNames;
            AddFiles(fileNames);
        }

        private void RemoveSelectedItems()
        {
            var rowIndexes = new List<int>();

            try
            {
                if (playList.Count <= 0) return;

                rowIndexes.AddRange(from DataGridViewRow r in dgv_PlayList.SelectedRows select r.Index);

                foreach (int index in rowIndexes.OrderByDescending(v => v).Where(index => index != currentPlayIndex))
                {
                    playList.RemoveAt(index);
                }

                PopulatePlaylist();
            }
            catch (Exception ex)
            {
                PlayerControl.HandleException(ex);
            }
        }

        private void ButtonAddClick(object sender, EventArgs e)
        {
            AddFilesToPlaylist();
        }

        private void ButtonDelClick(object sender, EventArgs e)
        {
            RemoveSelectedItems();
        }

        private void ButtonNewClick(object sender, EventArgs e)
        {
            NewPlaylist();
        }

        private void ButtonOpenClick(object sender, EventArgs e)
        {
            OpenPlaylist();
        }

        private void ButtonSaveClick(object sender, EventArgs e)
        {
            SavePlaylist();
        }

        private void ButtonLeftClick(object sender, EventArgs e)
        {
            PlayPrevious();
        }

        private void ButtonRightClick(object sender, EventArgs e)
        {
            PlayNext();
        }

        private void ButtonSortAscendingClick(object sender, EventArgs e)
        {
            SortPlayList(true);
        }

        private void ButtonSortDescendingClick(object sender, EventArgs e)
        {
            SortPlayList(false);
        }

        private void FormKeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyData == Keys.Escape)
            {
                Hide();
            }
        }

        private void dgv_PlayList_DragEnter(object sender, DragEventArgs e)
        {
            e.Effect = DragDropEffects.Copy;
        }

        private void dgv_PlayList_DragDrop(object sender, DragEventArgs e)
        {
            var files = (string[]) e.Data.GetData(DataFormats.FileDrop);
            if (files.Length > 1)
            {
                // Add multiple files to playlist
                AddFiles(files);
            }
            else
            {
                var filename = files[0];
                if (Playlist.IsPlaylistFile(filename))
                {
                    // Playlist file
                    OpenPlaylist(filename);
                }
            }
        }
    }

    public class PlaylistItem : IComparable<PlaylistItem>
    {
        public string FilePath { get; set; }
        public bool Active { get; set; }
        public bool HasChapter { get; set; }
        public List<int> SkipChapters { get; set; }
        public int EndChapter { get; set; }

        public PlaylistItem(string filePath, bool isActive)
        {
            if (string.IsNullOrWhiteSpace(filePath))
            {
                throw new ArgumentNullException("filePath");
            }

            FilePath = filePath;
            Active = isActive;
        }

        public PlaylistItem(string filePath, List<int> skipChapter, int endChapter, bool isActive)
        {
            if (string.IsNullOrWhiteSpace(filePath))
            {
                throw new ArgumentNullException("filePath");
            }

            FilePath = filePath;
            Active = isActive;
            SkipChapters = skipChapter;
            EndChapter = endChapter;
            HasChapter = true;
        }

        public override string ToString()
        {
            if (HasChapter)
            {
                return Path.GetFileName(FilePath) + " | SkipChapter: " + String.Join(",", SkipChapters) + " | EndChapter: " + EndChapter ?? "???";
            }

            return Path.GetFileName(FilePath) ?? "???";
        }

        public int CompareTo(PlaylistItem other)
        {
            return String.CompareOrdinal(FilePath, other.FilePath);
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
