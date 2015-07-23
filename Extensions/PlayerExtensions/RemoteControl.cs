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
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Timers;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;
using Timer = System.Timers.Timer;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class AcmPlug : PlayerExtension<RemoteControlSettings, RemoteControlConfig>
    {
        #region Properties

        public Dictionary<Guid, Socket> Clients { get; private set; }

        #endregion

        public AcmPlug()
        {
            Clients = new Dictionary<Guid, Socket>();
        }

        public override ExtensionUiDescriptor Descriptor
         {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("C7FC10786471409DA2F134FF8903D6DA"),
                    Name = "Remote Control",
                    Description = "Remote Control extension to allow control of MPDN over the network.",
                    Copyright = "Copyright DeadlyEmbrace © 2015. All rights reserved."
                };
            }
        }
        protected override string ConfigFileName
        {
            get { return "Example.RemoteSettings"; }
        }

        public override IList<Verb> Verbs
        {

            get
            {
                return new[]
                {
                    new Verb(Category.Help, string.Empty, "Connected Clients", "Ctrl+Shift+R",
                        "Show Remote Client connections", ConnectedClientMenuClick)
                };


            }
        }

        public override void Destroy()
        {
            base.Destroy();
            if (Settings.IsActive)
                ShutdownServer();
        }

        public override void Initialize()
        {
            base.Initialize();
            Settings.PropertyChanged += SettingsPropertyChanged;
            if (Settings.IsActive)
                SetupServer();
        }

        private void ShutdownServer()
        {
            Unsubscribe();
            _locationTimer.Stop();
            _locationTimer = null;
            foreach (var writer in _writers)
            {
                try
                {
                    writer.Value.WriteLine("Closing|Close");
                    writer.Value.Flush();
                }
                catch
                {
                }
            }
            if (_serverSocket != null)
                _serverSocket.Close();
        }

        private void SetupServer()
        {
            Subscribe();
            _locationTimer = new Timer(100);
            _locationTimer.Elapsed += _locationTimer_Elapsed;
            _clientManager = new RemoteClients(this);
            var playlist = Extension.PlayerExtensions.FirstOrDefault(t => t.Descriptor.Guid == _playlistGuid);
            if (playlist != null)
            {
                _playlistInstance = playlist as Playlist.Playlist;
                if (_playlistInstance != null)
                {
                    _playlistInstance.GetPlaylistForm.VisibleChanged += GetPlaylistFormVisibleChanged;
                    _playlistInstance.GetPlaylistForm.PlaylistChanged += GetPlaylistFormPlaylistChanged;
                }
            }
            Task.Factory.StartNew(Server);
        }

        private void GetPlaylistFormPlaylistChanged(object sender, EventArgs e)
        {
            GetPlaylist(Guid.Empty, true);
        }

        private void GetPlaylistFormVisibleChanged(object sender, EventArgs e)
        {
            PushToAllListeners("PlaylistShow|" + _playlistInstance.GetPlaylistForm.Visible);
        }

        private void Subscribe()
        {
            Player.Playback.Completed += PlayerControlPlaybackCompleted;
            Player.StateChanged += m_PlayerControl_PlayerStateChanged;
            Player.FullScreenMode.Entering += PlayerControlEnteringFullScreenMode;
            Player.FullScreenMode.Exiting += PlayerControlExitingFullScreenMode;
            Player.VolumeChanged += PlayerControlVolumeChanged;
            Media.SubtitleTrackChanged += PlayerControlSubtitleTrackChanged;
            Media.AudioTrackChanged += PlayerControlAudioTrackChanged;
        }

        private void SettingsPropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == "IsActive" && Settings.IsActive)
            {
                SetupServer();
            }
            if (e.PropertyName == "IsActive" && !Settings.IsActive)
            {
                ShutdownServer();
            }
        }

        private void Unsubscribe()
        {
            Player.Playback.Completed -= PlayerControlPlaybackCompleted;
            Player.StateChanged -= m_PlayerControl_PlayerStateChanged;
            Player.FullScreenMode.Entering -= PlayerControlEnteringFullScreenMode;
            Player.FullScreenMode.Exiting -= PlayerControlExitingFullScreenMode;
            Player.VolumeChanged -= PlayerControlVolumeChanged;
            Media.SubtitleTrackChanged -= PlayerControlSubtitleTrackChanged;
            Media.AudioTrackChanged -= PlayerControlAudioTrackChanged;
        }

        private void PlayerControlAudioTrackChanged(object sender, EventArgs e)
        {
            PushToAllListeners("AudioChanged|" + Media.AudioTrack.Description);
        }

        private void PlayerControlSubtitleTrackChanged(object sender, EventArgs e)
        {
            PushToAllListeners("SubChanged|" + Media.SubtitleTrack.Description);
        }

        private void PlayerControlVolumeChanged(object sender, EventArgs e)
        {
            PushToAllListeners("Volume|" + Player.Volume);
            PushToAllListeners("Mute|" + Player.Mute);
        }

        private void PlayerControlExitingFullScreenMode(object sender, EventArgs e)
        {
            PushToAllListeners("Fullscreen|False");
        }

        private void PlayerControlEnteringFullScreenMode(object sender, EventArgs e)
        {
            PushToAllListeners("Fullscreen|True");
        }

        private void m_PlayerControl_PlayerStateChanged(object sender, PlayerStateEventArgs e)
        {
            switch (e.NewState)
            {
                case PlayerState.Playing:
                    _locationTimer.Start();
                    PushToAllListeners(GetAllChapters());
                    PushToAllListeners(GetAllSubtitleTracks());
                    PushToAllListeners(GetAllAudioTracks());
                    break;
                case PlayerState.Stopped:
                    _locationTimer.Stop();
                    break;
                case PlayerState.Paused:
                    _locationTimer.Stop();
                    break;
            }

            PushToAllListeners(e.NewState + "|" + Media.FilePath);
        }

        private string GetAllAudioTracks()
        {
            if (Player.State == PlayerState.Playing || Player.State == PlayerState.Paused)
            {
                MediaTrack activeTrack = null;
                if (Media.AudioTrack != null)
                    activeTrack = Media.AudioTrack;
                var audioTracks = Media.AudioTracks;
                int counter = 1;
                StringBuilder audioStringBuilder = new StringBuilder();
                foreach (var track in audioTracks)
                {
                    if (counter > 1)
                        audioStringBuilder.Append("]]");
                    audioStringBuilder.Append(counter + ">>" + track.Description + ">>" + track.Type);
                    if (activeTrack != null && track.Description == activeTrack.Description)
                        audioStringBuilder.Append(">>True");
                    else
                        audioStringBuilder.Append(">>False");
                    counter++;
                }
                return "AudioTracks|" + audioStringBuilder;
            }
            return String.Empty;
        }

        private string GetAllSubtitleTracks()
        {
            if (Player.State == PlayerState.Playing || Player.State == PlayerState.Paused)
            {
                MediaTrack activeSub = null;
                if (Media.SubtitleTrack != null)
                    activeSub = Media.SubtitleTrack;
                var subtitles = Media.SubtitleTracks;
                int counter = 1;
                StringBuilder subSb = new StringBuilder();
                foreach (var sub in subtitles)
                {
                    if (counter > 1)
                        subSb.Append("]]");
                    subSb.Append(counter + ">>" + sub.Description + ">>" + sub.Type);
                    if (activeSub != null && sub.Description == activeSub.Description)
                        subSb.Append(">>True");
                    else
                        subSb.Append(">>False");
                    counter++;
                }
                return "Subtitles|" + subSb;
            }
            return String.Empty;
        }

        private string GetAllChapters()
        {
            if (Player.State == PlayerState.Playing || Player.State == PlayerState.Paused)
            {
                var chapters = Media.Chapters;
                int counter = 1;
                StringBuilder chapterSb = new StringBuilder();
                foreach (var chapter in chapters)
                {
                    if (counter > 1)
                        chapterSb.Append("]]");
                    chapterSb.Append(counter + ">>" + chapter.Name + ">>" + chapter.Position);
                    counter++;
                }
                return "Chapters|" + chapterSb;
            }
            return String.Empty;
        }

        private void PlayerControlPlaybackCompleted(object sender, EventArgs e)
        {
            PushToAllListeners("Finished|" + Media.FilePath);
        }

        private void ConnectedClientMenuClick()
        {
            if (_clientManager != null)
            {
                _clientManager.ShowDialog(Gui.VideoBox);
            }
            else
            {
                MessageBox.Show(Gui.VideoBox, "The remote control is not activated.");
            }
        }

        private void Server()
        {
            IPEndPoint localEndpoint = new IPEndPoint(IPAddress.Any, Settings.ConnectionPort);
            _serverSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            _serverSocket.Bind(localEndpoint);
            _serverSocket.Listen(10);
            while (true)
            {
                try
                {
                    var clientSocket = _serverSocket.Accept();
                    Task.Factory.StartNew(() => ClientHandler(clientSocket));
                }
                catch (Exception)
                {
                    break;
                }
            }
        }

        private void ClientHandler(Socket client)
        {
            Guid clientGuid = Guid.NewGuid();
            Clients.Add(clientGuid, client);

            NetworkStream nStream = new NetworkStream(client);
            StreamReader reader = new StreamReader(nStream);
            StreamWriter writer = new StreamWriter(nStream);
            _writers.Add(clientGuid, writer);
            var clientGUID = reader.ReadLine();
            if (clientGUID == null) throw new ArgumentNullException(nameof(clientGUID));
            if (!_authHandler.IsGUIDAuthed(clientGUID) && Settings.ValidateClients)
            {
                ClientAuth(clientGUID, clientGuid);
            }
            else
            {
                DisplayTextMessage("Remote Connected");
                WriteToSpecificClient("Connected|Authorized", clientGuid.ToString());
                WriteToSpecificClient("ClientGUID|" + clientGuid, clientGuid.ToString());
                if (!_authHandler.IsGUIDAuthed(clientGUID))
                    _authHandler.AddAuthedClient(clientGUID);
                if (_clientManager.Visible)
                    _clientManager.ForceUpdate();
            }
            while (true)
            {
                try
                {
                    var data = reader.ReadLine();
                    if (data == "Exit")
                    {
                        HandleData(data);
                        client.Close();
                        break;
                    }
                    HandleData(data);
                }
                catch
                {
                    break;
                }
            }
        }

        private void ClientAuth(string msgValue, Guid clientGuid)
        {
            WriteToSpecificClient("AuthCode|" + msgValue, clientGuid.ToString());
            if (
                MessageBox.Show(Gui.VideoBox, "Allow Remote Connection for " + msgValue, "Remote Authentication",
                    MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
            {
                DisplayTextMessage("Remote Connected");
                WriteToSpecificClient("Connected|Authorized", clientGuid.ToString());
                _authHandler.AddAuthedClient(msgValue);
                if (_clientManager.Visible)
                    _clientManager.ForceUpdate();
            }
            else
            {
                DisconnectClient("Unauthorized", clientGuid);
            }
        }

        private void WriteToSpecificClient(string msg, string clientGuid)
        {
            Guid pushGuid;
            Guid.TryParse(clientGuid, out pushGuid);

            if (_writers.ContainsKey(pushGuid))
            {
                _writers[pushGuid].WriteLine(msg);
                _writers[pushGuid].Flush();
            }
        }

        private void DisconnectClient(string exitMessage, Guid clientGuid)
        {
            WriteToSpecificClient("Exit|" + exitMessage, clientGuid.ToString());

            Clients[clientGuid].Disconnect(true);
            RemoveWriter(clientGuid.ToString());
        }

        private void HandleData(string data)
        {
            var command = data.Split('|');
            switch (command[0])
            {
                case "Exit":
                    DisplayTextMessage("Remote Disconnected");
                    RemoveWriter(command[1]);
                    break;
                case "Open":
                    GuiThread.DoAsync(() => OpenMedia(command[1]));
                    break;
                case "Pause":
                    GuiThread.DoAsync(() => PauseMedia(command[1]));
                    break;
                case "Play":
                    GuiThread.DoAsync(() => PlayMedia(command[1]));
                    break;
                case "Stop":
                    GuiThread.DoAsync(() => StopMedia(command[1]));
                    break;
                case "Seek":
                    GuiThread.DoAsync(() => SeekMedia(command[1]));
                    break;
                case "GetDuration":
                    GuiThread.DoAsync(() => GetFullDuration(command[1]));
                    break;
                case "GetCurrentState":
                    GuiThread.DoAsync(() => GetCurrentState(command[1]));
                    break;
                case "FullScreen":
                    GuiThread.DoAsync(() => FullScreen(command[1]));
                    break;
                case "MoveWindow":
                    GuiThread.DoAsync(() => MoveWindow(command[1]));
                    break;
                case "WriteToScreen":
                    DisplayTextMessage(command[1]);
                    break;
                case "Mute":
                    bool mute = false;
                    Boolean.TryParse(command[1], out mute);
                    GuiThread.DoAsync(() => Mute(mute));
                    break;
                case "Volume":
                    int vol = 0;
                    int.TryParse(command[1], out vol);
                    GuiThread.DoAsync(() => SetVolume(vol));
                    break;
                case "ActiveSubTrack":
                    GuiThread.DoAsync(() => SetSubtitle(command[1]));
                    break;
                case "ActiveAudioTrack":
                    GuiThread.DoAsync(() => SetAudioTrack(command[1]));
                    break;
                case "AddFilesToPlaylist":
                    AddFilesToPlaylist(command[1]);
                    break;
                case "InsertFileInPlaylist":
                    GuiThread.DoAsync(() => InsertIntoPlaylist(command[1], command[2]));
                    break;
                case "ClearPlaylist":
                    GuiThread.DoAsync(ClearPlaylist);
                    break;
                case "FocusPlayer":
                    GuiThread.DoAsync(FocusMpdn);
                    break;
                case "PlayNext":
                    GuiThread.DoAsync(PlaylistPlayNext);
                    break;
                case "PlayPrevious":
                    GuiThread.DoAsync(PlaylistPlayPrevious);
                    break;
                case "ShowPlaylist":
                    GuiThread.DoAsync(ShowPlaylist);
                    break;
                case "HidePlaylist":
                    GuiThread.DoAsync(HidePlaylist);
                    break;
                case "GetPlaylist":
                    GuiThread.DoAsync(() => GetPlaylist(command[1]));
                    break;
                case "PlaySelectedFile":
                    GuiThread.DoAsync(() => PlaySelectedFile(command[1]));
                    break;
                case "RemoveFile":
                    GuiThread.DoAsync(() => RemoveFromPlaylist(command[1]));
                    break;
            }
        }

        private void InsertIntoPlaylist(string fileIndex, string filePath)
        {
            int index;
            int.TryParse(fileIndex, out index);
            _playlistInstance.GetPlaylistForm.InsertFile(index, filePath);
        }

        private void RemoveFromPlaylist(string fileIndex)
        {
            int index;
            int.TryParse(fileIndex, out index);
            _playlistInstance.GetPlaylistForm.RemoveFile(index);
        }

        private void PlaySelectedFile(string fileIndex)
        {
            int myIndex;
            int.TryParse(fileIndex, out myIndex);
            _playlistInstance.GetPlaylistForm.SetPlaylistIndex(myIndex);
        }

        private void GetPlaylist(object guid, bool notify = false)
        {
            int counter = 0;
            StringBuilder sb = new StringBuilder();
            var fullPlaylist = _playlistInstance.GetPlaylistForm.Playlist;
            foreach (var item in fullPlaylist)
            {
                counter++;
                if (counter > 1)
                    sb.Append(">>");
                sb.Append(item.FilePath + "]]" + item.Active);
            }
            if (!notify)
            {
                WriteToSpecificClient("PlaylistContent|" + sb, guid.ToString());
            }
            else
            {
                PushToAllListeners("PlaylistContent|" + sb);
            }
        }

        private void HidePlaylist()
        {
            _playlistInstance.GetPlaylistForm.Hide();
        }

        private void ShowPlaylist()
        {
            _playlistInstance.GetPlaylistForm.Show();
        }

        private void PlaylistPlayNext()
        {
            _playlistInstance.GetPlaylistForm.PlayNext();
        }

        private void PlaylistPlayPrevious()
        {
            _playlistInstance.GetPlaylistForm.PlayPrevious();
        }

        public void FocusMpdn()
        {
            Player.ActiveForm.Focus();
        }

        private void ClearPlaylist()
        {
            _playlistInstance.GetPlaylistForm.ClearPlaylist();
        }

        private void AddFilesToPlaylist(string files)
        {
            var filesToAdd = new List<string>();
            var filePaths = Regex.Split(files, ">>");
            if (filePaths.Any())
            {
                filesToAdd.AddRange(filePaths.Where(File.Exists));
            }
            if (filesToAdd.Any())
            {
                GuiThread.DoAsync(() => _playlistInstance.GetPlaylistForm.AddFiles(filesToAdd.ToArray()));
            }
        }

        private void RemoveWriter(string guid)
        {
            var callerGuid = Guid.Parse(guid);
            _writers.Remove(callerGuid);
            Clients.Remove(callerGuid);
            _clientManager.ForceUpdate();
        }

        private void OpenMedia(object file)
        {
            Media.Open(file.ToString());
        }

        private void PauseMedia(object showOsd)
        {
            var dispOsd = false;
            Boolean.TryParse(showOsd.ToString(), out dispOsd);
            Media.Pause(dispOsd);
        }

        private void PlayMedia(object showOsd)
        {
            bool dispOsd = false;
            Boolean.TryParse(showOsd.ToString(), out dispOsd);
            if (!String.IsNullOrEmpty(Media.FilePath))
            {
                Media.Play(dispOsd);
            }
            else
            {
                _playlistInstance.GetPlaylistForm.PlayActive();
            }
        }

        private void StopMedia(object blank)
        {
            Media.Stop();
        }

        private void SeekMedia(object seekLocation)
        {
            double location = -1;
            double.TryParse(seekLocation.ToString(), out location);
            if (location != -1)
            {
                Media.Seek((long) location);
            }
        }

        private void SetVolume(int level)
        {
            Player.Volume = level;
        }

        private void SetSubtitle(string subDescription)
        {
            var selTrack = Media.SubtitleTracks.FirstOrDefault(t => t.Description == subDescription);
            if (selTrack != null)
                Media.SelectSubtitleTrack(selTrack);
        }

        private void SetAudioTrack(string audioDescription)
        {
            var selTrack = Media.AudioTracks.FirstOrDefault(t => t.Description == audioDescription);
            if (selTrack != null)
                Media.SelectAudioTrack(selTrack);
        }

        private void Mute(bool silence)
        {
            Player.Mute = silence;
            PushToAllListeners("Mute|" + silence);
        }

        private void GetFullDuration(object guid)
        {
            WriteToSpecificClient("FullLength|" + Media.Duration, guid.ToString());
        }

        private void GetCurrentState(object guid)
        {
            WriteToSpecificClient(GetAllChapters(), guid.ToString());
            WriteToSpecificClient(Player.State + "|" + Media.FilePath, guid.ToString());
            WriteToSpecificClient("Fullscreen|" + Player.FullScreenMode.Active, guid.ToString());
            WriteToSpecificClient("Mute|" + Player.Mute, guid.ToString());
            WriteToSpecificClient("Volume|" + Player.Volume, guid.ToString());
            GetPlaylist(guid);
            if (Player.State == PlayerState.Playing || Player.State == PlayerState.Paused)
            {
                WriteToSpecificClient("FullLength|" + Media.Duration, guid.ToString());
                WriteToSpecificClient("Position|" + Media.Position, guid.ToString());
            }
            if (_playlistInstance != null)
            {
                PushToAllListeners("PlaylistShow|" + _playlistInstance.GetPlaylistForm.Visible);
            }
            WriteToSpecificClient(GetAllSubtitleTracks(), guid.ToString());
            WriteToSpecificClient(GetAllAudioTracks(), guid.ToString());
        }

        private void FullScreen(object fullScreen)
        {
            bool goFullscreen = false;
            Boolean.TryParse(fullScreen.ToString(), out goFullscreen);
            Player.FullScreenMode.Active = goFullscreen;
        }

        private void MoveWindow(string msg)
        {
            var args = msg.Split(new[] {">>"}, StringSplitOptions.None);

            int left, top, width, height;
            if (int.TryParse(args[0], out left) &&
                int.TryParse(args[1], out top) &&
                int.TryParse(args[2], out width) &&
                int.TryParse(args[3], out height))
            {
                Player.ActiveForm.Left = left;
                Player.ActiveForm.Top = top;
                Player.ActiveForm.Width = width;
                Player.ActiveForm.Height = height;

                switch (args[4])
                {
                    case "Normal":
                        Player.ActiveForm.WindowState = FormWindowState.Normal;
                        break;
                    case "Maximized":
                        Player.ActiveForm.WindowState = FormWindowState.Maximized;
                        break;
                    case "Minimized":
                        Player.ActiveForm.WindowState = FormWindowState.Minimized;
                        break;
                }
            }
        }

        private void PushToAllListeners(string msg)
        {
            foreach (var writer in _writers)
            {
                try
                {
                    writer.Value.WriteLine(msg);
                    writer.Value.Flush();
                }
                catch
                {
                }
            }
        }

        private void DisplayTextMessage(object msg)
        {
            GuiThread.DoAsync(() => Player.OsdText.Show(msg.ToString()));
        }

        public void DisconnectClient(string guid)
        {
            Guid clientGuid;
            Guid.TryParse(guid, out clientGuid);
            DisconnectClient("Disconnected by User", clientGuid);
        }

        private void _locationTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            try
            {
                PushToAllListeners("Postion|" + Media.Position);
            }
            catch (Exception)
            {
            }
        }

        #region Variables

        private Socket _serverSocket;
        private readonly Dictionary<Guid, StreamWriter> _writers = new Dictionary<Guid, StreamWriter>();
        private readonly RemoteControl_AuthHandler _authHandler = new RemoteControl_AuthHandler();
        private RemoteClients _clientManager;
        private Timer _locationTimer;
        private readonly Guid _playlistGuid = Guid.Parse("A1997E34-D67B-43BB-8FE6-55A71AE7184B");
        private Playlist.Playlist _playlistInstance;

        #endregion
    }

    public class RemoteControlSettings : INotifyPropertyChanged
    {
        #region Public Methods

        public RemoteControlSettings()
        {
            ConnectionPort = 6545;
            ValidateClients = true;
        }

        #endregion

        #region Events

        public event PropertyChangedEventHandler PropertyChanged;

        #endregion

        #region Private Methods

        public void OnPropertyChanged(PropertyChangedEventArgs e)
        {
            PropertyChanged.Handle(h => h(this, e));
        }

        #endregion

        #region Variables

        private int _connectionPort;
        private bool _validateClients;
        private bool _isActive;

        #endregion

        #region Properties

        public int ConnectionPort
        {
            get { return _connectionPort; }
            set
            {
                _connectionPort = value;
                OnPropertyChanged(new PropertyChangedEventArgs("ConnectionPort"));
            }
        }

        public bool ValidateClients
        {
            get { return _validateClients; }
            set
            {
                _validateClients = value;
                OnPropertyChanged(new PropertyChangedEventArgs("ValidateClients"));
            }
        }

        public bool IsActive
        {
            get { return _isActive; }
            set
            {
                _isActive = value;
                OnPropertyChanged(new PropertyChangedEventArgs("IsActive"));
            }
        }

        #endregion
    }
}