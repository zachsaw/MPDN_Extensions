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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.PlayerExtensions.Playlist;
using Mpdn.Extensions.PlayerExtensions.DataContracts;
using YAXLib;
using Timer = System.Windows.Forms.Timer;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class AcmPlug : PlayerExtension<RemoteControlSettings, RemoteControlConfig>
    {
        private const int SERVER_VERSION = 3;

        #region Properties

        public ConcurrentDictionary<Guid, Socket> Clients { get; private set; }

        #endregion

        public AcmPlug()
        {
            Clients = new ConcurrentDictionary<Guid, Socket>();
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
                    Copyright = ""
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
            m_LocationTimer.Stop();
            m_LocationTimer = null;
            foreach (var writer in m_Writers)
            {
                try
                {
                    var w = writer.Value;
                    lock (w)
                    {
                        w.WriteLine("Closing|Close");
                        w.Flush();
                    }
                }
                catch
                {
                }
            }
            if (m_ServerSocket != null)
                m_ServerSocket.Close();
        }

        private void SetupServer()
        {
            if (m_LocationTimer != null)
            {
                ShutdownServer();
            }

            Subscribe();
            m_LocationTimer = new Timer {Interval = 100};
            m_LocationTimer.Tick += _locationTimer_Elapsed;
            m_ClientManager = new RemoteClients(this);
            Task.Factory.StartNew(Server);
        }

        private void GetPlaylistFormPlaylistChanged(object sender, EventArgs e)
        {
            GetPlaylist(null);
        }

        private void GetPlaylistFormVisibleChanged(object sender, EventArgs e)
        {
            PushToAllListeners("PlaylistShow|" + PlaylistForm.Visible.ToString(CultureInfo.InvariantCulture));
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
            Media.VideoTrackChanged += MediaOnVideoTrackChanged;
            PlaylistForm.VisibleChanged += GetPlaylistFormVisibleChanged;
            PlaylistForm.PlaylistChanged += GetPlaylistFormPlaylistChanged;
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
            PlaylistForm.VisibleChanged -= GetPlaylistFormVisibleChanged;
            PlaylistForm.PlaylistChanged -= GetPlaylistFormPlaylistChanged;
            Player.Playback.Completed -= PlayerControlPlaybackCompleted;
            Player.StateChanged -= m_PlayerControl_PlayerStateChanged;
            Player.FullScreenMode.Entering -= PlayerControlEnteringFullScreenMode;
            Player.FullScreenMode.Exiting -= PlayerControlExitingFullScreenMode;
            Player.VolumeChanged -= PlayerControlVolumeChanged;
            Media.SubtitleTrackChanged -= PlayerControlSubtitleTrackChanged;
            Media.AudioTrackChanged -= PlayerControlAudioTrackChanged;
            Media.VideoTrackChanged -= MediaOnVideoTrackChanged;
        }

        private void MediaOnVideoTrackChanged(object sender, EventArgs eventArgs)
        {
            PushToAllListeners("VideoChanged|" + Media.VideoTrack.Description);
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
            PushToAllListeners("Volume|" + Player.Volume.ToString(CultureInfo.InvariantCulture));
            PushToAllListeners("Mute|" + Player.Mute.ToString(CultureInfo.InvariantCulture));
        }

        private void PlayerControlExitingFullScreenMode(object sender, EventArgs e)
        {
            PushToAllListeners("Fullscreen|" + false.ToString(CultureInfo.InvariantCulture));
        }

        private void PlayerControlEnteringFullScreenMode(object sender, EventArgs e)
        {
            PushToAllListeners("Fullscreen|" + true.ToString(CultureInfo.InvariantCulture));
        }

        private void m_PlayerControl_PlayerStateChanged(object sender, PlayerStateEventArgs e)
        {
            switch (e.NewState)
            {
                case PlayerState.Playing:
                    m_LocationTimer.Start();
                    PushToAllListeners(GetAllChapters());
                    PushToAllListeners(GetAllSubtitleTracks());
                    PushToAllListeners(GetAllAudioTracks());
                    PushToAllListeners(GetAllVideoTracks());
                    break;
                case PlayerState.Stopped:
                case PlayerState.Closed:
                    m_LocationTimer.Stop();
                    break;
                case PlayerState.Paused:
                    m_LocationTimer.Start();
                    break;
            }

            PushToAllListeners(e.NewState + "|" + Media.FilePath);
            if (e.OldState == PlayerState.Closed)
            {
                GetPlaylistActiveIndex();
            }
        }

        private void GetPlaylistActiveIndex()
        {
            PushToAllListeners("PlaylistActiveIndex|" +
                               (PlaylistForm.Playlist ?? new List<PlaylistItem>())
                                   .TakeWhile(item => !item.Active)
                                   .Count().ToString(CultureInfo.InvariantCulture));
        }

        private static string EscapeDelimiters(string input)
        {
            // Simple method to escape delimiters (can't reconstruct it back)
            var result = input;
            do
            {
                input = result;
                result = input.Replace("]]", " ] ]").Replace(">>", " > >");
            } while (result != input);
            return result;
        }

        private string GetAllVideoTracks()
        {
            if (Player.State != PlayerState.Playing && Player.State != PlayerState.Paused) return string.Empty;
            MediaTrack activeTrack = null;
            if (Media.VideoTrack != null)
                activeTrack = Media.VideoTrack;
            var videoTracks = Media.VideoTracks;
            var counter = 1;
            var videoStringBuilder = new StringBuilder();
            foreach (var track in videoTracks)
            {
                if (counter > 1)
                    videoStringBuilder.Append("]]");
                videoStringBuilder.Append(counter.ToString(CultureInfo.InvariantCulture) + ">>" + EscapeDelimiters(track.Description) +
                                          ">>" + track.Type);
                if (activeTrack != null && track.Description == activeTrack.Description)
                    videoStringBuilder.Append(">>True");
                else
                    videoStringBuilder.Append(">>False");
                counter++;
            }
            return "VideoTracks|" + videoStringBuilder;
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
                    audioStringBuilder.Append(counter.ToString(CultureInfo.InvariantCulture) + ">>" + EscapeDelimiters(track.Description) +
                                              ">>" + track.Type);
                    if (activeTrack != null && track.Description == activeTrack.Description)
                        audioStringBuilder.Append(">>True");
                    else
                        audioStringBuilder.Append(">>False");
                    counter++;
                }
                return "AudioTracks|" + audioStringBuilder;
            }
            return string.Empty;
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
                    subSb.Append(counter.ToString(CultureInfo.InvariantCulture) + ">>" + EscapeDelimiters(sub.Description) + ">>" +
                                 sub.Type);
                    if (activeSub != null && sub.Description == activeSub.Description)
                        subSb.Append(">>True");
                    else
                        subSb.Append(">>False");
                    counter++;
                }
                return "Subtitles|" + subSb;
            }
            return string.Empty;
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
                    chapterSb.Append(counter.ToString(CultureInfo.InvariantCulture) + ">>" + EscapeDelimiters(chapter.Name) + ">>" +
                                     chapter.Position.ToString(CultureInfo.InvariantCulture));
                    counter++;
                }
                return "Chapters|" + chapterSb;
            }
            return string.Empty;
        }

        private void PlayerControlPlaybackCompleted(object sender, EventArgs e)
        {
            PushToAllListeners("Finished|" + Media.FilePath);
        }

        private void ConnectedClientMenuClick()
        {
            if (m_ClientManager != null)
            {
                m_ClientManager.ShowDialog(Gui.VideoBox);
            }
            else
            {
                MessageBox.Show(Gui.VideoBox, "The remote control is not activated.");
            }
        }

        private void Server()
        {
            IPEndPoint localEndpoint = new IPEndPoint(IPAddress.Any, Settings.ConnectionPort);
            m_ServerSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            m_ServerSocket.Bind(localEndpoint);
            m_ServerSocket.Listen(10);
            while (true)
            {
                try
                {
                    var clientSocket = m_ServerSocket.Accept();
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
            Clients.TryAdd(clientGuid, client);

            NetworkStream nStream = new NetworkStream(client);
            StreamReader reader = new StreamReader(nStream);
            StreamWriter writer = new StreamWriter(nStream) {AutoFlush = false};
            m_Writers.TryAdd(clientGuid, writer);
            var authGuid = reader.ReadLine();
            if (!m_AuthHandler.IsGUIDAuthed(authGuid) && Settings.ValidateClients)
            {
                ClientAuth(writer, authGuid, clientGuid);
            }
            else
            {
                WriteToSpecificClient(writer, "Connected|Authorized");
                WriteToSpecificClient(writer, "ClientGUID|" + clientGuid);
                if (!m_AuthHandler.IsGUIDAuthed(authGuid))
                    m_AuthHandler.AddAuthedClient(authGuid);
                if (m_ClientManager.Visible)
                    m_ClientManager.ForceUpdate();
            }
            while (true)
            {
                try
                {
                    var data = reader.ReadLine();
                    try
                    {
                        if (string.IsNullOrWhiteSpace(data))
                        {
                            break;
                        }
                        data = UnsanatiseMessage(data);
                        if (data == "Exit")
                        {
                            HandleData(writer, data);
                            client.Close();
                        }
                        else
                        {
                            if (!HandleData(writer, data))
                            {
                                WriteToSpecificClient(writer, "Error|Command not supported on server");
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Trace.WriteLine(ex);
                    }
                }
                catch
                {
                    break;
                }
            }
            GuiThread.DoAsync(() => RemoveWriter(clientGuid));
        }

        private void ClientAuth(StreamWriter writer, string msgValue, Guid clientGuid)
        {
            WriteToSpecificClient(writer, "AuthCode|" + msgValue);
            var allow = false;
            GuiThread.Do(() =>
                allow =
                    MessageBox.Show(Gui.VideoBox, "Allow Remote Connection for " + msgValue, "Remote Authentication",
                        MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes);
            if (allow)
            {
                WriteToSpecificClient(writer, "Connected|Authorized");
                WriteToSpecificClient(writer, "ClientGUID|" + clientGuid);
                m_AuthHandler.AddAuthedClient(msgValue);
                if (m_ClientManager.Visible)
                    m_ClientManager.ForceUpdate();
            }
            else
            {
                DisconnectClient(writer, "Unauthorized", clientGuid);
            }
        }

        private string SanatiseMessage(string msg)
        {
            return msg.Replace("\r\n", "\n").Replace("\n", " \a ");
        }

        private string UnsanatiseMessage(string msg)
        {
            return msg.Replace(" \a ", Environment.NewLine);
        }

        private void WriteToSpecificClient(StreamWriter writer, string msg)
        {
            Task.Factory.StartNew(() =>
            {
                try
                {
                    lock (writer)
                    {
                        writer.WriteLine(SanatiseMessage(msg));
                        writer.Flush();
                    }
                }
                catch
                {
                }
            });
        }

        private void DisconnectClient(StreamWriter writer, string exitMessage, Guid clientGuid)
        {
            WriteToSpecificClient(writer, "Exit|" + exitMessage);

            Clients[clientGuid].Disconnect(true);
            RemoveWriter(clientGuid);
        }

        private bool HandleData(StreamWriter writer, string command)
        {
            var seperator = command.IndexOf('|');
            if (seperator < 0)
                return false;
            var cmd = command.Substring(0, seperator);
            var param = command.Substring(seperator + 1);
            switch (cmd)
            {
                case "Exit":
                    Guid clientGuid;
                    if (Guid.TryParse(param, out clientGuid))
                    {
                        GuiThread.DoAsync(() => RemoveWriter(clientGuid));
                    }
                    break;
                case "Open":
                    GuiThread.DoAsync(() => OpenMedia(param));
                    break;
                case "Pause":
                    GuiThread.DoAsync(() => PauseMedia(param));
                    break;
                case "Play":
                    GuiThread.DoAsync(() => PlayMedia(param));
                    break;
                case "Stop":
                    GuiThread.DoAsync(() => StopMedia(param));
                    break;
                case "Close":
                    GuiThread.DoAsync(() => CloseMedia(param));
                    break;
                case "Seek":
                    GuiThread.DoAsync(() => SeekMedia(param));
                    break;
                case "GetDuration":
                    GuiThread.DoAsync(() => GetFullDuration(writer));
                    break;
                case "GetCurrentState":
                    GuiThread.DoAsync(() => GetCurrentState(writer));
                    break;
                case "FullScreen":
                    GuiThread.DoAsync(() => FullScreen(param));
                    break;
                case "MoveWindow":
                    GuiThread.DoAsync(() => MoveWindow(param));
                    break;
                case "WriteToScreen":
                    DisplayTextMessage(param);
                    break;
                case "Mute":
                    bool mute;
                    bool.TryParse(param, out mute);
                    GuiThread.DoAsync(() => Mute(mute));
                    break;
                case "Volume":
                    int vol;
                    int.TryParse(param, NumberStyles.Number, CultureInfo.InvariantCulture, out vol);
                    GuiThread.DoAsync(() => SetVolume(vol));
                    break;
                case "ActiveSubTrack":
                    GuiThread.DoAsync(() => SetSubtitle(param));
                    break;
                case "ActiveAudioTrack":
                    GuiThread.DoAsync(() => SetAudioTrack(param));
                    break;
                case "AddFilesToPlaylist":
                    AddFilesToPlaylist(param);
                    break;
                case "InsertFileInPlaylist":
                {
                    var parameters = param.Split('|');
                    if (parameters.Length != 2)
                        return false;
                    GuiThread.DoAsync(() => InsertIntoPlaylist(parameters[0], parameters[1]));
                    break;
                }
                case "UpdatePlaylist":
                {
                    var data = Deserialize<UpdatePlaylistData>(param);
                    GuiThread.DoAsync(() => UpdatePlaylist(data.Playlist, data.ActiveIndex, data.CloseMedia));
                    break;
                }
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
                    GuiThread.DoAsync(() => GetPlaylist(writer));
                    break;
                case "PlaySelectedFile":
                    GuiThread.DoAsync(() => PlaySelectedFile(param));
                    break;
                case "RemoveFile":
                    GuiThread.DoAsync(() => RemoveFromPlaylist(param));
                    break;
                case "ActiveVideoTrack":
                    GuiThread.DoAsync(() => SetVideoTrack(param));
                    break;
                case "Dir":
                {
                    var parameters = param.Split('|');
                    if (parameters.Length == 1)
                    {
                        HandleDir(writer, param);
                    }
                    else
                    {
                        HandleDir(writer, parameters[0], parameters[1]);
                    }
                    break;
                }
                case "GetDriveLetters":
                    GetDriveLetters(writer);
                    break;
                default:
                    return false;
            }
            return true;
        }

        private void InsertIntoPlaylist(string fileIndex, string filePath)
        {
            SafeCall(() =>
            {
                int index;
                int.TryParse(fileIndex, NumberStyles.Number, CultureInfo.InvariantCulture, out index);
                PlaylistForm.InsertFile(index, filePath);
            });
        }

        private PlaylistForm PlaylistForm
        {
            get
            {
                if (m_PlaylistInstance != null)
                    return m_PlaylistInstance.GetPlaylistForm;

                var playlist = Extension.PlayerExtensions.FirstOrDefault(t => t.Descriptor.Guid == s_PlaylistGuid);
                if (playlist == null)
                {
                    throw new Exception("RemoteControl requires playlist extension to function");
                }

                m_PlaylistInstance = (Playlist.Playlist) playlist;
                return m_PlaylistInstance.GetPlaylistForm;
            }
        }

        private void UpdatePlaylist(PlaylistData playlist, int index, bool closeMedia)
        {
            SafeCall(() =>
            {
                if (closeMedia)
                {
                    Media.Close();
                    Player.ClearScreen();
                }
                PlaylistForm.UpdatePlaylist(playlist, index, closeMedia);
            });
        }

        private void RemoveFromPlaylist(string fileIndex)
        {
            int index;
            int.TryParse(fileIndex, NumberStyles.Number, CultureInfo.InvariantCulture, out index);
            SafeCall(() => PlaylistForm.RemoveFile(index));
        }

        private void PlaySelectedFile(string fileIndex)
        {
            int index;
            int.TryParse(fileIndex, NumberStyles.Number, CultureInfo.InvariantCulture, out index);
            SafeCall(() => PlaylistForm.SetPlaylistIndex(index));
        }

        private void GetPlaylist(StreamWriter writer)
        {
            var content =
                Serialize(new PlaylistData
                {
                    PlaylistName = PlaylistForm.PlaylistTitle,
                    Playlist = PlaylistForm.Playlist ?? new List<PlaylistItem>()
                });
            if (writer != null)
            {
                WriteToSpecificClient(writer, "PlaylistContent|" + content);
            }
            else
            {
                PushToAllListeners("PlaylistContent|" + content);
            }
        }

        private void HidePlaylist()
        {
            SafeCall(() => PlaylistForm.Hide());
        }

        private void ShowPlaylist()
        {
            SafeCall(() => PlaylistForm.Show());
        }

        private void PlaylistPlayNext()
        {
            SafeCall(() => PlaylistForm.PlayNext());
        }

        private void PlaylistPlayPrevious()
        {
            SafeCall(() => PlaylistForm.PlayPrevious());
        }

        public void FocusMpdn()
        {
            SafeCall(() => Player.ActiveForm.Focus());
        }

        private void ClearPlaylist()
        {
            SafeCall(() => PlaylistForm.NewPlaylist(true));
        }

        private void AddFilesToPlaylist(string files)
        {
            var playlistFiles = new List<string>();

            var filesToAdd = new List<string>();
            var filePaths = Regex.Split(files, ">>");
            if (filePaths.Any(t => t.EndsWith(".mpl")))
            {
                playlistFiles.AddRange(filePaths.Where(t => t.EndsWith(".mpl") && File.Exists(t)));
            }

            if (filePaths.Any())
            {
                filesToAdd.AddRange(filePaths.Where(t => !t.EndsWith(".mpl") && (File.Exists(t) || IsValidUrl(t))));
            }
            if (playlistFiles.Any())
            {
                foreach (var playlistFile in playlistFiles)
                {
                    var file = playlistFile;
                    GuiThread.DoAsync(() => PlaylistForm.OpenPlaylist(file));
                }
            }
            if (filesToAdd.Any())
            {
                GuiThread.DoAsync(() => PlaylistForm.AddFiles(filesToAdd.ToArray()));
            }
        }

        private bool IsValidUrl(string url)
        {
            Uri uriResult;
            return Uri.TryCreate(url, UriKind.Absolute, out uriResult)
                          && (uriResult.Scheme == Uri.UriSchemeHttp
                              || uriResult.Scheme == Uri.UriSchemeHttps);
        }

        private void RemoveWriter(Guid callerGuid)
        {
            StreamWriter writer;
            m_Writers.TryRemove(callerGuid, out writer);
            Socket socket;
            Clients.TryRemove(callerGuid, out socket);
            m_ClientManager.ForceUpdate();
        }

        private void SafeCall(Action action)
        {
            try
            {
                action();
            }
            catch (Exception ex)
            {
                GuiThread.DoAsync(() => PushToAllListeners("Error|" + "Failed to open media!\r\n" + ex.Message));
            }
        }

        private void OpenMedia(object file)
        {
            Task.Factory.StartNew(() =>
            {
                SafeCall(() =>
                {
                    var media = Media.Load(file.ToString());
                    GuiThread.DoAsync(() => Media.Open(media));
                });
            });
        }

        private void PauseMedia(object showOsd)
        {
            bool dispOsd;
            bool.TryParse(showOsd.ToString(), out dispOsd);
            SafeCall(() => Media.Pause(dispOsd));
        }

        private void PlayMedia(object showOsd)
        {
            bool dispOsd;
            bool.TryParse(showOsd.ToString(), out dispOsd);
            if (!string.IsNullOrEmpty(Media.FilePath))
            {
                SafeCall(() => Media.Play(dispOsd));
            }
            else
            {
                SafeCall(() => PlaylistForm.PlayActive());
            }
        }

        private void StopMedia(object blank)
        {
            SafeCall(Media.Stop);
        }

        private void CloseMedia(object blank)
        {
            SafeCall(() =>
            {
                Media.Close();
                Player.ClearScreen();
            });
        }

        private void SeekMedia(object seekLocation)
        {
            long location;
            if (!long.TryParse(seekLocation.ToString(), NumberStyles.Number, CultureInfo.InvariantCulture, out location))
                return;

            if (Player.State == PlayerState.Closed)
                return;

            SafeCall(() => Media.Seek(location));
        }

        private void SetVolume(int level)
        {
            SafeCall(() => Player.Volume = level);
        }

        private void SetSubtitle(string subDescription)
        {
            if (Player.State == PlayerState.Closed)
                return;

            SafeCall(() =>
            {
                var selTrack = Media.SubtitleTracks.FirstOrDefault(t => t.Description == subDescription);
                if (selTrack != null)
                    Media.SelectSubtitleTrack(selTrack);
            });
        }

        private void SetVideoTrack(string videoDescription)
        {
            if (Player.State == PlayerState.Closed)
                return;

            SafeCall(() =>
            {
                var selTrack = Media.VideoTracks.FirstOrDefault(t => t.Description == videoDescription);
                if (selTrack != null)
                    Media.SelectVideoTrack(selTrack);
            });
        }

        private void SetAudioTrack(string audioDescription)
        {
            if (Player.State == PlayerState.Closed)
                return;

            SafeCall(() =>
            {
                var selTrack = Media.AudioTracks.FirstOrDefault(t => t.Description == audioDescription);
                if (selTrack != null)
                    Media.SelectAudioTrack(selTrack);
            });
        }

        private void Mute(bool silence)
        {
            SafeCall(() => Player.Mute = silence);
            PushToAllListeners("Mute|" + silence.ToString(CultureInfo.InvariantCulture));
        }

        private void GetFullDuration(StreamWriter writer)
        {
            WriteToSpecificClient(writer, "FullLength|" + Media.Duration.ToString(CultureInfo.InvariantCulture));
        }

        private void GetCurrentState(StreamWriter writer)
        {
            WriteToSpecificClient(writer, "ServerVersion|" + SERVER_VERSION.ToString(CultureInfo.InvariantCulture));
            WriteToSpecificClient(writer, GetAllChapters());
            WriteToSpecificClient(writer, Player.State + "|" + Media.FilePath);
            WriteToSpecificClient(writer, "Fullscreen|" + Player.FullScreenMode.Active.ToString(CultureInfo.InvariantCulture));
            WriteToSpecificClient(writer, "Mute|" + Player.Mute.ToString(CultureInfo.InvariantCulture));
            WriteToSpecificClient(writer, "Volume|" + Player.Volume.ToString(CultureInfo.InvariantCulture));
            GetPlaylist(writer);
            if (Player.State != PlayerState.Closed)
            {
                WriteToSpecificClient(writer, "FullLength|" + Media.Duration.ToString(CultureInfo.InvariantCulture));
            }
            if (Player.State == PlayerState.Playing || Player.State == PlayerState.Paused)
            {
                WriteToSpecificClient(writer, "Position|" + Media.Position.ToString(CultureInfo.InvariantCulture));
            }
            PushToAllListeners("PlaylistShow|" + PlaylistForm.Visible.ToString(CultureInfo.InvariantCulture));
            WriteToSpecificClient(writer, GetAllSubtitleTracks());
            WriteToSpecificClient(writer, GetAllAudioTracks());
            WriteToSpecificClient(writer, GetAllVideoTracks());
        }

        private void FullScreen(string fullScreen)
        {
            bool goFullscreen;
            bool.TryParse(fullScreen, out goFullscreen);
            SafeCall(() => Player.FullScreenMode.Active = goFullscreen);
        }

        private void MoveWindow(string msg)
        {
            var args = msg.Split(new[] {">>"}, StringSplitOptions.None);

            int left, top, width, height;
            if (int.TryParse(args[0], NumberStyles.Number, CultureInfo.InvariantCulture, out left) &&
                int.TryParse(args[1], NumberStyles.Number, CultureInfo.InvariantCulture, out top) &&
                int.TryParse(args[2], NumberStyles.Number, CultureInfo.InvariantCulture, out width) &&
                int.TryParse(args[3], NumberStyles.Number, CultureInfo.InvariantCulture, out height))
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
            foreach (var writer in m_Writers)
            {
                var w = writer.Value;
                var guid = writer.Key;
                Task.Factory.StartNew(() =>
                {
                    try
                    {
                        lock (w)
                        {
                            w.WriteLine(SanatiseMessage(msg));
                            w.Flush();
                        }
                    }
                    catch
                    {
                        GuiThread.DoAsync(() => RemoveWriter(guid));
                    }
                });
            }
        }

        private string GetDirectoryListing(string path, IEnumerable<string> patterns)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                var lastMediaFile = Player.Config.Settings.GeneralSettings.LastSelectedMediaFileName;
                if (string.IsNullOrWhiteSpace(lastMediaFile) || IsValidUrl(lastMediaFile))
                {
                    path = Directory.GetCurrentDirectory();
                }
                else
                {
                    var lastMediaPath = MpdnPath.GetDirectoryName(lastMediaFile);
                    path = !lastMediaPath.StartsWith(@"\\") && Directory.Exists(lastMediaPath)
                        ? lastMediaPath
                        : Directory.GetCurrentDirectory();
                }
            }

            path = Path.GetFullPath(path);
            if (!Directory.Exists(path))
                return string.Empty;

            var sb = new StringBuilder();
            sb.Append(path);
            var directoryInfo = (new DirectoryInfo(path));
            foreach (var d in directoryInfo.GetDirectories())
            {
                sb.Append("]]D>>" + d.Name);
            }
            var files = patterns.SelectMany(s => directoryInfo.EnumerateFiles(s));
            foreach (var f in files)
            {
                sb.Append("]]F>>" + f.Name + ">>" + f.Length.ToString(CultureInfo.InvariantCulture));
            }

            return sb.ToString();
        }

        private void HandleDir(StreamWriter writer, string path, string pattern = null)
        {
            var patterns = new List<string>();
            patterns.AddRange(string.IsNullOrWhiteSpace(pattern)
                ? Player.RegisteredMediaExtensions.Select(s => "*" + s)
                : pattern.Split(';'));

            WriteToSpecificClient(writer, "Dir|" + GetDirectoryListing(path, patterns));
        }

        private void GetDriveLetters(StreamWriter writer)
        {
            WriteToSpecificClient(writer, "DriveLetters|" + string.Join("]]", Directory.GetLogicalDrives()));
        }

        private void DisplayTextMessage(object msg)
        {
            GuiThread.DoAsync(() => Player.OsdText.Show(msg.ToString()));
        }

        public void DisconnectClient(string guid)
        {
            Guid clientGuid;
            if (!Guid.TryParse(guid, out clientGuid))
                return;
            StreamWriter writer;
            if (!m_Writers.TryGetValue(clientGuid, out writer))
                return;
            DisconnectClient(writer, "Disconnected by User", clientGuid);
        }

        private void _locationTimer_Elapsed(object sender, EventArgs eventArgs)
        {
            try
            {
                if (m_LastPosition == Media.Position) return;
                m_LastPosition = Media.Position;
                PushToAllListeners("Postion|" + Media.Position.ToString(CultureInfo.InvariantCulture));
            }
            catch
            {
            }
        }

        private static string Serialize<T>(T value)
        {
            return CreateSerializer(typeof(T)).Serialize(value);
        }

        private static T Deserialize<T>(string serializedObject)
        {
            return (T) CreateSerializer(typeof(T)).Deserialize(serializedObject);
        }

        private static YAXSerializer CreateSerializer(Type type)
        {
            return new YAXSerializer(type, YAXExceptionHandlingPolicies.DoNotThrow, YAXExceptionTypes.Ignore,
                YAXSerializationOptions.DontSerializeCyclingReferences |
                YAXSerializationOptions.DontSerializePropertiesWithNoSetter |
                YAXSerializationOptions.SerializeNullObjects);
        }

        #region Variables

        private Socket m_ServerSocket;
        private readonly ConcurrentDictionary<Guid, StreamWriter> m_Writers = new ConcurrentDictionary<Guid, StreamWriter>();
        private readonly RemoteControl_AuthHandler m_AuthHandler = new RemoteControl_AuthHandler();
        private RemoteClients m_ClientManager;
        private Timer m_LocationTimer;
        private static readonly Guid s_PlaylistGuid = new Guid("A1997E34-D67B-43BB-8FE6-55A71AE7184B");
        private Playlist.Playlist m_PlaylistInstance;
        private long m_LastPosition = -1;

        #endregion
    }

    public class RemoteControlSettings : INotifyPropertyChanged
    {
        #region Public Methods

        public RemoteControlSettings()
        {
            ConnectionPort = 6545;
            ValidateClients = false;
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

namespace Mpdn.Extensions.PlayerExtensions.DataContracts
{
    public class UpdatePlaylistData
    {
        public int ActiveIndex { get; set; }
        public bool CloseMedia { get; set; }
        public PlaylistData Playlist { get; set; }
    }

    public class PlaylistData
    {
        public string PlaylistName { get; set; }
        public List<PlaylistItem> Playlist { get; set; }
    }
}