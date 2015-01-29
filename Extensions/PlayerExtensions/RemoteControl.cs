using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Timers;
using System.Windows.Forms;
using Timer = System.Timers.Timer;

namespace Mpdn.PlayerExtensions
{
    public class AcmPlug : ConfigurablePlayerExtension<RemoteControlSettings, RemoteControlConfig>
    {
        #region Variables
        private Socket _serverSocket;
        private readonly Dictionary<Guid, StreamWriter> _writers = new Dictionary<Guid, StreamWriter>();
        private readonly Dictionary<Guid, Socket> _clients = new Dictionary<Guid, Socket>();
        private readonly RemoteControlAuthHandler _authHandler = new RemoteControlAuthHandler();
        private RemoteClients _clientManager;
        private Timer _locationTimer;
        #endregion

        #region Properties
        public Dictionary<Guid, Socket> GetClients
        {
            get { return _clients; }
        }
        #endregion

        #region Overwritten Methods

        protected override PlayerExtensionDescriptor ScriptDescriptor
        {
            get 
            {
                return new PlayerExtensionDescriptor
                {
                    Guid = new Guid("C7FC1078-6471-409D-A2F1-34FF8903D6DA"),
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

        public override void Destroy()
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
                { }
            }
            _serverSocket.Close();
        }

        public override void Initialize()
        {
            base.Initialize();
            Subscribe();
            _clientManager = new RemoteClients(this);
            _locationTimer = new Timer(100);
            _locationTimer.Elapsed += _locationTimer_Elapsed;
            Task.Factory.StartNew(Server);
        }

        #endregion

        private void Subscribe()
        {
            PlayerControl.PlaybackCompleted += m_PlayerControl_PlaybackCompleted;
            PlayerControl.PlayerStateChanged += m_PlayerControl_PlayerStateChanged;
            PlayerControl.EnteringFullScreenMode += m_PlayerControl_EnteringFullScreenMode;
            PlayerControl.ExitingFullScreenMode += m_PlayerControl_ExitingFullScreenMode;
            PlayerControl.VolumeChanged += PlayerControl_VolumeChanged;
            PlayerControl.SubtitleTrackChanged += PlayerControl_SubtitleTrackChanged;
            PlayerControl.AudioTrackChanged += PlayerControl_AudioTrackChanged;
        }

        private void Unsubscribe()
        {
            PlayerControl.PlaybackCompleted += m_PlayerControl_PlaybackCompleted;
            PlayerControl.PlayerStateChanged += m_PlayerControl_PlayerStateChanged;
            PlayerControl.EnteringFullScreenMode += m_PlayerControl_EnteringFullScreenMode;
            PlayerControl.ExitingFullScreenMode += m_PlayerControl_ExitingFullScreenMode;
            PlayerControl.VolumeChanged += PlayerControl_VolumeChanged;
            PlayerControl.SubtitleTrackChanged -= PlayerControl_SubtitleTrackChanged;
        }

        void PlayerControl_AudioTrackChanged(object sender, EventArgs e)
        {
            PushToAllListeners("AudioChanged|" + PlayerControl.ActiveAudioTrack.Description);
        }

        void PlayerControl_SubtitleTrackChanged(object sender, EventArgs e)
        {
            PushToAllListeners("SubChanged|" + PlayerControl.ActiveSubtitleTrack.Description);
        }

        void PlayerControl_VolumeChanged(object sender, EventArgs e)
        {
            PushToAllListeners("Volume|" + PlayerControl.Volume.ToString());
            PushToAllListeners("Mute|" + PlayerControl.Mute.ToString());
        }

        void _locationTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            try
            {
                PushToAllListeners("Postion|" + PlayerControl.MediaPosition);
            }
            catch(Exception)
            { }
        }

        void m_PlayerControl_ExitingFullScreenMode(object sender, EventArgs e)
        {
            PushToAllListeners("Fullscreen|False");
        }

        void m_PlayerControl_EnteringFullScreenMode(object sender, EventArgs e)
        {
            PushToAllListeners("Fullscreen|True");
        }


        void m_PlayerControl_PlayerStateChanged(object sender, PlayerStateEventArgs e)
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

            PushToAllListeners(e.NewState + "|" + PlayerControl.MediaFilePath);
        }

        private string GetAllAudioTracks()
        {
            if (PlayerControl.PlayerState == PlayerState.Playing || PlayerControl.PlayerState == PlayerState.Paused)
            {
                MediaTrack activeTrack = null;
                if (PlayerControl.ActiveAudioTrack != null)
                    activeTrack = PlayerControl.ActiveAudioTrack;
                var audioTracks = PlayerControl.AudioTracks;
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
            else
            {
                return String.Empty;
            }
        }

        private string GetAllSubtitleTracks()
        {
            if(PlayerControl.PlayerState == PlayerState.Playing || PlayerControl.PlayerState == PlayerState.Paused)
            {
                MediaTrack activeSub = null;
                if (PlayerControl.ActiveSubtitleTrack != null)
                    activeSub = PlayerControl.ActiveSubtitleTrack;
                var subtitles = PlayerControl.SubtitleTracks;
                int counter = 1;
                StringBuilder subSb = new StringBuilder();
                foreach(var sub in subtitles)
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
            else
            {
                return String.Empty;
            }
        }

        private string GetAllChapters()
        {
            if (PlayerControl.PlayerState == PlayerState.Playing || PlayerControl.PlayerState == PlayerState.Paused)
            {
                var chapters = PlayerControl.Chapters;
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
            else
            {
                return String.Empty;
            }
        }

        void m_PlayerControl_PlaybackCompleted(object sender, EventArgs e)
        {
            foreach(var writer in _writers)
            {
                try
                {
                    writer.Value.WriteLine("Finished" + "|" + PlayerControl.MediaFilePath);
                    writer.Value.Flush();
                }
                catch
                { }
            }
        }

        public override IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.Help, string.Empty, "Connected Clients", "Ctrl+Shift+R", "Show Remote Client connections", Test1Click),
                };
            }
        }

        private void Test1Click()
        {
            _clientManager.ShowDialog();
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
            _clients.Add(clientGuid, client);

            NetworkStream nStream = new NetworkStream(client);
            StreamReader reader = new StreamReader(nStream);
            StreamWriter writer = new StreamWriter(nStream);
            _writers.Add(clientGuid, writer);
            var clientGUID = reader.ReadLine();
            if (!_authHandler.IsGuidAuthed(clientGUID))
            {
                ClientAuth(clientGUID, clientGuid);
            }
            else
            {
                DisplayTextMessage("Remote Connected");
                WriteToSpesificClient("Connected|Authorized", clientGuid.ToString());
                WriteToSpesificClient("ClientGUID|" + clientGuid.ToString(), clientGuid.ToString());
                _authHandler.AddAuthedClient(clientGUID);
                if(_clientManager.Visible)
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
                    else
                    {
                        HandleData(data);
                    }
                }
                catch(Exception ex)
                {
                    break;
                }
            }
        }

        private void ClientAuth(string msgValue, Guid clientGuid)
        {
            WriteToSpesificClient("AuthCode|" + msgValue, clientGuid.ToString());
            if(MessageBox.Show("Allow Remote Connection for " + msgValue, "Remote Authentication", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
            {
                DisplayTextMessage("Remote Connected");
                WriteToSpesificClient("Connected|Authorized", clientGuid.ToString());
                _authHandler.AddAuthedClient(msgValue);
                if (_clientManager.Visible)
                    _clientManager.ForceUpdate();
            }
            else
            {
                DisconnectClient("Unauthorized", clientGuid);
            }
        }

        private void WriteToSpesificClient(string msg, string clientGuid)
        {
            Guid pushGuid;
            Guid.TryParse(clientGuid, out pushGuid);

            if (pushGuid != null)
            {
                if (_writers.ContainsKey(pushGuid))
                {
                    _writers[pushGuid].WriteLine(msg);
                    _writers[pushGuid].Flush();
                }
            }
        }

        private void DisconnectClient(string exitMessage, Guid clientGuid)
        {
            WriteToSpesificClient("Exit|" + exitMessage, clientGuid.ToString());

            _clients[clientGuid].Disconnect(true);
            RemoveWriter(clientGuid.ToString());
        }

        private void HandleData(string data)
        {
            var command = data.Split('|');
            switch(command[0])
            {
                case "Exit":
                    DisplayTextMessage("Remote Disconnected");
                    RemoveWriter(command[1]);
                    break;
                case "Open":
                    PlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => OpenMedia(command[1])));
                    break;
                case "Pause":
                    PlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => PauseMedia(command[1])));
                    break;
                case "Play":
                    PlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => PlayMedia(command[1])));
                    break;
                case "Stop":
                    PlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => StopMedia(command[1])));
                    break;
                case "Seek":
                    PlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => SeekMedia(command[1])));
                    break;
                case "GetDuration":
                    PlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => GetFullDuration(command[1])));
                    break;
                case "GetCurrentState":
                    PlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => GetCurrentState(command[1])));
                    break;
                case "FullScreen":
                    PlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => FullScreen(command[1])));
                    break;
                case "WriteToScreen":
                    DisplayTextMessage(command[1]);
                    break;
                case "Mute":
                    bool mute = false;
                    Boolean.TryParse(command[1], out mute);
                    PlayerControl.VideoPanel.BeginInvoke((MethodInvoker)(() => Mute(mute)));
                    break;
                case "Volume":
                    int vol = 0;
                    int.TryParse(command[1], out vol);
                    PlayerControl.VideoPanel.BeginInvoke((MethodInvoker)(() => SetVolume(vol)));
                    break;
                case "ActiveSubTrack":
                    PlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => SetSubtitle(command[1])));
                    break;
            }
        }

        private void RemoveWriter(string guid)
        {
            Guid callerGuid = Guid.Parse(guid);
            _writers.Remove(callerGuid);
            _clients.Remove(callerGuid);
            _clientManager.ForceUpdate();
        }

        private void OpenMedia(object file)
        {
            PlayerControl.OpenMedia(file.ToString());
        }

        private void PauseMedia(object showOsd)
        {
            bool dispOsd = false;
            Boolean.TryParse(showOsd.ToString(), out dispOsd);
            PlayerControl.PauseMedia(dispOsd);
        }

        private void PlayMedia(object showOsd)
        {
            bool dispOsd = false;
            Boolean.TryParse(showOsd.ToString(), out dispOsd);
            PlayerControl.PlayMedia(dispOsd);
        }

        private void StopMedia(object blank)
        {
            PlayerControl.StopMedia();
        }

        private void SeekMedia(object seekLocation)
        {
            double location = -1;
            double.TryParse(seekLocation.ToString(), out location);
            if(location != -1)
            {
                PlayerControl.SeekMedia((long)location);
            }
        }

        private void GetFullDuration(object guid)
        {
            WriteToSpesificClient("FullLength|" + PlayerControl.MediaDuration, guid.ToString());
        }

        private void GetCurrentState(object guid)
        {
            WriteToSpesificClient(GetAllChapters(), guid.ToString());
            WriteToSpesificClient(PlayerControl.PlayerState + "|" + PlayerControl.MediaFilePath, guid.ToString());
            WriteToSpesificClient("Fullscreen|" + PlayerControl.InFullScreenMode, guid.ToString());
            WriteToSpesificClient("Mute|" + PlayerControl.Mute, guid.ToString());
            WriteToSpesificClient("Volume|" + PlayerControl.Volume, guid.ToString());
            WriteToSpesificClient("FullLength|" + PlayerControl.MediaDuration, guid.ToString());
            WriteToSpesificClient("Postion|" + PlayerControl.MediaPosition, guid.ToString());
            WriteToSpesificClient(GetAllSubtitleTracks(), guid.ToString());
            WriteToSpesificClient(GetAllAudioTracks(), guid.ToString());
        }

        private void FullScreen(object fullScreen)
        {
            bool goFullscreen = false;
            Boolean.TryParse(fullScreen.ToString(), out goFullscreen);
            if(goFullscreen)
            {
                PlayerControl.GoFullScreen();
            }
            else
            {
                PlayerControl.GoWindowed();
            }
        }

        private void PushToAllListeners(string msg)
        {
            foreach (var writer in _writers)
            {
                try
                {
                    try
                    {

                        writer.Value.WriteLine(msg);
                        writer.Value.Flush();
                    }
                    catch(Exception)
                    { }
                }
                catch
                { }
            }
        }

        private void DisplayTextMessage(object msg)
        {
            PlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => PlayerControl.ShowOsdText(msg.ToString())));
        }

        private void Mute(bool silence)
        {
            PlayerControl.Mute = silence;
            PushToAllListeners("Mute|" + silence);
        }

        public void DisconnectClient(string guid)
        {
            Guid clientGuid;
            Guid.TryParse(guid, out clientGuid);
            DisconnectClient("Disconnected by User", clientGuid);
        }

        private void SetVolume(int level)
        {
            PlayerControl.Volume = level;
        }
        
        private void SetSubtitle(string subDescription)
        {
            var selTrack = PlayerControl.SubtitleTracks.FirstOrDefault(t => t.Description == subDescription);
           if(selTrack != null)
            PlayerControl.SelectSubtitleTrack(selTrack);
        }
    }

    public class RemoteControlSettings
    {
        #region Variables
        public int ConnectionPort { get; set; }
        #endregion

        public RemoteControlSettings()
        {
            ConnectionPort = 6545;
        }
    }
}
