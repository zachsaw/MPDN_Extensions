using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
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
        private IPlayerControl _mPlayerControl;
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

        public override void Initialize(IPlayerControl playerControl)
        {
            base.Initialize(playerControl);
            _mPlayerControl = playerControl;
            _mPlayerControl.PlaybackCompleted += m_PlayerControl_PlaybackCompleted;
            _mPlayerControl.PlayerStateChanged += m_PlayerControl_PlayerStateChanged;
            _mPlayerControl.EnteringFullScreenMode += m_PlayerControl_EnteringFullScreenMode;
            _mPlayerControl.ExitingFullScreenMode += m_PlayerControl_ExitingFullScreenMode;
            _clientManager = new RemoteClients(this);
            _locationTimer = new Timer(100);
            _locationTimer.Elapsed += _locationTimer_Elapsed;
            Task.Factory.StartNew(Server);
        }

        void _locationTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            PushToAllListeners("Postion|" + _mPlayerControl.MediaPosition);
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
                        break;
                    case PlayerState.Stopped:
                        _locationTimer.Stop();
                        break;
                    case PlayerState.Paused:
                        _locationTimer.Stop();
                        break;
            }

            PushToAllListeners(e.NewState + "|" + _mPlayerControl.MediaFilePath);
        }

        void m_PlayerControl_PlaybackCompleted(object sender, EventArgs e)
        {
            foreach(var writer in _writers)
            {
                try
                {
                    writer.Value.WriteLine("Finished" + "|" + _mPlayerControl.MediaFilePath);
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
                ClientAuth(clientGUID.ToString(), clientGuid);
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
                    _mPlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => OpenMedia(command[1])));
                    break;
                case "Pause":
                    _mPlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => PauseMedia(command[1])));
                    break;
                case "Play":
                    _mPlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => PlayMedia(command[1])));
                    break;
                case "Stop":
                    _mPlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => StopMedia(command[1])));
                    break;
                case "Seek":
                    _mPlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => SeekMedia(command[1])));
                    break;
                case "GetDuration":
                    _mPlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => GetFullDuration(command[1])));
                    break;
                case "GetCurrentState":
                    _mPlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => GetCurrentState(command[1])));
                    break;
                case "FullScreen":
                    _mPlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => FullScreen(command[1])));
                    break;
                case "WriteToScreen":
                    DisplayTextMessage(command[1]);
                    break;
                case "Mute":
                    bool mute = false;
                    Boolean.TryParse(command[1], out mute);
                    _mPlayerControl.VideoPanel.BeginInvoke((MethodInvoker)(() => Mute(mute)));
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
            _mPlayerControl.OpenMedia(file.ToString());
        }

        private void PauseMedia(object showOsd)
        {
            bool dispOsd = false;
            Boolean.TryParse(showOsd.ToString(), out dispOsd);
            _mPlayerControl.PauseMedia(dispOsd);
        }

        private void PlayMedia(object showOsd)
        {
            bool dispOsd = false;
            Boolean.TryParse(showOsd.ToString(), out dispOsd);
            _mPlayerControl.PlayMedia(dispOsd);
        }

        private void StopMedia(object blank)
        {
            _mPlayerControl.StopMedia();
        }

        private void SeekMedia(object seekLocation)
        {
            double location = -1;
            double.TryParse(seekLocation.ToString(), out location);
            if(location != -1)
            {
                _mPlayerControl.SeekMedia((long)location);
                _mPlayerControl.PlayMedia();
                _mPlayerControl.PauseMedia();
            }
        }

        private void GetFullDuration(object guid)
        {
            WriteToSpesificClient("FullLength|" + _mPlayerControl.MediaDuration, guid.ToString());
        }

        private void GetCurrentState(object guid)
        {
            WriteToSpesificClient(_mPlayerControl.PlayerState + "|" + _mPlayerControl.MediaFilePath, guid.ToString());
            WriteToSpesificClient("Fullscreen|" + _mPlayerControl.InFullScreenMode, guid.ToString());
            WriteToSpesificClient("Mute|" + _mPlayerControl.Mute, guid.ToString());
        }

        private void FullScreen(object fullScreen)
        {
            bool goFullscreen = false;
            Boolean.TryParse(fullScreen.ToString(), out goFullscreen);
            if(goFullscreen)
            {
                _mPlayerControl.GoFullScreen();
            }
            else
            {
                _mPlayerControl.GoWindowed();
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
                { }
            }
        }

        private void DisplayTextMessage(object msg)
        {
            _mPlayerControl.VideoPanel.BeginInvoke((MethodInvoker) (() => _mPlayerControl.ShowOsdText(msg.ToString())));
        }

        private void Mute(bool silence)
        {
            _mPlayerControl.Mute = silence;
            PushToAllListeners("Mute|" + silence);
        }

        public void DisconnectClient(string guid)
        {
            Guid clientGuid;
            Guid.TryParse(guid, out clientGuid);
            DisconnectClient("Disconnected by User", clientGuid);
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
