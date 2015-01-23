using Mpdn;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Threading;

namespace Mpdn.PlayerExtensions
{
    public class ACMPlug : ConfigurablePlayerExtension<RemoteControlSettings, RemoteControlConfig>
    {
        #region Variables
        private Socket serverSocket;
        private SynchronizationContext context;
        private Dictionary<Guid, StreamWriter> writers = new Dictionary<Guid, StreamWriter>();
        private Dictionary<Guid, Socket> clients = new Dictionary<Guid, Socket>();
        private RemoteControl_AuthHandler authHandler = new RemoteControl_AuthHandler();
        private System.Timers.Timer hideTimer;
        private RemoteClients clientManager;
        #endregion

        #region Properties
        public Dictionary<Guid, Socket> getClients
        {
            get { return clients; }
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
            foreach (var writer in writers)
            {
                try
                {
                    writer.Value.WriteLine("Closing|Close");
                    writer.Value.Flush();
                }
                catch
                { }
            }
            serverSocket.Close();
        }

        public override void Initialize()
        {
            base.Initialize();
            context = WindowsFormsSynchronizationContext.Current;
            PlayerControl.PlaybackCompleted += m_PlayerControl_PlaybackCompleted;
            PlayerControl.PlayerStateChanged += m_PlayerControl_PlayerStateChanged;
            PlayerControl.EnteringFullScreenMode += m_PlayerControl_EnteringFullScreenMode;
            PlayerControl.ExitingFullScreenMode += m_PlayerControl_ExitingFullScreenMode;
            clientManager = new RemoteClients(this);
            Task.Factory.StartNew(Server);
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
            foreach (var writer in writers)
            {
                try
                {
                    writer.Value.WriteLine(e.NewState.ToString() + "|" + PlayerControl.MediaFilePath);
                    writer.Value.Flush();
                }
                catch
                { }
            }
        }

        void m_PlayerControl_PlaybackCompleted(object sender, EventArgs e)
        {
            foreach(var writer in writers)
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
            clientManager.ShowDialog();
        }

        private void Server()
        {
            IPEndPoint localEndpoint = new IPEndPoint(IPAddress.Any, Settings.ConnectionPort);
            serverSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            serverSocket.Bind(localEndpoint);
            serverSocket.Listen(10);
            while (true)
            {
                try
                {
                    var clientSocket = serverSocket.Accept();
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
            clients.Add(clientGuid, client);

            NetworkStream nStream = new NetworkStream(client);
            StreamReader reader = new StreamReader(nStream);
            StreamWriter writer = new StreamWriter(nStream);
            writers.Add(clientGuid, writer);
            var clientGUID = reader.ReadLine();
            if (!authHandler.IsGUIDAuthed(clientGUID))
            {
                ClientAuth(clientGUID.ToString(), clientGuid);
            }
            else
            {
                DisplayTextMessage("Remote Connected");
                WriteToSpesificClient("Connected|Authorized", clientGuid.ToString());
                WriteToSpesificClient("ClientGUID|" + clientGuid.ToString(), clientGuid.ToString());
                authHandler.AddAuthedClient(clientGUID);
                if(clientManager.Visible)
                    clientManager.ForceUpdate();
            }
            while (true)
            {
                try
                {
                    var data = reader.ReadLine();
                    if (data == "Exit")
                    {
                        handleData(data);
                        client.Close();
                        break;
                    }
                    else
                    {
                        handleData(data);
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
                authHandler.AddAuthedClient(msgValue);
                if (clientManager.Visible)
                    clientManager.ForceUpdate();
            }
            else
            {
                DisconnectClient("Unauthorized", clientGuid);
            }
        }

        private void WriteToSpesificClient(string msg, string clientGUID)
        {
            Guid pushGUID;
            Guid.TryParse(clientGUID, out pushGUID);

            if (pushGUID != null)
            {
                if (writers.ContainsKey(pushGUID))
                {
                    writers[pushGUID].WriteLine(msg);
                    writers[pushGUID].Flush();
                }
            }
        }

        private void DisconnectClient(string exitMessage, Guid clientGUID)
        {
            WriteToSpesificClient("Exit|" + exitMessage, clientGUID.ToString());

            clients[clientGUID].Disconnect(true);
            RemoveWriter(clientGUID.ToString());
        }

        private void handleData(string data)
        {
            var command = data.Split('|');
            switch(command[0])
            {
                case "Exit":
                    DisplayTextMessage("Remote Disconnected");
                    RemoveWriter(command[1]);
                    break;
                case "Open":
                    context.Send(new SendOrPostCallback(OpenMedia), command[1]);
                    break;
                case "Pause":
                    context.Send(new SendOrPostCallback(PauseMedia), command[1]);
                    break;
                case "Play":
                    context.Send(new SendOrPostCallback(PlayMedia), command[1]);
                    break;
                case "Stop":
                    context.Send(new SendOrPostCallback(StopMedia), command[1]);
                    break;
                case "Seek":
                    context.Send(new SendOrPostCallback(SeekMedia), command[1]);
                    break;
                case "GetLocation":
                    context.Send(new SendOrPostCallback(GetLocation), command[1]);
                    break;
                case "GetDuration":
                    context.Send(new SendOrPostCallback(GetFullDuration), command[1]);
                    break;
                case "GetCurrentState":
                    context.Send(new SendOrPostCallback(GetCurrentState), command[1]);
                    break;
                case "FullScreen":
                    context.Send(new SendOrPostCallback(FullScreen), command[1]);
                    break;
                case "WriteToScreen":
                    context.Send(new SendOrPostCallback(DisplayTextMessage), command[1]);
                    break;
            }
        }

        private void RemoveWriter(string GUID)
        {
            Guid callerGUID = Guid.Parse(GUID);
            writers.Remove(callerGUID);
            clients.Remove(callerGUID);
            clientManager.ForceUpdate();
        }

        private void OpenMedia(object file)
        {
            PlayerControl.OpenMedia(file.ToString());
        }

        private void PauseMedia(object showOSD)
        {
            bool dispOSD = false;
            Boolean.TryParse(showOSD.ToString(), out dispOSD);
            PlayerControl.PauseMedia(dispOSD);
        }

        private void PlayMedia(object showOSD)
        {
            bool dispOSD = false;
            Boolean.TryParse(showOSD.ToString(), out dispOSD);
            PlayerControl.PlayMedia(dispOSD);
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
                PlayerControl.PlayMedia();
                PlayerControl.PauseMedia();
            }
        }

        private void GetLocation(object GUID)
        {
            WriteToSpesificClient("Postion|" + PlayerControl.MediaPosition, GUID.ToString());
        }

        private void GetFullDuration(object GUID)
        {
            WriteToSpesificClient("FullLength|" + PlayerControl.MediaDuration, GUID.ToString());
        }

        private void GetCurrentState(object GUID)
        {
            WriteToSpesificClient(PlayerControl.PlayerState + "|" + PlayerControl.MediaFilePath, GUID.ToString());
            WriteToSpesificClient("Fullscreen|" + PlayerControl.InFullScreenMode, GUID.ToString());
        }

        private void FullScreen(object FullScreen)
        {
            bool goFullscreen = false;
            Boolean.TryParse(FullScreen.ToString(), out goFullscreen);
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
            foreach (var writer in writers)
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
            PlayerControl.ShowOsdText(msg.ToString());
            //This is a temporary workaround as ShowOsdText doesn't seem to auto hide OSD text
            hideTimer = new System.Timers.Timer(1000);
            hideTimer.Elapsed += hideTimer_Elapsed;
            hideTimer.AutoReset = false;
            hideTimer.Start();
        }

        void hideTimer_Elapsed(object sender, System.Timers.ElapsedEventArgs e)
        {
            PlayerControl.HideOsdText();
        }

        public void DisconnectClient(string GUID)
        {
            Guid clientGuid;
            Guid.TryParse(GUID, out clientGuid);
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
