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

namespace ACMPlugin
{
    public class ACMPlug : IPlayerExtension
    {
        #region Variables
        private Socket serverSocket;
        private IPlayerControl m_PlayerControl;
        private SynchronizationContext context;
        private Dictionary<Guid, StreamWriter> writers = new Dictionary<Guid, StreamWriter>();
        #endregion

        public ExtensionDescriptor Descriptor
        {
            get 
            {
                return new ExtensionDescriptor
                {
                    Guid = new Guid("C7FC1078-6471-409D-A2F1-34FF8903D6DA"),
                    Name = "Remote Control",
                    Description = "Remote Control extension to allow control of MPDN over the network. Server example can be found here: https://github.com/DeadlyEmbrace/MPDN_RemoteControl",
                    Copyright = "Copyright DeadlyEmbrace © 2015. All rights reserved."
                };
            }
        }

        public void Destroy()
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

        public void Initialize(IPlayerControl playerControl)
        {
            context = WindowsFormsSynchronizationContext.Current;
            m_PlayerControl = playerControl;
            m_PlayerControl.PlaybackCompleted += m_PlayerControl_PlaybackCompleted;
            m_PlayerControl.PlayerStateChanged += m_PlayerControl_PlayerStateChanged;
            m_PlayerControl.EnteringFullScreenMode += m_PlayerControl_EnteringFullScreenMode;
            m_PlayerControl.ExitingFullScreenMode += m_PlayerControl_ExitingFullScreenMode;
            Task.Run(() => Server());
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
                    writer.Value.WriteLine(e.NewState.ToString() + "|" + m_PlayerControl.MediaFilePath);
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
                    writer.Value.WriteLine("Finished" + "|" + m_PlayerControl.MediaFilePath);
                    writer.Value.Flush();
                }
                catch
                { }
            }
        }


        public bool ShowConfigDialog(System.Windows.Forms.IWin32Window owner)
        {
            return false;
        }

        public IList<Verb> Verbs
        {
            get
            {
                return new[]
                {
                    new Verb(Category.Help, string.Empty, "Test", "Ctrl+Shift+T", "test test test 1", Test1Click),
                };
            }
        }

        private void Test1Click()
        {
            MessageBox.Show("Hello World!");
        }

        private void Server()
        {
            IPEndPoint localEndpoint = new IPEndPoint(IPAddress.Any, 6545);
            serverSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            serverSocket.Bind(localEndpoint);
            serverSocket.Listen(10);
            while (true)
            {
                try
                {
                    var clientSocket = serverSocket.Accept();
                    Task.Run(() => ClientHandler(clientSocket));
                }
                catch (Exception ex)
                {
                    break;
                }
            }
        }

        private void ClientHandler(Socket client)
        {
            Guid clientGuid = Guid.NewGuid();

            NetworkStream nStream = new NetworkStream(client);
            StreamReader reader = new StreamReader(nStream);
            StreamWriter writer = new StreamWriter(nStream);
            writers.Add(clientGuid, writer);
            writer.WriteLine("ClientGUID|" + clientGuid);
            writer.Flush();

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

        private void handleData(string data)
        {
            var command = data.Split('|');
            switch(command[0])
            {
                case "Exit":
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
                    break
            }
        }

        private void RemoveWriter(string GUID)
        {
            Guid callerGUID = Guid.Parse(GUID);
            writers.Remove(callerGUID);
        }

        private void OpenMedia(object file)
        {
            m_PlayerControl.OpenMedia(file.ToString());
        }

        private void PauseMedia(object showOSD)
        {
            bool dispOSD = false;
            Boolean.TryParse(showOSD.ToString(), out dispOSD);
            m_PlayerControl.PauseMedia(dispOSD);
        }

        private void PlayMedia(object showOSD)
        {
            bool dispOSD = false;
            Boolean.TryParse(showOSD.ToString(), out dispOSD);
            m_PlayerControl.PlayMedia(dispOSD);
        }

        private void StopMedia(object blank)
        {
            m_PlayerControl.StopMedia();
        }

        private void SeekMedia(object seekLocation)
        {
            double location = -1;
            double.TryParse(seekLocation.ToString(), out location);
            if(location != -1)
            {
                m_PlayerControl.SeekMedia((long)location);
                m_PlayerControl.PlayMedia();
                m_PlayerControl.PauseMedia();
            }
        }

        private void GetLocation(object GUID)
        {
            Guid callerGUID = Guid.Parse(GUID.ToString());
            var callerItem = writers[callerGUID];
            if (callerItem != null)
            {
                callerItem.WriteLine("Postion|" + m_PlayerControl.MediaPosition);
                callerItem.Flush();
            }
        }

        private void GetFullDuration(object GUID)
        {
            Guid callerGUID = Guid.Parse(GUID.ToString());
            var callerItem = writers[callerGUID];
            if(callerItem != null)
            {
                callerItem.WriteLine("FullLength|" + m_PlayerControl.MediaDuration);
                callerItem.Flush();
            }
        }

        private void GetCurrentState(object GUID)
        {
            Guid callerGUID = Guid.Parse(GUID.ToString());
            var callerItem = writers[callerGUID];
            if (callerItem != null)
            {
                callerItem.WriteLine(m_PlayerControl.PlayerState + "|" + m_PlayerControl.MediaFilePath);
                callerItem.Flush();
                callerItem.WriteLine("Fullscreen|" + m_PlayerControl.InFullScreenMode);
                callerItem.Flush();
            }
        }

        private void FullScreen(object FullScreen)
        {
            bool goFullscreen = false;
            Boolean.TryParse(FullScreen.ToString(), out goFullscreen);
            if(goFullscreen)
            {
                m_PlayerControl.GoFullScreen();
            }
            else
            {
                m_PlayerControl.GoWindowed();
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
    }
}
