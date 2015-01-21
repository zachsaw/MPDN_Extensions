using System;
using System.Collections.Generic;
using System.IO;

namespace Mpdn.PlayerExtensions
{
    public class RemoteControlAuthHandler
    {
        #region Variables

        private const string Folder = "RemoteDetails";
        private const string FilePath = "accessGUID.conf";
        private readonly Guid _nullGuid = Guid.Parse("{00000000-0000-0000-0000-000000000000}");
        private readonly string _fullPath;
        private readonly List<Guid> _authedClients = new List<Guid>();
        #endregion

        public RemoteControlAuthHandler()
        {
            _fullPath = Path.Combine(Folder, FilePath);
            ReadAuthedClients();
        }

        private void ReadAuthedClients()
        {
            if(Directory.Exists(Folder))
            {
                var file = File.Open(_fullPath, FileMode.Open, FileAccess.Read);
                StreamReader reader = new StreamReader(file);
                bool readAgain;
                do
                {
                    var line = reader.ReadLine();
                    Guid tmpGuid;
                    Guid.TryParse(line, out tmpGuid);
                    if (tmpGuid != _nullGuid)
                    {
                        _authedClients.Add(tmpGuid);
                    }
                    line = reader.ReadLine();
                    readAgain = !string.IsNullOrEmpty(line);
                }
                while(readAgain);
                reader.Close();
            }
        }

        public bool IsGuidAuthed(string clientGuid)
        {
            bool isAuthed = false;
            Guid tmpGuid;
            Guid.TryParse(clientGuid, out tmpGuid);
            if(_authedClients.Contains(tmpGuid))
            {
                isAuthed = true;
            }
            return isAuthed;
        }

        

        public void AddAuthedClient(string clientGuid)
        {
            FileStream myFile = null;
            if(!Directory.Exists(Folder))
            {
                Directory.CreateDirectory(Folder);
            }
            if(!File.Exists(_fullPath))
            {
                myFile = File.Create(_fullPath);
            }

            try
            {
                if(myFile == null)
                    myFile = File.Open(_fullPath, FileMode.Append, FileAccess.Write);
                StreamWriter writer = new StreamWriter(myFile);
                writer.WriteLine(clientGuid);
                writer.Flush();
                writer.Close();
                Guid tmpGuid;
                Guid.TryParse(clientGuid, out tmpGuid);
                _authedClients.Add(tmpGuid);
            }
            catch(Exception)
            {

            }
        }
    }
}
