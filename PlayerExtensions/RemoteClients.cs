using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using ACMPlugin;
using System.Net;

namespace ACMPlugin
{
    public partial class RemoteClients : Form
    {
        #region Variables
        public ACMPlug mainRemote;
        #endregion

        #region Delegates
        private delegate void UpdateDGRowDelegate(string[] row);
        private delegate void ClearGridDelegate();
        #endregion

        #region Constructor
        public RemoteClients(ACMPlug control)
        {
            InitializeComponent();
            mainRemote = control;
            this.Load += RemoteClients_Load;
        }
        #endregion

        #region Internal Methods
        internal void ForceUpdate()
        {
            PopulateGrid();
        }
        #endregion

        void RemoteClients_Load(object sender, EventArgs e)
        {
            PopulateGrid();
        }

        private void PopulateGrid()
        {
            dgMainGrid.Invoke(new ClearGridDelegate(ClearGrid));
            foreach(var item in mainRemote.getClients)
            {
                try
                {
                    IPEndPoint remoteIpEndPoint = item.Value.RemoteEndPoint as IPEndPoint;
                    string[] tmpRow = { item.Key.ToString(), remoteIpEndPoint.Address + ":" + remoteIpEndPoint.Port };
                    dgMainGrid.Invoke(new UpdateDGRowDelegate(addRow), new object[]{tmpRow});
                }
                catch(Exception ex)
                {
                    MessageBox.Show("Error " + ex.ToString());
                }
            } 

        }

        private void ClearGrid()
        {
            dgMainGrid.Rows.Clear();
        }

        private void addRow(string[] row)
        {
            dgMainGrid.Rows.Add(row);
            dgMainGrid.Refresh();
        }

        private void btnClose_Click(object sender, EventArgs e)
        {
            this.Close();
        }
    }
}
