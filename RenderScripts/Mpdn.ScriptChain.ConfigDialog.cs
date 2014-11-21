using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using System.Collections.Generic;

namespace Mpdn.RenderScript
{
    namespace Mpdn.ScriptChain
    {
        public partial class ScriptChainDialog : ScriptChainDialogBase
        {
            #region Listing RenderChainScripts

            private class ScriptAbstract
            {
                public Type ScriptType { get; private set; }
                public IRenderChainUi ScriptUi { get; private set; }

                public ScriptAbstract(Type scriptType)
                {
                    ScriptType = scriptType;
                    ScriptUi = (IRenderChainUi)ScriptType.GetConstructor(Type.EmptyTypes).Invoke(new object[0]);
                }
       
                public override String ToString()
                {
                    return ScriptUi.Descriptor.Name;
                }

                public ChainAbstract GetChain()
                {
                    return new ChainAbstract( ScriptType );
                }
            }

            private class ChainAbstract
            {
                public IRenderChain Chain;
                public Type UiType;
                public IRenderChainUi ChainUi;

                public ChainAbstract() { }

                public ChainAbstract(ChainUiPair Pair)
                {
                    Chain = Pair.Chain;
                    UiType = Pair.UiType;
                    ChainUi = Pair.ChainUi;
                }

                public ChainAbstract(Type ScriptType)
                {
                    UiType = ScriptType;
                    ChainUi = (IRenderChainUi)UiType.GetConstructor(Type.EmptyTypes).Invoke(new object[0]);
                    ChainUi.Initialize();
                    Chain = ChainUi.GetChain();
                }

                public override String ToString()
                {
                    return ChainUi.Descriptor.Name;
                }

                public ChainUiPair ToPair()
                {
                    return new ChainUiPair
                    {
                        Chain = Chain,
                        UiType = UiType,
                        ChainUi = ChainUi
                    };
                }
            }

            private ScriptAbstract[] GetTypes()
            {
                var q = from t in Assembly.GetExecutingAssembly().GetTypes()
                        where t.IsClass
                            && typeof(IRenderChainUi).IsAssignableFrom(t)
                            && t.GetConstructor(Type.EmptyTypes) != null
                        select new ScriptAbstract(t);
                return q.ToArray();
            }

            #endregion

            public ScriptChainDialog()
            {
                InitializeComponent();

                listBox.Items.AddRange(GetTypes());
                UpdateButtons();
            }

            protected override void LoadSettings()
            {
                scriptListBox.Items.AddRange(Settings.ScriptList.Select(x => new ChainAbstract(x)).ToArray());
            }

            protected override void SaveSettings()
            {
                Settings.ScriptList = scriptListBox.Items.Cast<ChainAbstract>().Select(x => x.ToPair()).ToList();
            }

            private void UpdateButtons()
            {
                buttonAdd.Enabled = listBox.SelectedIndex >= 0;

                var index = scriptListBox.SelectedIndex;
                var count = scriptListBox.Items.Count;
                var item = (ChainAbstract)scriptListBox.SelectedItem;

                buttonConfigure.Enabled = item != null && item.ChainUi.Descriptor.HasConfigDialog;
                buttonRemove.Enabled = index >= 0;
                buttonUp.Enabled = index > 0;
                buttonDown.Enabled = index >= 0 && index < count - 1;
                buttonClear.Enabled = count > 0;
            }

            private void ButtonAddClick(object sender, EventArgs e)
            {
                var selection = (ScriptAbstract)listBox.SelectedItem;

                if (selection != null)
                    scriptListBox.Items.Add(selection.GetChain());
            }

            private void SelectionChange(object sender, EventArgs e)
            {
                UpdateButtons();
            }

            private void ButtonRemoveClick(object sender, EventArgs e)
            {
                var index = scriptListBox.SelectedIndex;
                scriptListBox.Items.RemoveAt(index);
                scriptListBox.SelectedIndex = index < scriptListBox.Items.Count ? index : scriptListBox.Items.Count - 1;
            }

            private void ButtonClearClick(object sender, EventArgs e)
            {
                scriptListBox.Items.Clear();
            }

            private void ButtonConfigureClick(object sender, EventArgs e)
            {
                var item = (ChainAbstract)scriptListBox.SelectedItem;
                item.ChainUi.ShowConfigDialog(Owner);
            }
        }

        public class ScriptChainDialogBase : ScriptConfigDialog<ScriptChain>
        {
        }
    }
}