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
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using Mpdn.Extensions.Framework;
using Mpdn.Extensions.Framework.Config;

namespace Mpdn.Extensions.AudioScripts
{
    namespace Mpdn
    {
        public partial class AudioChainConfigDialog : AudioChainConfigDialogBase
        {
            private const string SELECTED_INDICATOR_STR = "➔";

            private int m_SelectedIndex = -1;
            public int SelectedIndex
            {
                get { return m_SelectedIndex; }
                set
                {
                    if (0 <= value && value < listViewChain.Items.Count)
                    {
                        m_SelectedIndex = value;
                        foreach (ListViewItem item in listViewChain.SelectedItems)
                            item.Selected = false;
                        listViewChain.Items[value].Selected = true;
                    }
                    else
                        m_SelectedIndex = -1;
                }
            }

            public AudioChainConfigDialog()
            {
                InitializeComponent();
            }

            protected override void LoadSettings()
            {
                AddScripts(Settings.AudioScripts, false);
            }

            protected override void SaveSettings()
            {
                Settings.AudioScripts.Clear();
                foreach (ListViewItem i in listViewChain.Items)
                {
                    Settings.AudioScripts.Add((IExtensionUi) i.Tag);
                }
            }

            private void FormLoad(object sender, EventArgs e)
            {
                var audioScripts = Extension.AudioScripts
                    .Where(script => script as IAudioChain != null)
                    .Select(x => x.CreateNew())
                    .Concat(new[] {ExtensionUi.Identity})
                    .OrderBy(x => x.Descriptor.Name);

                foreach (var script in audioScripts)
                {
                    var item = listViewAvail.Items.Add(string.Empty);
                    item.SubItems.Add(script.Descriptor.Name);
                    item.SubItems.Add(script.Descriptor.Description);
                    item.Tag = script;
                }

                listViewChain.SelectedIndices.Clear();

                ResizeLists();
                UpdateButtons();
            }

            private void ResizeLists()
            {
                listViewChain.BeginUpdate();
                {
                    listViewChain.AutoResizeColumns(ColumnHeaderAutoResizeStyle.ColumnContent);
                    listViewChain.AutoResizeColumns(ColumnHeaderAutoResizeStyle.HeaderSize);
                }
                listViewChain.EndUpdate();

                listViewAvail.BeginUpdate();
                {
                    listViewAvail.AutoResizeColumns(ColumnHeaderAutoResizeStyle.ColumnContent);
                    listViewAvail.AutoResizeColumns(ColumnHeaderAutoResizeStyle.HeaderSize);
                }
                listViewAvail.EndUpdate();
            }

            private void SplitterMoved(object sender, SplitterEventArgs e)
            {
                ResizeLists();
            }

            private void RenderChainListSizeChanged(object sender, EventArgs e)
            {
                ResizeLists();
            }

            private enum MoveDirection
            {
                Up = -1,
                Down = 1
            };

            private static void MoveListViewItems(ListView listView, MoveDirection direction)
            {
                var valid = listView.SelectedItems.Count > 0 &&
                            ((direction == MoveDirection.Down &&
                              (listView.SelectedItems[listView.SelectedItems.Count - 1].Index < listView.Items.Count - 1))
                             || (direction == MoveDirection.Up && (listView.SelectedItems[0].Index > 0)));

                if (!valid)
                    return;

                var start = true;
                var firstIdx = 0;
                var items = new List<ListViewItem>();

                foreach (ListViewItem i in listView.SelectedItems)
                {
                    if (start)
                    {
                        firstIdx = i.Index;
                        start = false;
                    }
                    items.Add(i);
                }

                listView.BeginUpdate();

                foreach (ListViewItem i in listView.SelectedItems)
                {
                    i.Remove();
                }

                if (direction == MoveDirection.Up)
                {
                    var insertTo = firstIdx - 1;
                    foreach (var i in items)
                    {
                        i.Selected = true;
                        listView.Items.Insert(insertTo, i);
                        insertTo++;
                    }
                }
                else
                {
                    var insertTo = firstIdx + 1;
                    foreach (var i in items)
                    {
                        i.Selected = true;
                        listView.Items.Insert(insertTo, i);
                        insertTo++;
                    }
                }

                listView.EndUpdate();
                listView.Focus();
            }

            private void AddScripts(IEnumerable<IExtensionUi> audioScripts, bool createNew, int index = -1)
            {
                listViewChain.SelectedItems.Clear();
                foreach (var script in audioScripts)
                {
                    var item = index < 0
                        ? listViewChain.Items.Add(string.Empty)
                        : listViewChain.Items.Insert(index++, string.Empty);

                    item.SubItems.Add(script.Descriptor.Name);
                    item.SubItems.Add(script.Descriptor.Description);
                    item.Tag = createNew ? script.CreateNew() : script;

                    listViewChain.SelectedIndices.Add(item.Index);
                }
                ResizeLists();
                UpdateButtons();
            }

            private void AddScript(IExtensionUi audioScript, int index = -1)
            {
                AddScripts(new[] {audioScript}, true, index);
            }

            private void RemoveItem(ListViewItem selectedItem)
            {
                var index = selectedItem.Index;
                selectedItem.Remove();
                if (index < listViewChain.Items.Count)
                {
                    listViewChain.Items[index].Selected = true;
                }
                else if (listViewChain.Items.Count > 0)
                {
                    listViewChain.Items[listViewChain.Items.Count - 1].Selected = true;
                }

                ResizeLists();
                UpdateButtons();
            }

            private void ConfigureItem(ListViewItem item)
            {
                var ui = (IExtensionUi) item.Tag;
                if (ui.HasConfigDialog() && ui.ShowConfigDialog(this))
                    UpdateItemText(item);
            }

            private void UpdateItemText(ListViewItem item)
            {
                var preset = (IExtensionUi) item.Tag;

                while (item.SubItems.Count < 3)
                    item.SubItems.Add(string.Empty);
                item.SubItems[1].Text = preset.Descriptor.Name;
                item.SubItems[2].Text = preset.Descriptor.Description;

                ResizeLists();
            }

            private void UpdateButtons()
            {
                buttonAdd.Enabled = listViewAvail.SelectedItems.Count > 0;
                buttonMinus.Enabled = listViewChain.SelectedItems.Count > 0;
                buttonClear.Enabled = listViewChain.Items.Count > 0;
                buttonUp.Enabled = listViewChain.SelectedItems.Count > 0 && listViewChain.SelectedItems[0].Index > 0;
                buttonDown.Enabled = listViewChain.SelectedItems.Count > 0 &&
                                     listViewChain.SelectedItems[listViewChain.SelectedItems.Count - 1].Index <
                                     listViewChain.Items.Count - 1;
                buttonConfigure.Enabled = buttonMinus.Enabled;

                menuAdd.Enabled = buttonAdd.Enabled;
                menuRemove.Enabled = buttonMinus.Enabled;
                menuClear.Enabled = buttonClear.Enabled;

                if (listViewChain.SelectedItems.Count > 0)
                {
                    var item = listViewChain.SelectedItems[0];
                    var ui = (IExtensionUi) item.Tag;
                    buttonConfigure.Enabled = ui.HasConfigDialog();
                }

                menuConfigure.Enabled = buttonConfigure.Enabled;
            }

            private void ListViewSelectedIndexChanged(object sender, EventArgs e)
            {
                foreach (ListViewItem i in listViewAvail.Items)
                {
                    i.Text = string.Empty;
                }

                if (listViewAvail.SelectedItems.Count > 0)
                {
                    var item = listViewAvail.SelectedItems[0];
                    var script = (IExtensionUi) item.Tag;

                    item.Text = SELECTED_INDICATOR_STR;
                    labelCopyright.Text = script == null ? string.Empty : script.Descriptor.Copyright;
                }

                UpdateButtons();
            }

            private void ListViewChainSelectedIndexChanged(object sender, EventArgs e)
            {
                foreach (ListViewItem i in listViewChain.Items)
                {
                    i.Text = string.Empty;
                }

                if (listViewChain.SelectedItems.Count > 0)
                {
                    var item = listViewChain.SelectedItems[0];
                    item.Text = SELECTED_INDICATOR_STR;
                    m_SelectedIndex = item.Index;
                }
                else
                {
                    m_SelectedIndex = -1;
                }

                UpdateButtons();
            }

            private void SelectAll(object sender, EventArgs e)
            {
                foreach (ListViewItem item in listViewChain.Items)
                    item.Selected = true;
            }

            #region Buttons

            private void ButtonConfigureClicked(object sender, EventArgs e)
            {
                if (listViewChain.SelectedItems.Count <= 0)
                    return;

                var item = listViewChain.SelectedItems[0];
                ConfigureItem(item);
            }

            private void ButtonAddClicked(object sender, EventArgs e)
            {
                foreach (ListViewItem item in listViewAvail.SelectedItems)
                    AddScript((IExtensionUi) item.Tag);
            }

            private void ButtonMinusClicked(object sender, EventArgs e)
            {
                foreach (ListViewItem item in listViewChain.SelectedItems)
                    RemoveItem(item);
            }

            private void ButtonClearClicked(object sender, EventArgs e)
            {
                while (listViewChain.Items.Count > 0)
                {
                    RemoveItem(listViewChain.Items[0]);
                }
                UpdateButtons();
            }

            private void ButtonUpClicked(object sender, EventArgs e)
            {
                MoveListViewItems(listViewChain, MoveDirection.Up);
                UpdateButtons();
            }

            private void ButtonDownClicked(object sender, EventArgs e)
            {
                MoveListViewItems(listViewChain, MoveDirection.Down);
                UpdateButtons();
            }

            #endregion

            #region Drap / Drop

            private void ItemCopyDrag(object sender, ItemDragEventArgs e)
            {
                DoDragDrop(((ListView) sender).SelectedItems, DragDropEffects.Copy);
            }

            private void ItemMoveDrag(object sender, ItemDragEventArgs e)
            {
                DoDragDrop(((ListView) sender).SelectedItems, DragDropEffects.Move);
            }

            private void ListDragEnter(object sender, DragEventArgs e)
            {
                e.Effect = e.AllowedEffect;
            }

            private void ListDragDrop(object sender, DragEventArgs e)
            {
                Point cp = listViewChain.PointToClient(new Point(e.X, e.Y));
                ListViewItem dragToItem = listViewChain.GetItemAt(cp.X, cp.Y);
                bool after = (dragToItem != null) && listViewChain.GetItemRect(dragToItem.Index).Bottom - 8 <= cp.Y;
                var draggedItems =
                    e.Data.GetData(typeof (ListView.SelectedListViewItemCollection)) as
                        ListView.SelectedListViewItemCollection;
                if (draggedItems == null || draggedItems.Count == 0)
                    return;

                if (e.Effect == DragDropEffects.Copy)
                {
                    var items = draggedItems.Cast<ListViewItem>();
                    var index = dragToItem == null ? listViewChain.Items.Count : dragToItem.Index + (after ? 1 : 0);

                    AddScripts(items.Select(item => (item.Tag as IExtensionUi)), true, index);
                }
                else if (e.Effect == DragDropEffects.Move)
                {
                    if (draggedItems.Contains(dragToItem))
                        return;

                    var items = new List<ListViewItem>();
                    foreach (ListViewItem item in draggedItems.Cast<ListViewItem>())
                    {
                        item.Remove();
                        items.Add(item);
                    }

                    var index = dragToItem == null ? listViewChain.Items.Count : dragToItem.Index + (after ? 1 : 0);
                    foreach (ListViewItem item in items)
                    {
                        listViewChain.Items.Insert(index, item);
                        index++;
                    }
                }

                listViewChain.Focus();
            }

            private void ListDragDropRemove(object sender, DragEventArgs e)
            {
                var draggedItems =
                    e.Data.GetData(typeof (ListView.SelectedListViewItemCollection)) as
                        ListView.SelectedListViewItemCollection;
                if (draggedItems == null)
                    return;

                if (e.Effect != DragDropEffects.Move) return;

                foreach (var item in draggedItems.Cast<ListViewItem>())
                {
                    RemoveItem(item);
                }
            }

            #endregion

            #region Copy / Pasting

            private void MenuChainCopyClicked(object sender, EventArgs e)
            {
                List<IExtensionUi> items = listViewChain.SelectedItems
                    .Cast<ListViewItem>()
                    .Select(item => (IExtensionUi) item.Tag)
                    .ToList();

                var text = ConfigHelper.SaveToString(items);
                Clipboard.SetText(text);
            }

            private void MenuChainCutClicked(object sender, EventArgs e)
            {
                MenuChainCopyClicked(sender, e);
                ButtonMinusClicked(sender, e);
            }

            private void MenuChainPasteClicked(object sender, EventArgs e)
            {
                if (!Clipboard.ContainsText()) return;

                var text = Clipboard.GetText();
                var items = ConfigHelper.LoadFromString<List<IExtensionUi>>(text);
                if (items != null)
                {
                    AddScripts(items, false, SelectedIndex + 1);
                }
            }

            #endregion
        }

        public class AudioChainConfigDialogBase : ScriptConfigDialog<AudioChainSettings>
        {
        }
    }
}