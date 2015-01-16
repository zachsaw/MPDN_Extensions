using System;
using System.ComponentModel;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace Mpdn.PlayerExtensions.GitHub
{
    /// <summary>
    ///     ListBox which provides a fully integrated drag-and-drop functionality, which also works when multiple ListBox items
    ///     are selected (multiselect).
    ///     Items can be moved and copied between different DragDropListBoxes or can be moved inside of one DragDropListBox in
    ///     order to be reordered.
    ///     Drag-and-drop works also with or types of controls implementing IDragDropSource.
    /// </summary>
    /// <remarks>
    ///     Provides additional properties for the fine-tuning of the drag-and-drop behavior in the section
    ///     "Behavior (drag-and-drop)" of the properties window of the form designer.
    /// </remarks>
    public class DragDropListBox : ListBox, IDragDropSource
    {
        private Rectangle m_DragOriginBox = Rectangle.Empty;
        private bool m_RestoringSelection;
        private int[] m_SelectionSave = new int[0];

        #region IDragDropSource Members

        private string m_DragDropGroup = "";

        private bool m_IsDragDropCopySource = true;

        private bool m_IsDragDropMoveSource = true;

        /// <summary>
        ///     Drag-and-drop group to which the control belongs. Drag-and-drop is restricted to happen between controls having the
        ///     same DragDropGroup.
        /// </summary>
        /// <example>
        ///     Let's assume that we have a form with four DragDropListBoxes on it. Two of them contain cats and two of them
        ///     contain dogs.
        ///     We only want to be able to move cats between the cat lists and dogs between the dog lists (cats and dogs don't like
        ///     each other).
        ///     We can achieve this simply by setting the DragDropGroup property of the cat lists to "cats". In the dog lists we
        ///     can leave the
        ///     DragDropGroup empty or we can set it to "dogs" for instance. It just has to be different from the DragDropGroup in
        ///     the cat lists.
        ///     <code>catList1.DragDropGroup = "cats";   catList2.DragDropGroup = "cats";</code>
        /// </example>
        [Category("Behavior (drag-and-drop)"), DefaultValue(""),
         Description(
             "Drag-and-drop group to which the control belongs. Drag-and-drop is restricted to happen between controls having the same DragDropGroup."
             )]
        public string DragDropGroup
        {
            get { return m_DragDropGroup; }
            set { m_DragDropGroup = value; }
        }

        /// <summary>
        ///     Indicates whether the user can copy items from this control by draging them to another control.
        /// </summary>
        [Category("Behavior (drag-and-drop)"), DefaultValue(true),
         Description("Indicates whether the user can copy items from this control by draging them to another control.")]
        public bool IsDragDropCopySource
        {
            get { return m_IsDragDropCopySource; }
            set { m_IsDragDropCopySource = value; }
        }

        /// <summary>
        ///     Indicates whether the user can remove items from this control by draging them to another control.
        /// </summary>
        [Category("Behavior (drag-and-drop)"), DefaultValue(true),
         Description("Indicates whether the user can remove items from this control by draging them to another control."
             )]
        public bool IsDragDropMoveSource
        {
            get { return m_IsDragDropMoveSource; }
            set { m_IsDragDropMoveSource = value; }
        }

        /// <summary>
        ///     Returns the selected list items in a array.
        /// </summary>
        /// <returns>Array with the selected items.</returns>
        public object[] GetSelectedItems()
        {
            var items = new object[SelectedItems.Count];
            SelectedItems.CopyTo(items, 0);
            return items;
        }

        /// <summary>
        ///     Removes the selected items from the list and adjusts the item-index passed to this method,
        ///     so that this index points to the same item afterwards.
        /// </summary>
        /// <param name="itemIndexToAjust">Item-index to be adjusted.</param>
        public void RemoveSelectedItems(ref int itemIndexToAjust)
        {
            for (var i = SelectedIndices.Count - 1; i >= 0; i--)
            {
                var at = SelectedIndices[i];
                Items.RemoveAt(at);
                if (at < itemIndexToAjust)
                {
                    itemIndexToAjust--; // Adjust index pointing to stuff behind the delete position.
                }
            }
        }

        /// <summary>
        ///     Is called when a drag-and-drop operation is completed in order to raise the Dropped event.
        /// </summary>
        /// <param name="e">Event arguments which hold information on the completed operation.</param>
        /// <remarks>
        ///     Is called for the target as well as for the source.
        ///     The role a control plays (source or target) can be determined from e.Operation.
        /// </remarks>
        public virtual void OnDropped(DroppedEventArgs e)
        {
            var dropEvent = Dropped;
            if (dropEvent != null)
            {
                dropEvent(this, e);
            }
        }

        #endregion

        #region Other Public Properties

        private bool m_IsDragDropTarget = true;
        private bool m_AllowReorder = true;

        /// <summary>
        ///     Indicates whether the user can redorder the list by dragging items.
        /// </summary>
        [Category("Behavior (drag-and-drop)"), DefaultValue(true),
         Description("Indicates whether the user can redorder the list by dragging items.")]
        public bool AllowReorder
        {
            get { return m_AllowReorder; }
            set
            {
                m_AllowReorder = value;
                base.AllowDrop = m_IsDragDropTarget || m_AllowReorder;
            }
        }

        /// <summary>
        ///     Indicates whether the user can drop items from another control.
        /// </summary>
        [Category("Behavior (drag-and-drop)"), DefaultValue(true),
         Description("Indicates whether the user can drop items from another control.")]
        public bool IsDragDropTarget
        {
            get { return m_IsDragDropTarget; }
            set
            {
                m_IsDragDropTarget = value;
                base.AllowDrop = m_IsDragDropTarget || m_AllowReorder;
            }
        }

        /// <summary>
        ///     Occurs when a extended DragDropListBox drag-and-drop operation is completed.
        /// </summary>
        /// <remarks>
        ///     This event is raised for the target as well as for the source.
        ///     The role a control plays (source or target) can be determined from the Operation property of the DroppedEventArgs.
        /// </remarks>
        [Category("Drag Drop"),
         Description("Occurs when a extended DragDropListBox drag-and-drop operation is completed.")]
        public event EventHandler<DroppedEventArgs> Dropped;

        #endregion

        #region Overridden Event Methods

        protected override void OnDragDrop(DragEventArgs drgevent)
        {
            base.OnDragDrop(drgevent);

            // Retrieve the drag item data. 
            // Conditions have been testet in OnDragEnter and OnDragOver, so everything should be ok here.
            var src = drgevent.Data.GetData("IDragDropSource") as IDragDropSource;
            if (src == null)
                return;

            var srcItems = src.GetSelectedItems();

            // If the list box is sorted, we don't know where the items will be inserted
            // and we will have troubles selecting the inserted items. So let's disable sorting here.
            var sortedSave = Sorted;
            Sorted = false;

            // Insert at the currently hovered row.
            var row = DropIndex(drgevent.Y);
            var insertPoint = row;
            if (row >= Items.Count)
            {
                // Append items to the end.
                Items.AddRange(srcItems);
            }
            else
            {
                // Insert items before row.
                foreach (var item in srcItems)
                {
                    Items.Insert(row++, item);
                }
            }
            // Remove all the selected items from the source, if moving.
            DropOperation operation; // Remembers the operation for the event we'll raise.
            if (drgevent.Effect == DragDropEffects.Move)
            {
                var adjustedInsertPoint = insertPoint;
                src.RemoveSelectedItems(ref adjustedInsertPoint);
                if (src == this)
                {
                    // Items are being reordered.
                    insertPoint = adjustedInsertPoint;
                    operation = DropOperation.Reorder;
                }
                else
                {
                    operation = DropOperation.MoveToHere;
                }
            }
            else
            {
                operation = DropOperation.CopyToHere;
            }

            // Adjust the selected items in the target.
            ClearSelected();
            if (SelectionMode == SelectionMode.One)
            {
                // Select the first item inserted.
                SelectedIndex = insertPoint;
            }
            else if (SelectionMode != SelectionMode.None)
            {
                // Select the inserted items.
                for (var i = insertPoint; i < insertPoint + srcItems.Length; i++)
                {
                    SetSelected(i, true);
                }
            }

            // Now that we've selected the inserted items, restore the "Sorted" property.
            Sorted = sortedSave;

            // Notify the target (this control).
            var e = new DroppedEventArgs
            {
                Operation = operation,
                Source = src,
                Target = this,
                DroppedItems = srcItems
            };
            OnDropped(e);

            // Notify the source (the other control).
            if (operation != DropOperation.Reorder)
            {
                e = new DroppedEventArgs
                {
                    Operation =
                        operation == DropOperation.MoveToHere ? DropOperation.MoveFromHere : DropOperation.CopyFromHere,
                    Source = src,
                    Target = this,
                    DroppedItems = srcItems
                };
                src.OnDropped(e);
            }
        }

        protected override void OnDragOver(DragEventArgs drgevent)
        {
            base.OnDragOver(drgevent);

            drgevent.Effect = GetDragDropEffect(drgevent);
            if (drgevent.Effect == DragDropEffects.None)
            {
            }

            // Everything is fine, give a visual cue
//			int dropIndex = DropIndex(drgevent.Y);
//			if (dropIndex != _visualCue.Index) {
//				_visualCue.Clear();
//				_visualCue.Draw(dropIndex);
//			}
        }

        protected override void OnDragEnter(DragEventArgs drgevent)
        {
            base.OnDragEnter(drgevent);
            drgevent.Effect = GetDragDropEffect(drgevent);
        }

//		protected override void OnDragLeave(EventArgs e)
//		{
//			base.OnDragLeave(e);
//			_visualCue.Clear();
//		}

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);

            var clickedItemIndex = IndexFromPoint(e.Location);
            if (clickedItemIndex >= 0 && MouseButtons == MouseButtons.Left &&
                (m_IsDragDropCopySource || m_IsDragDropMoveSource || m_AllowReorder) &&
                (GetSelected(clickedItemIndex) || ModifierKeys == Keys.Shift))
            {
                RestoreSelection(clickedItemIndex);

                // Remember start position of possible drag operation.
                var dragSize = SystemInformation.DragSize;
                    // Size that the mouse must move before a drag operation starts.
                m_DragOriginBox = new Rectangle(new Point(e.X - (dragSize.Width/2), e.Y - (dragSize.Height/2)), dragSize);
            }
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            base.OnMouseUp(e);

            if (m_DragOriginBox != Rectangle.Empty)
            {
                m_DragOriginBox = Rectangle.Empty; // Reset drag drop.

                var clickedItemIndex = IndexFromPoint(e.Location);
                if (clickedItemIndex >= 0 && e.Button == MouseButtons.Left &&
                    !ModifierKeys.HasFlag(Keys.Shift) && !ModifierKeys.HasFlag(Keys.Control))
                {
                    SelectedIndex = -1;
                    SelectedIndex = clickedItemIndex;
                }
            }
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (m_DragOriginBox != Rectangle.Empty && !m_DragOriginBox.Contains(e.X, e.Y))
            {
                // Initiate drag-and-drop
                DoDragDrop(new DataObject("IDragDropSource", this), DragDropEffects.All);
                m_DragOriginBox = Rectangle.Empty;
            }
        }

        protected override void OnSelectedIndexChanged(EventArgs e)
        {
            base.OnSelectedIndexChanged(e);
            SaveSelection();
        }

        #endregion

        #region private Helper methods

        /// <summary>
        ///     Restores the lost selection from the array selectionSave.
        /// </summary>
        /// <remarks>
        ///     If the user clicks into the selection, he might want to drag the selected items,
        ///     but the mouse click destroys the current selection, if more than one item is selected.
        /// </remarks>
        private void RestoreSelection(int clickedItemIndex)
        {
            // Restore the selection, unless modifier keys are pressed, which indicates that the user is currently editing the selection.
            // The item the user clickes at must have been selected before the click (_selectionSave stores the state before the click).
            if (SelectionMode == SelectionMode.MultiExtended && ModifierKeys == Keys.None &&
                Array.IndexOf(m_SelectionSave, clickedItemIndex) >= 0)
            {
                m_RestoringSelection = true;
                    // Disable saving the selection while it is restored. (SetSelected raises the SelectedIndexChanged
                // event, where we call SaveSelection.)
                foreach (var i in m_SelectionSave)
                {
                    SetSelected(i, true);
                }
                // Select the item that was clicked, in order to make it the current item. This also fixes a bug, where the listbox
                // selects too many items, if the list is clicked after items have been selected programmatically.
                SetSelected(clickedItemIndex, true);
                m_RestoringSelection = false;
            }
        }

        /// <summary>
        ///     Saves the current selection to the array selectionSave.
        /// </summary>
        private void SaveSelection()
        {
            if (!m_RestoringSelection && SelectionMode == SelectionMode.MultiExtended)
            {
                var sel = SelectedIndices;
                if (m_SelectionSave.Length != sel.Count)
                {
                    m_SelectionSave = new int[sel.Count];
                }
                SelectedIndices.CopyTo(m_SelectionSave, 0);
            }
        }

        /// <summary>
        ///     Gets the index of the item before which items are being dropped. The index is calculated from the vertical position
        ///     of the mouse.
        ///     If the drop position lies after the last item in the list, then the index of the last item + 1 (which is equal to
        ///     Item.Count) is returned instead.
        /// </summary>
        /// <param name="yScreen">y-coordinate of the mouse expressed in screen coordinates.</param>
        /// <returns>Index of an item in the list or Items.Count</returns>
        private int DropIndex(int yScreen)
        {
            // The DragEventArgs gives us screen coordinates. Convert the screen coordinates to client coordinates.
            var y = PointToClient(new Point(0, yScreen)).Y;

            // Make sure we are inside of the client rectangle.
            // If we are on the border of the ListBox, then IndexFromPoint does not return a match.
            if (y < 0)
            {
                y = 0;
            }
            else if (y > ClientRectangle.Bottom - 1)
            {
                y = ClientRectangle.Bottom - 1;
            }

            var index = IndexFromPoint(0, y); // The x-coordinate doesn't make any difference.
            if (index == NoMatches)
            {
                // Not hovering over an item
                return Items.Count; // Append to the end of the list.
            }

            // If hovering below the middle of the item, then insert after the item.
            var rect = GetItemRectangle(index);
            if (y > rect.Top + rect.Height/2)
            {
                index++;
            }

            var lastFullyVisibleItemIndex = TopIndex + ClientRectangle.Height/ItemHeight;
            if (index > lastFullyVisibleItemIndex)
            {
                // Do not insert after the last fully visible item
                return lastFullyVisibleItemIndex;
            }
            return index;
        }

        /// <summary>
        ///     Determines the drag-and-drop operation which is beeing performed, which can be either None, Move or Copy.
        /// </summary>
        /// <param name="drgevent">DragEventArgs.</param>
        /// <returns>The current drag-and-drop operation.</returns>
        private DragDropEffects GetDragDropEffect(DragEventArgs drgevent)
        {
            const int ctrlKeyPlusLeftMouseButton = 9; // KeyState.

            var effect = drgevent.Effect;

            // Retrieve the source control of the drag-and-drop operation.
            var src = drgevent.Data.GetData("IDragDropSource") as IDragDropSource;

            if (src != null && m_DragDropGroup == src.DragDropGroup)
            {
                // The stuff being draged is compatible.
                if (src == this)
                {
                    // Drag-and-drop happens within this control.
                    if (m_AllowReorder && !Sorted)
                    {
                        effect = DragDropEffects.Move;
                    }
                }
                else if (m_IsDragDropTarget)
                {
                    // If only Copy is allowed then copy. If Copy and Move are allowed, then Move, unless the Ctrl-key is pressed.
                    if (src.IsDragDropCopySource &&
                        (!src.IsDragDropMoveSource || drgevent.KeyState == ctrlKeyPlusLeftMouseButton))
                    {
                        effect = DragDropEffects.Copy;
                    }
                    else if (src.IsDragDropMoveSource)
                    {
                        effect = DragDropEffects.Move;
                    }
                }
            }
            return effect;
        }

        #endregion
    }

    public enum DropOperation
    {
        Reorder,
        MoveToHere,
        CopyToHere,
        MoveFromHere,
        CopyFromHere
    }

    public class DroppedEventArgs : EventArgs
    {
        public DropOperation Operation { get; set; }
        public IDragDropSource Source { get; set; }
        public IDragDropSource Target { get; set; }
        public object[] DroppedItems { get; set; }
    }

    public interface IDragDropSource
    {
        string DragDropGroup { get; }
        bool IsDragDropCopySource { get; }
        bool IsDragDropMoveSource { get; }
        object[] GetSelectedItems();
        void RemoveSelectedItems(ref int rowIndexToAjust);
        void OnDropped(DroppedEventArgs e);
    }

    /// <summary>
    ///     Win 32 API stuff used in this Project.
    /// </summary>
    internal static class Win32
    {
        public const int R2_NOT = 6; // Inverted drawing mode

        [DllImport("gdi32.dll", EntryPoint = "SetROP2", CallingConvention = CallingConvention.StdCall)]
        public static extern int SetROP2(IntPtr hdc, int fnDrawMode);

        [DllImport("user32.dll", EntryPoint = "GetDC", CallingConvention = CallingConvention.StdCall)]
        public static extern IntPtr GetDC(IntPtr hWnd);

        [DllImport("user32.dll", EntryPoint = "ReleaseDC", CallingConvention = CallingConvention.StdCall)]
        public static extern IntPtr ReleaseDC(IntPtr hWnd, IntPtr hDc);

        [DllImport("gdi32.dll", EntryPoint = "MoveToEx", CallingConvention = CallingConvention.StdCall)]
        public static extern bool MoveToEx(IntPtr hdc, int x, int y, IntPtr lpPoint);

        [DllImport("gdi32.dll", EntryPoint = "LineTo", CallingConvention = CallingConvention.StdCall)]
        public static extern bool LineTo(IntPtr hdc, int x, int y);
    }
}