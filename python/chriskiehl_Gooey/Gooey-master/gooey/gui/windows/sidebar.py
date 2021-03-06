import wx
from gooey.gui.pubsub import pub
from gooey.gui import events

from gooey.gui.util import wx_util


class Sidebar(wx.Panel):

  def __init__(self, parent, *args, **kwargs):
    super(Sidebar, self).__init__(parent, *args, **kwargs)
    self.SetDoubleBuffered(True)

    self._parent = parent

    self._do_layout()

  def _do_layout(self):
    self.SetDoubleBuffered(True)
    self.SetBackgroundColour('#f2f2f2')
    self.SetSize((180, 0))
    self.SetMinSize((180, 0))

    STD_LAYOUT = (0, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)

    container = wx.BoxSizer(wx.VERTICAL)
    container.AddSpacer(15)
    container.Add(wx_util.h1(self, 'Actions'), *STD_LAYOUT)
    container.AddSpacer(5)
    self.listbox = wx.ListBox(self, -1)
    container.Add(self.listbox, 1, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)
    container.AddSpacer(20)
    self.SetSizer(container)

    self.Bind(wx.EVT_LISTBOX, self.selection_change, self.listbox)

  def set_list_contents(self, contents):
    self.listbox.AppendItems(contents)
    self.listbox.SetSelection(0)

  def selection_change(self, evt):
    pub.send_message(
      events.LIST_BOX,
      selection=self.listbox.GetItems()[self.listbox.GetSelection()])
    evt.Skip()
