# -*- coding: utf-8 -*-

"""The user interface for our app"""

import os,sys

# Import Qt modules
from PyQt4 import QtCore,QtGui

# Import the compiled UI module
from windowUi import Ui_MainWindow

# Import our backend
import todo

# Create a class for our main window
class Main(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        
        # This is always the same
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)

        # Let's do something interesting: load the database contents 
        # into our task list widget
        for task in todo.Task.query().all():
            tags=','.join([t.name for t in task.tags])
            item=QtGui.QTreeWidgetItem([task.text,str(task.date),tags])
            item.task=task
            if task.done:
                item.setCheckState(0,QtCore.Qt.Checked)
            else:
                item.setCheckState(0,QtCore.Qt.Unchecked)
            self.ui.list.addTopLevelItem(item)

    def on_list_itemChanged(self,item,column):
        if item.checkState(0):
            item.task.done=True
        else:
            item.task.done=False
        todo.saveData()


def main():
    # Init the database before doing anything else
    todo.initDB()
    
    # Again, this is boilerplate, it's going to be the same on 
    # almost every app you write
    app = QtGui.QApplication(sys.argv)
    window=Main()
    window.show()
    # It's exec_ because exec is a reserved word in Python
    sys.exit(app.exec_())
    

if __name__ == "__main__":
    main()
    
