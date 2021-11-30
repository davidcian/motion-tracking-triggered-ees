import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui

import pyqtgraph as pg

class MyWidget(QtWidgets.QWidget):
  def __init__(self):
    super().__init__()

    self.graphWidget = pg.PlotWidget()

    hour = [1,2,3,4,5,6,7,8,9,10]
    temperature = [30,32,34,32,33,31,29,32,35,45]

    self.graphWidget.plot(hour, temperature)

    self.layout = QtWidgets.QVBoxLayout(self)
    self.layout.addWidget(self.graphWidget)

if __name__ == '__main__':
  app = QtWidgets.QApplication([])

  widget = MyWidget()
  widget.resize(800, 600)
  widget.show()

  sys.exit(app.exec())