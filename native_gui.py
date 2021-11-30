import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui

import pyqtgraph as pg

class MyWidget(QtWidgets.QWidget):
  def __init__(self):
    super().__init__()

    self.graphWidget = pg.PlotWidget()

    self.frame_indices = list(range(100))
    self.x_val = [random.randint(0, 100) for _ in range(100)]
    self.y_val = [random.randint(0, 100) for _ in range(100)]
    self.z_val = [random.randint(0, 100) for _ in range(100)]

    self.graphWidget.setTitle("Real vs. filtered 3D coordinates")
    self.graphWidget.setLabel('left', "Coordinate value")
    self.graphWidget.setLabel('bottom', "Frame index")
    self.graphWidget.setBackground('w')
    self.graphWidget.addLegend()

    pen1 = pg.mkPen(color='r', width=2)
    pen2 = pg.mkPen(color='g', width=2)
    pen3 = pg.mkPen(color='b', width=2)

    x_line_ref = self.graphWidget.plot(self.frame_indices, self.x_val, name='X', pen=pen1)
    y_line_ref = self.graphWidget.plot(self.frame_indices, self.y_val, name='Y', pen=pen2)
    z_line_ref = self.graphWidget.plot(self.frame_indices, self.z_val, name='Z', pen=pen3)

    self.layout = QtWidgets.QVBoxLayout(self)
    self.layout.addWidget(self.graphWidget)

    self.timer = QtCore.QTimer(self)
    y_seqs = [self.x_val, self.y_val, self.z_val]
    refs = [x_line_ref, y_line_ref, z_line_ref]
    self.connect(self.timer, QtCore.SIGNAL("timeout()"), lambda: self.update_plot_data(y_seqs, refs))
    self.timer.start(50)
  
  def update_plot_data(self, y_seqs, refs):
    self.frame_indices = self.frame_indices[1:]
    self.frame_indices.append(self.frame_indices[-1] + 1)

    for i, ref in enumerate(refs):
      y_seqs[i] = y_seqs[i][1:]
      y_seqs[i].append(random.randint(0, 100))

      ref.setData(self.frame_indices, y_seqs[i])

if __name__ == '__main__':
  app = QtWidgets.QApplication([])

  widget = MyWidget()
  widget.resize(800, 600)
  widget.show()

  sys.exit(app.exec())