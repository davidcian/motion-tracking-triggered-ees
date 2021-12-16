from PySide6 import QtCore, QtWidgets, QtGui

class ImplantWidget(QtWidgets.QWidget):
  def __init__(self):
    super().__init__()

    stim_configs = {"config1": [0, 5]}

    selected_stim_config = "config1"

    electrode_positions = [(23, 12), (6, 40), (40, 40), 
      (23, 67), (6, 96), (40, 96),
      (23, 123), (6, 153), (40, 153),
      (23, 180), (6, 209), (40, 209),
      (23, 235), (6, 264), (40, 264),
      (23, 292)]
    
    self.scene = QtWidgets.QGraphicsScene(self)

    self.empty_implant_res = QtGui.QPixmap("empty-implant-image.png")
    self.empty_implant_pixmap = QtWidgets.QGraphicsPixmapItem(self.empty_implant_res)
    self.scene.addItem(self.empty_implant_pixmap)
    self.empty_implant_pixmap.setPos(50, 0)

    self.electrode_res = QtGui.QPixmap("single-electrode-image-red.png")

    for active_electrode_idx in stim_configs[selected_stim_config]:
      electrode_pos = electrode_positions[active_electrode_idx]
      self.electrode_pixmap = QtWidgets.QGraphicsPixmapItem(self.electrode_res, self.empty_implant_pixmap)
      self.electrode_pixmap.setPos(electrode_pos[0], electrode_pos[1])

    # Draw stimulation intensity bar
    self.bar = QtWidgets.QGraphicsRectItem(0, 0, 10, 100)
    self.scene.addItem(self.bar)

    self.stable_stim_line = QtWidgets.QGraphicsLineItem(0, 50, 15, 50)
    self.scene.addItem(self.stable_stim_line)

    self.update_stable_stim(60)

    self.increase_stim_line = QtWidgets.QGraphicsLineItem(0, 30, 15, 30)
    self.scene.addItem(self.increase_stim_line)

    self.decrease_stim_line = QtWidgets.QGraphicsLineItem(0, 70, 15, 70)
    self.scene.addItem(self.decrease_stim_line)

    self.view = QtWidgets.QGraphicsView(self.scene, self)

    self.view.show()

  def update_stable_stim(self, value):
    self.stable_stim_line.setLine(0, value, 15, value)

  def update_increase_stim(self, value):
    self.increase_stim_line.setLine(0, value, 15, value)

  def update_decrease_stim(self, value):
    self.decrease_stim_line.setLine(0, value, 15, value)