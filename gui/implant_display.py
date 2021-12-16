from PySide6 import QtCore, QtWidgets, QtGui

from PySide6.QtGui import QPen, QColor

red_pen = QPen(QColor('red'))

class ImplantWidget(QtWidgets.QWidget):
  def __init__(self):
    super().__init__()

    stim_configs = {"config12": [[13, 14], [15]]}

    selected_stim_config = "config12"

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

    self.anode_res = QtGui.QPixmap("single-electrode-image-white.png")
    self.cathode_res = QtGui.QPixmap("single-electrode-image-red.png")
    self.inactive_res = QtGui.QPixmap("single-electrode-image-grey.png")

    for i, electrode_pos in enumerate(electrode_positions):
      if i in stim_configs[selected_stim_config][0]:
        anode_pixmap = QtWidgets.QGraphicsPixmapItem(self.anode_res, self.empty_implant_pixmap)
        anode_pixmap.setPos(electrode_pos[0], electrode_pos[1])
      elif i in stim_configs[selected_stim_config][1]:
        cathode_pixmap = QtWidgets.QGraphicsPixmapItem(self.cathode_res, self.empty_implant_pixmap)
        cathode_pixmap.setPos(electrode_pos[0], electrode_pos[1])
      else:
        inactive_pixmap = QtWidgets.QGraphicsPixmapItem(self.inactive_res, self.empty_implant_pixmap)
        inactive_pixmap.setPos(electrode_pos[0], electrode_pos[1])

    for anode_electrode_idx in stim_configs[selected_stim_config][0]:
      electrode_pos = electrode_positions[anode_electrode_idx]
      self.electrode_pixmap = QtWidgets.QGraphicsPixmapItem(self.anode_res, self.empty_implant_pixmap)
      self.electrode_pixmap.setPos(electrode_pos[0], electrode_pos[1])

    # Draw stimulation intensity bar
    self.stim_bar_height = 200
    self.bar = QtWidgets.QGraphicsRectItem(0, 0, 10, self.stim_bar_height)
    self.scene.addItem(self.bar)

    self.stable_stim_line = QtWidgets.QGraphicsLineItem(0, 50, 15, 50)
    self.scene.addItem(self.stable_stim_line)

    #self.increase_stim_line = QtWidgets.QGraphicsLineItem(0, 30, 15, 30)
    #self.scene.addItem(self.increase_stim_line)

    #self.decrease_stim_line = QtWidgets.QGraphicsLineItem(0, 70, 15, 70)
    #self.scene.addItem(self.decrease_stim_line)

    self.actual_stim_line = QtWidgets.QGraphicsLineItem(0, 50, 15, 50)
    self.actual_stim_line.setPen(red_pen)
    self.scene.addItem(self.actual_stim_line)

    self.view = QtWidgets.QGraphicsView(self.scene, self)

    self.view.show()

  def update_stim(self, line_item, value):
    line_item.setLine(0, self.stim_bar_height - value, 15, self.stim_bar_height - value)

  def update(self, current_angle, target_angle):
    # Suppose the stable stim is proportional to the angle measured from bottom
    stable_stim = (((current_angle / 180) * self.stim_bar_height) // 2) + (self.stim_bar_height // 4)
    self.update_stim(self.stable_stim_line, stable_stim)
    # Suppose the actual stim is proportional to the angle difference
    extra = int(0.6 * ((target_angle - current_angle) / 180) * self.stim_bar_height)
    actual_stim = stable_stim + extra
    self.update_stim(self.actual_stim_line, actual_stim)