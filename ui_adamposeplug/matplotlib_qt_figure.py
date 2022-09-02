#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2022/8/11 14:07
@description:
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class Figure3D(FigureCanvas):
    def __init__(self, width=3, height=2, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(Figure3D, self).__init__(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

class Figure2D(FigureCanvas):
    def __init__(self, width=3, height=2, dpi=80):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(Figure2D, self).__init__(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
