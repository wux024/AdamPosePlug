#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2022/8/21 17:19
"""
import sys
from PyQt5.QtWidgets import *
from ui_adamposeplug import *

if __name__ == "__main__":
    styles = ['windowsvista', 'Windows', 'Fusion']
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.create(styles[1]))
    mymain = AdamPosePlugMainWindow()
    mymain.show()
    sys.exit(app.exec())