#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2022/05/12
@description: Affiliate GUI for AdamPose focuses on
the reproduction of 3D pose estimation and displays
the motion parameters of key points.
"""
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import pandas as pd
import cv2
import yaml
import os
import numpy as np

from .AdamPosePlugUI import UI_AdamPosePlug
from .matplotlib_qt_figure import Figure3D
from data_analysis import *


COLORS = np.random.randint(0, 255, size=(21, 3), dtype="uint8")
mCOLORS = np.random.rand(21, 3)


class AdamPosePlugMainWindow(QMainWindow, UI_AdamPosePlug):
    def __init__(self):
        super(AdamPosePlugMainWindow, self).__init__()
        self.parameter_init()
        self.setupUi(self)
        self.gui_init()

    def parameter_init(self):
        # Some constants
        self.pause_flag = True
        self.static_flag = True
        self.key_points = None
        self.body_parts_sub = []
        self.body_parts = None
        self.body_parts_hash = None
        self.config = None
        self.imgs = None
        self.ixs = None
        self.colors = []

        self.pose_angles_display = False
        self.displacement_display = False
        self.velocity_display = False

        self.data_file_path = None
        self.video_file_path = None

        self.p3ds = None
        self.cap = None
        self.ts = None
        self.fps = None
        self.frame_number = None
        self.frame_width = None
        self.frame_height = None
        self.scale_width = None
        self.speed = None
        self.PoseEstimation2DVideo = None
        self.PoseEstimation3DVideo = None

        # Set font
        self.font = QFont()
        self.font.setFamily("Times New Roman")
        self.font.setPointSize(10)
        self.font.setBold(False)

        self.count = 0
        self.video_frame_count = 0

    # Initialize the GUI
    def gui_init(self):
        self.EditConfigFile.setEnabled(False)
        self.Start.setEnabled(False)
        self.Pause.setEnabled(False)
        self.Fast.setEnabled(False)
        self.Slow.setEnabled(False)
        self.PoseAngles.setEnabled(False)
        self.Displacement.setEnabled(False)
        self.Velocity.setEnabled(False)
        self.lineEdit.setReadOnly(True)
        for i in range(2,10):
            exec('self.lineEdit_%s.setReadOnly(True)' % i)
        # Link button and callback function
        self.LoadConfigFile.clicked.connect(self.load_config_file)
        self.EditConfigFile.clicked.connect(self.edit_config_file)

        # Button
        self.Start.clicked.connect(self.start)
        self.Pause.clicked.connect(self.pause_to_continue)
        self.Fast.clicked.connect(self.fast)
        self.Slow.clicked.connect(self.slow)
        self.PoseAngles.clicked.connect(self.pose_angles)
        self.Displacement.clicked.connect(self.displacement)
        self.Velocity.clicked.connect(self.velocity)
        self.Quit.clicked.connect(self.quit)

        # Information
        self.labtext = QTextBrowser()
        self.LineFigureLayoutText = QVBoxLayout(self.PromptInformation)
        self.LineFigureLayoutText.addWidget(self.labtext)
        self.labtext.setAlignment(Qt.AlignLeft)
        self.labtext.append("Welcome, I am AdamPosePlug!")
        self.labtext.setFont(self.font)

    # Callbacks for all keys on the GUI
    def start(self):
        """
        The callback of self.Start
        """
        self.visualize_init()
        self.Start.setEnabled(False)

    def pause_to_continue(self):
        """
        The callback of self.Pause
        """
        _translate = QCoreApplication.translate
        if self.pause_flag:
            self.timer.stop()
            self.pause_flag = False
            self.Pause.setText(_translate("AdamPosePlug", "Continue"))
        else:
            self.timer.start()
            self.pause_flag = True
            self.Pause.setText(_translate("AdamPosePlug", "Pause"))

    def fast(self):
        """
        The callback of self.Fast
        """
        if self.speed:
            self.timer.stop()
            self.speed *= 2
            self.ts = 1/self.speed
            self.timer.start(int(self.ts * 1000))

    def slow(self):
        """
        The callback of self.Slow
        """
        if self.speed:
            self.timer.stop()
            self.speed /= 2
            self.ts = 1 / self.speed
            self.timer.start(int(self.ts * 1000))
            
    def pose_angles(self):
        """
        The callback of self.PoseAngles
        """
        if not self.pose_angles_display:
            self.pose_angles_display = True
        else:
            self.pose_angles_display = False
            self.lineEdit.setPlaceholderText("")
            for i in range(2,4):
                exec('self.lineEdit_%s.setPlaceholderText("")' % i)

    def displacement(self):
        """
        The callback of self.Displacement
        """
        if not self.displacement_display:
            self.displacement_display = True
        else:
            for i in range(4, 7):
                exec('self.lineEdit_%s.setPlaceholderText("")' % i)
            self.displacement_display = False

    def velocity(self):
        """
        The callback of self.Velocity
        """
        if not self.velocity_display:
            self.velocity_display = True
        else:
            self.velocity_display = False
            for i in range(7, 10):
                exec('self.lineEdit_%s.setPlaceholderText("")' % i)

    def quit(self):
        """
        The callback of self.Quit
        """
        QApplication.instance().quit()

    # Load a fixed-format configuration file from the local
    def load_config_file(self):
        """
        The callback of self.LoadConfigFile
        """
        _translate = QCoreApplication.translate
        if not self.Start.isEnabled() and self.Pause.isEnabled():
            self.timer.stop()
        self.clear_layout()
        self.config_file_name, _ = QFileDialog.getOpenFileName(self, 'OpenFile', '.')
        if self.config_file_name:
            self.labtext.append("Load configuration file.....")
            self.load_file()
            self.labtext.append("Load configuration file Successfully!")
            self.labtext.append("Configuration File:" + self.config_file_name)
        else:
            return
        self.EditConfigFile.setEnabled(True)
        self.Start.setEnabled(True)
        if not self.pause_flag:
            self.Pause.setText(_translate("AdamPosePlug", "Pause"))
            self.pause_flag = True
        self.Pause.setEnabled(True)
        self.Fast.setEnabled(True)
        self.Slow.setEnabled(True)
        self.initialize_pose_parameter()
        self.create_keypoints_selection()
        self.get_data_file()
        self.get_video_file()
        # data preprocess
        self.datas, self.vdatas = self.data_preprocess()

    # Edit configuration file
    def edit_config_file(self):
        """
        The callback of self.EditConfigFile
        """
        path = os.getcwd()
        if self.config_file_name:
            md = "C:\\Windows\\System32\\notepad.exe" + " " + self.config_file_name
            os.system(md)
        else:
            return
        os.chdir(path)

    # Load configuration file
    def load_file(self):
        f = open(self.config_file_name, 'r', encoding='utf-8')
        cfg = f.read()
        self.config = yaml.full_load(cfg)

    # Initialize some key parameters of the target pose, such as key point information, data frame rate, etc.
    def initialize_pose_parameter(self):
        self.key_points = self.config['keypoints']
        self.body_parts = self.config['constraints']
        self.body_parts_hash = dict(zip(self.key_points, range(len(self.key_points))))
        self.fps = self.config['fps']
        self.speed = self.fps
        self.ts = 1 / self.fps

    # Get raw data for pose estimation
    def get_data_file(self):
        self.data_file_path = self.config['data_file_path']
        if self.data_file_path:
            self.p3ds = pd.read_csv(self.data_file_path)
        else:
            return

    # Get raw video for pose estimation
    def get_video_file(self):
        self.video_file_path = self.config['video_file_path']
        if self.video_file_path:
            self.cap = cv2.VideoCapture(self.video_file_path)
            self.frame_number = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            return

    # Creates a selectable keypoint array on the GUI based on the information provided by the configuration file
    def create_keypoints_selection(self):
        _translate = QCoreApplication.translate
        for keypoint_id, keypoint in enumerate(self.key_points):
            i = keypoint_id // 7
            j = keypoint_id % 7
            exec('self.keypoint%s=QCheckBox(self.layoutWidget_3)' % keypoint_id)
            exec('self.keypoint%s.setObjectName("keypoint%s")' % (keypoint_id, keypoint_id))
            exec('self.gridLayout.addWidget(self.keypoint%s, %d, %d, 1, 1)' % (keypoint_id, i, j))
            exec('self.keypoint%s.setText(_translate("AdamPosePlug", "%s"))' % (keypoint_id, keypoint))
            exec('self.keypoint%s.setFont(self.font)' % keypoint_id)
            exec('self.keypoint%s.stateChanged.connect(self.select_keypoints)' % keypoint_id)

    # The callback of keypoint array
    def select_keypoints(self):
        self.body_parts_sub = []
        for keypoint_id in range(len(self.key_points)):
            exec('self.body_parts_sub += [self.keypoint%s.text()] if self.keypoint%s.isChecked() else []'
                 % (keypoint_id, keypoint_id))
        if len(self.body_parts_sub)>0:
            self.Displacement.setEnabled(True)
            self.Velocity.setEnabled(True)
            if len(self.body_parts_sub) == 3:
                self.PoseAngles.setEnabled(True)
                self.angles = self.angles_process()
            else:
                self.pose_angles_display = False
                self.PoseAngles.setEnabled(False)
        if self.PoseEstimation3DVideo:
            self.PoseEstimation3DVideo.ax.cla()
            self.draw_figure_init()

    # Calculate the three pose angles formed by the three selected keypoints
    def angles_process(self):
        vecs = []
        for angle_keypoint in self.body_parts_sub:
            vec = [self.p3ds[angle_keypoint + '_x'], self.p3ds[angle_keypoint + '_y'], self.p3ds[angle_keypoint + '_z']]
            vec = [dat[~np.isnan(dat)] for dat in vec]
            vec = np.array(vec).T
            vecs.append(vec)

        angles = compute_pose_angles(vecs)*180.0/np.pi
        angles = np.array([[round(angleij,2) for angleij in anglei] for anglei in angles])
        return angles

    # Calculate displacement and velocity of selected keypoints
    def data_preprocess(self):
        datas = []
        vdatas = []
        for bpname in self.key_points:
            # dispalcement
            data = [self.p3ds[bpname + '_x'], self.p3ds[bpname + '_y'], self.p3ds[bpname + '_z']]
            data = [dat[~np.isnan(dat)] for dat in data]
            data = np.array(data).T
            data = np.array([[round(dataij, 2) for dataij in datai] for datai in data])
            datas.append(data)

            x = data[:,0]
            y = data[:,1]
            z = data[:,2]

            t = np.linspace(0, len(x), len(x), endpoint=False) * self.ts

            # velocity
            dx = cal_deriv(t, x)
            dy = cal_deriv(t, y)
            dz = cal_deriv(t, z)
            vdata = np.array([dx, dy, dz]).T
            vdata = np.array([[round(dataij, 2) for dataij in datai] for datai in vdata])
            vdatas.append(vdata)

        return np.array(datas), np.array(vdatas)

    # Visual initialization
    def visualize_init(self):
        if not self.PoseEstimation3DVideo:
            self.layout_create()
            self.figure_create()
            self.figure_set()
            self.add_figure()
        self.PoseEstimation3DVideo.ax.cla()
        self.draw_figure_init()
        self.timer_start()

    # Timer
    def timer_start(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.frame_count)
        self.timer.start(int(self.ts * 1000))

    # Video frame count
    def frame_count(self):
        self.video_frame()
        self.draw_figure()
        self.display_information()
        self.count += 1
        if self.count == self.frame_number:
            self.count = 0
            self.video_frame_count = 0
            self.cap = cv2.VideoCapture(self.video_file_path)

    # video frame extraction
    def video_frame(self):
        if self.count == self.video_frame_count:
            self.video_frame_count += 1
        elif self.count > self.video_frame_count:
            while self.video_frame_count != self.count:
                ret, frame = self.cap.read()
                self.video_frame_count += 1
        elif self.count < self.video_frame_count:
            self.count = self.video_frame_count
        ret, frame = self.cap.read()
        frame_height = int(self.frame_height-580)
        frame = frame[0:frame_height,:,:]
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (int(self.frame_width/frame_height*450), 450))
        Qframe = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
        self.PoseEstimation2DVideo.setPixmap(QPixmap.fromImage(Qframe))

    def layout_create(self):
        self.VideoPose2D = QVBoxLayout(self.PoseEstimation2D)
        self.VideoPose3D = QHBoxLayout(self.PoseEstimation3D)

    def figure_create(self):
        self.PoseEstimation2DVideo = QLabel()
        self.PoseEstimation3DVideo = Figure3D()

    def add_figure(self):
        self.VideoPose2D.addWidget(self.PoseEstimation2DVideo)
        self.VideoPose3D.addWidget(self.PoseEstimation3DVideo)

    def figure_set(self):
        self.PoseEstimation2DVideo.setAlignment(Qt.AlignCenter)

    def draw_figure_init(self):
        self.PoseEstimation3DVideo.ax.set_axis_off()
        self.PoseEstimation3DVideo.fig.tight_layout()
        self.PoseEstimation3DVideo.fig.subplots_adjust(top=1,left=0,right=1,hspace=0,wspace=0,bottom=0)
        self.imgs = []
        self.ixs = []
        dx = -150
        dy = -80
        dz = -200

        x = [dx, dx, dx]
        y = [dy, dy, dy]
        z = [dz, dz, dz]

        u = [1, 0, 0]
        v = [0, 1, 0]
        w = [0, 0, 1]
        colors = ['r','g','y']
        for i in range(3):
            self.PoseEstimation3DVideo.ax.quiver(x[i], y[i], z[i],
                      u[i], v[i], w[i],
                      length=60, normalize=True, color=colors[i],arrow_length_ratio=0.12)
        fontdict = {'family':'Times New Roman', 'size':19, 'style':'italic'}
        self.PoseEstimation3DVideo.ax.text(dx+65, dy, dz, 'x',fontdict=fontdict)
        self.PoseEstimation3DVideo.ax.text(dx, dy+65, dz, 'y',fontdict=fontdict)
        self.PoseEstimation3DVideo.ax.text(dx, dy, dz+65, 'z',fontdict=fontdict)
        for i,keypoint in enumerate(self.key_points):
            if len(self.body_parts_sub)>0:
                if keypoint in self.body_parts_sub:
                    self.imgs.append(self.PoseEstimation3DVideo.ax.plot(self.datas[i, 0, 0], 
                                                                        self.datas[i, 0, 1],
                                                                        -self.datas[i, 0, 2],
                                                                        marker='o',
                                                                        color=mCOLORS[i],
                                                                        markersize=10)[0])
                else:
                    self.imgs.append(self.PoseEstimation3DVideo.ax.plot(self.datas[i, 0, 0], 
                                                                        self.datas[i, 0, 1],
                                                                        -self.datas[i, 0, 2],
                                                                        marker='o', 
                                                                        color='b', 
                                                                        markersize=10)[0])
            else:
                self.imgs.append(self.PoseEstimation3DVideo.ax.plot(self.datas[i, 0, 0], 
                                                                    self.datas[i, 0, 1],
                                                                    -self.datas[i, 0, 2],
                                                                    marker='o',
                                                                    color='b',
                                                                    markersize=10)[0])
        for bodypart in self.body_parts:
            ix = [self.body_parts_hash[bp] for bp in bodypart]
            self.ixs.append(ix)
            self.imgs.append(
                self.PoseEstimation3DVideo.ax.plot(self.datas[ix, 0, 0], 
                                                   self.datas[ix, 0, 1],
                                                   -self.datas[ix, 0, 2], 'b-')[0])

    def draw_figure(self):
        count = self.count
        for i,bodypart in enumerate(self.key_points):
            self.imgs[i].set_data(self.datas[i, count, 0:2].transpose())
            self.imgs[i].set_3d_properties(-self.datas[i, count, 2].transpose(), 'z')
        for ix, line in zip(self.ixs, self.imgs[len(self.key_points):]):
            line.set_data(self.datas[ix, count, 0:2].transpose())
            line.set_3d_properties(-self.datas[ix, count, 2].transpose(), 'z')
        self.PoseEstimation3DVideo.draw()

    def display_information(self):
        ix = [self.body_parts_hash[bp] for bp in self.body_parts_sub]
        if self.pose_angles_display:
            self.lineEdit.setPlaceholderText(str(self.angles[self.count,0]))
            for i,j in enumerate(range(2,4)):
                exec('self.lineEdit_%s.setPlaceholderText("%s")' % (j,self.angles[self.count,i+1]))
        if self.displacement_display:
            for i,j in enumerate(range(4,7)):
                exec('self.lineEdit_%s.setPlaceholderText("%s")' % (j,self.datas[ix,self.count,i]))
        if self.velocity_display:
            for i,j in enumerate(range(7,10)):
                exec('self.lineEdit_%s.setPlaceholderText("%s")' % (j,self.vdatas[ix,self.count,i]))

    def information_clear(self):
        self.lineEdit.clear()
        for i in range(2,10):
            exec('self.lineEdit_%s.clear()'%i)

    def clear_layout(self):
        for i in range(self.gridLayout.count()):
            self.gridLayout.itemAt(i).widget().deleteLater()
