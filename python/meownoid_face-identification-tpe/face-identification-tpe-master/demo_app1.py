import sys
import os.path
import random

import numpy as np

from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QDialog)
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QPen, QColor, QFont
from PyQt5.QtCore import QCoreApplication, Qt

from skimage import io

from model import FaceVerificator


IMWIDTH = 600
IMHEIGHT = 600
BASECOLOR = QColor('yellow')
TEXTCOLOR = QColor('yellow')
BASEWIDTH = 2.0
BOXSIZE = 227
TEXTWIDTH = 2.0
TEXTSIZE = 16
TEXTFONT = QFont('Sans', TEXTSIZE)
MATCHBACKCOLOR = QColor('cyan')


class TablePopup(QDialog):
    def __init__(self, scores, comp):
        super().__init__()
        self.initUI(scores, comp)
        self.setAttribute(Qt.WA_DeleteOnClose)

    def initUI(self, scores, comp):
        layout = QVBoxLayout(self)
        rows, cols = scores.shape
        table = QTableWidget(rows, cols)
        layout.addWidget(table)

        for i in range(rows):
            for j in range(cols):
                item = QTableWidgetItem('{:.3f}'.format(scores[i, j]))
                if comp[i, j]:
                    item.setBackground(MATCHBACKCOLOR)
                table.setItem(i, j, item)

        hh = list(map(lambda s: '2: {}'.format(s), range(cols)))
        vh = list(map(lambda s: '1: {}'.format(s), range(rows)))

        table.setHorizontalHeaderLabels(hh)
        table.setVerticalHeaderLabels(vh)

        self.setLayout(layout)
        self.setWindowTitle('Scores table')
        self.show()


class Main(QWidget):
    def __init__(self):
        super().__init__()

        self.i_loaded = [False, False]
        self.i_path = [None, None]
        self.i_np = [None, None]
        self.i_scale = [1.0, 1.0]
        self.i_pixmap = [None, None]
        self.i_lbl = [None, None]

        self.dist = 0.85

        self.matched = False

        self.fv = FaceVerificator('./model')
        self.fv.initialize_model()

        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 1280, 720)
        self.setWindowTitle('Face verification demo 1')

        self.l1_btn = QPushButton('Load 1...')
        self.l2_btn = QPushButton('Load 2...')

        self.match_btn = QPushButton('Match')
        self.exit_btn = QPushButton('Exit')

        self.i_lbl[0] = QLabel()
        self.i_lbl[1] = QLabel()

        self.exit_btn.clicked.connect(QCoreApplication.instance().quit)
        self.l1_btn.clicked.connect(self.l1_clicked)
        self.l2_btn.clicked.connect(self.l2_clicked)
        self.match_btn.clicked.connect(self.match_clicked)

        hbox = QHBoxLayout()
        hbox.addWidget(self.l1_btn)
        hbox.addWidget(self.l2_btn)
        hbox.addStretch(1)
        hbox.addWidget(self.match_btn)
        hbox.addStretch(1)
        hbox.addWidget(self.exit_btn)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.i_lbl[0])
        hbox2.addStretch(1)
        hbox2.addWidget(self.i_lbl[1])

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox2)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.show()

    def image_file_dialog(self):
        return QFileDialog().getOpenFileName(self,
                                             'Single File',
                                             './',
                                             'Image files (*.jpg *.jpeg *.png)')[0]

    def load_img_to(self, img_path, n):
        pixmap = QPixmap(img_path)

        scale = 1
        if pixmap.width() > IMWIDTH:
            scale = IMWIDTH / pixmap.width()
        if pixmap.height() > IMHEIGHT:
            scale = IMHEIGHT / pixmap.height()

        if scale != 1:
            neww = pixmap.width() * scale
            newh = pixmap.height() * scale
            pixmap = pixmap.scaled(neww, newh)

        self.i_scale[n] = scale

        self.i_pixmap[n] = pixmap
        self.i_lbl[n].setPixmap(pixmap)
        self.i_lbl[n].show()

    def ln_clicked(self, n):
        img_path = self.image_file_dialog()

        if (img_path is None) or (not os.path.exists(img_path)):
            return

        self.i_path[n] = img_path
        self.i_loaded[n] = True
        self.i_np[n] = io.imread(img_path)
        self.load_img_to(img_path, n)

        if n == 0 and self.i_loaded[1]:
            self.load_img_to(self.i_path[1], 1)
        elif n == 1 and self.i_loaded[0]:
            self.load_img_to(self.i_path[0], 0)

        self.matched = False

    def l1_clicked(self):
        self.ln_clicked(0)

    def l2_clicked(self):
        self.ln_clicked(1)

    def match_clicked(self):
        if self.matched:
            return

        if not (self.i_loaded[0] and self.i_loaded[1]):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Load both images!')
            msg.setWindowTitle('Error')
            msg.exec_()
            return

        self.do_vodo_magic()

    def no_faces_on_err(self, n):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText('No faces found on image {}'.format(n))
        msg.setWindowTitle('Error')
        msg.exec_()
        return

    def generate_color(self):
        return QColor(random.randint(0, 255),
                      random.randint(0, 255),
                      random.randint(0, 255))

    def draw_box(self, n, box, color, style, num):
        x1, y1, x2, y2 = box.left(), box.top(), box.right(), box.bottom()

        x1 = int(x1 * self.i_scale[n])
        y1 = int(y1 * self.i_scale[n])
        x2 = int(x2 * self.i_scale[n])
        y2 = int(y2 * self.i_scale[n])

        width = BASEWIDTH
        if style == 'match':
            width *= 2

        painter = QPainter(self.i_pixmap[n])
        painter.setPen(QPen(QBrush(color), width))
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        painter.setPen(QPen(QBrush(TEXTCOLOR), TEXTWIDTH))
        painter.setFont(TEXTFONT)
        painter.drawText(x1, y2 + TEXTSIZE + 2 * BASEWIDTH, '{}: {}'.format(n + 1, num))
        painter.end()
        self.i_lbl[n].setPixmap(self.i_pixmap[n])

    def do_vodo_magic(self):
        faces_0 = self.fv.process_image(self.i_np[0])
        faces_1 = self.fv.process_image(self.i_np[1])

        n_faces_0 = len(faces_0)
        n_faces_1 = len(faces_1)

        if n_faces_0 == 0:
            self.no_faces_on_err(1)
            return

        if n_faces_1 == 0:
            self.no_faces_on_err(2)
            return

        rects_0 = list(map(lambda p: p[0], faces_0))
        rects_1 = list(map(lambda p: p[0], faces_1))

        embs_0 = list(map(lambda p: p[1], faces_0))
        embs_1 = list(map(lambda p: p[1], faces_1))

        scores, comps = self.fv.compare_many(self.dist, embs_0, embs_1)

        drawn_1 = [False] * n_faces_1

        for i in range(n_faces_0):
            color = BASECOLOR
            style = 'base'

            k = np.argmax(scores[i])
            if comps[i, k]:
                color = self.generate_color()
                style = 'match'
                drawn_1[k] = True
                self.draw_box(1, rects_1[k], color, style, k)

            self.draw_box(0, rects_0[i], color, style, i)

        color = BASECOLOR
        for j in range(n_faces_1):
            if not drawn_1[j]:
                self.draw_box(1, rects_1[j], color, 'base', j)

        tbl = TablePopup(scores, comps)
        tbl.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Main()
    sys.exit(app.exec_())
