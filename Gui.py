import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
import numpy as np
from Networks import *
from PIL import Image

net = SimpleCNN(input_dim=(1, 28, 28), conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                hidden_size=100, output_size=10, weight_init_std=0.01)

net.load_params()

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口图标
        self.setWindowIcon(QIcon('icon.png'))

        # 设置窗口标题和初始大小
        self.setWindowTitle('Tiger or Lion')
        self.setGeometry(100, 100, 800, 600)

        # 创建垂直布局
        layout = QVBoxLayout()

        # 创建按钮
        self.btn = QPushButton('请导入待识别的图片')
        self.btn.clicked.connect(self.open_image)
        layout.addWidget(self.btn)

        # 创建标签用于展示图片
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # 创建标签来显示文本
        self.textLabel = QLabel()
        self.textLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.textLabel)

        # 设置布局
        self.setLayout(layout)

    def open_image(self):
        # 打开文件对话框，获取图片文件路径
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '',
                                                   'Image Files (*.png *.jpg *.jpeg *.bmp *.gif)')
        if file_name:
            # 加载图片
            pixmap = QPixmap(file_name)

            im = Image.open(file_name)

            gray_im = im.convert("L")

            resize_im = gray_im.resize((28, 28))

            im_a = np.array(resize_im)

            im_a = np.expand_dims(np.expand_dims(im_a, 0), 0)

            y = np.argmax(net.predict(im_a))

            # 调整图片大小以适应窗口
            pixmap = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)
            self.label.setPixmap(pixmap)

            self.textLabel.setText(f"Loaded Image: {y}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
