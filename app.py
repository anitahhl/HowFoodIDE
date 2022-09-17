import cv2  # opencv
import urllib.request
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
from demo import Ui_MainWindow
import sys


url = 'http://192.168.66.32:81/stream'
# url = 'http://192.168.66.32/'
winName = 'ESP32 CAMERA'
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

classNames = []
classFile = 'yolo/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'yolo/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'yolo/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
# net.setInputSize(480,480)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while (1):
    imgResponse = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # vertical
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #black and white

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
            cv2.putText(img, 'orange', (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, # classNames[classId - 1]
                                (0, 255, 0), 2)

        cv2.imshow(winName, img)

    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break

    cv2.destroyAllWindows()

'''
        food = classNames[classId - 1]
        if food: #(food == 'orange') or (food == 'onion'):
            class MainWindow(QtWidgets.QMainWindow):
                def __init__(self):
                    super(MainWindow, self).__init__()
                    self.ui = Ui_MainWindow()
                    self.ui.setupUi(self)

                    now = QtCore.QDate.currentDate()
                    current_date = now.toString('yyyy/MM/dd')
                    exp_dict = {
                        'orange': 7,
                        'onion': 20,
                    }
                    exp = now.addDays(7) #exp_dict[food])
                    exp_date = exp.toString('yyyy/MM/dd')

                    self.ui.label.setText(food)
                    self.ui.label_3.setText('0')
                    self.ui.label_5.setText(current_date)
                    self.ui.label_6.setText(exp_date)

            if __name__ == '__main__':
                app = QtWidgets.QApplication([])
                mainwindow = MainWindow()
                mainwindow.show()
                sys.exit(app.exec_())
        else:
            pass
'''