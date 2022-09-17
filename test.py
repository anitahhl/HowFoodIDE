from cvzone.SerialModule import SerialObject
from time import sleep
import cv2

arduino = SerialObject('/dev/cu.usbmodem141301')

while True:
    arduino.sendData([1])
    sleep(3)
    arduino.sendData([0])
    sleep(1)