import time
import urllib as url
import serial

ArduinoSerial=serial.Serial("/dev/ttyACM1",115200)
ArduinoSerial1=serial.Serial("/dev/ttyACM0",19200)
time.sleep(2)

while True:
  s=ArduinoSerial.readline()
  print s
  temp=url.urlopen("https://api.thingspeak.com/update?api_key=6T00OH5K5ENKA9HX&field1="+str(s))
  time.sleep(15)
  t=ArduinoSerial1.readline()
  print t
  temp=url.urlopen("https://api.thingspeak.com/update?api_key=6T00OH5K5ENKA9HX&field2="+str(t))
  time.sleep(15)
