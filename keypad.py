try:
    import RPi.GPIO as GPIO
    emulation=False
except:
    print('RPi.GPIO not available, simulating with U I O P')
    emulation=True
import time
import os
import cv2

class GPIOPinReader:
    def __init__(self, shutdown_delay=5):
        self.pins = [6, 13, 19, 26]

        self.shutdown_delay = shutdown_delay

        if not emulation:
            GPIO.setmode(GPIO.BCM)
            for pin in self.pins:
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def read_pins(self):
        for pin_id in range(len(self.pins)):
            if not emulation:
                if not GPIO.input(self.pins[pin_id]):
                    if pin_id==3:
                        start_press = time.time()
                        while not GPIO.input(self.pins[pin_id]):
                            print(time.time()- start_press)
                            time.sleep(0.2)
                            if time.time() - start_press > self.shutdown_delay:
                                os.system('sudo shutdown -h now')
                                return 255
                        os.system('./screen_blank.sh')
                        return 255
                    print('[KEYPAD] Pressed key', pin_id)
                    return pin_id
            else:
                ret = cv2.waitKey(1)
                pin_assignments = { ord('u'):0, 
                                    ord('i'):1, 
                                    ord('o'):2,
                                    ord('p'):3}
                if ret in pin_assignments.keys():
                    print('[KEYPAD] Pressed key', pin_assignments[ret])
                    return pin_assignments[ret]

        return -1
    
    def waitKey(self, delay=-1):
        start = time.time()
        cv2.waitKey(1)
        key = self.read_pins()
        if key != -1:
            return key
        while (time.time() - start)*1000 < delay or delay<0:
            time.sleep(0.001)
            key = self.read_pins()
            if key != -1:
                return key
        return -1

if __name__=='__main__':
    pin_reader = GPIOPinReader()
    while True:
        pin_reader.read_pins()