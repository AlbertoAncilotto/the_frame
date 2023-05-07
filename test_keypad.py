from keypad import GPIOPinReader

gpio = GPIOPinReader()
while True:
    key = gpio.waitKey(1)
    print(key)