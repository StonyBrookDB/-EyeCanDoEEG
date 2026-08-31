import numpy as np
from serial import Serial
import csv
from datetime import datetime


class EEG_Driver:
    def __init__(self, serial_port, baud_rate=3000000, buffer_size=2020):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.buffer_size = buffer_size
        self.ser = Serial(port=self.serial_port, baudrate=self.baud_rate, timeout=1)

        self.ch0 = np.zeros(buffer_size)
        self.ch1 = np.zeros(buffer_size)
        self.ch2 = np.zeros(buffer_size)
        self.ch3 = np.zeros(buffer_size)
        self.ch4 = np.zeros(buffer_size)
        self.ch5 = np.zeros(buffer_size)
        self.ch6 = np.zeros(buffer_size)
        self.ch7 = np.zeros(buffer_size)
        self.ch8 = np.zeros(buffer_size)
        self.ch9 = np.zeros(buffer_size)
        self.ch10 = np.zeros(buffer_size)
        self.ch11 = np.zeros(buffer_size)
        self.ch12 = np.zeros(buffer_size)
        self.ch13 = np.zeros(buffer_size)
        self.ch14 = np.zeros(buffer_size)
        self.ch15 = np.zeros(buffer_size)

    def read_sample(self):
        """Read one raw frame from the serial port and return a list of the
        16 signed channel values, or None if no complete frame was read
        (e.g. a stray byte, or the terminator didn't match)."""
        data = self.ser.read(1)
        if data != b'\xAB':
            return None
        data += self.ser.read(51)
        if len(data) != 52 or data[-2:] != b'\xDC\xBA':
            return None
        return [
            int.from_bytes(data[2 + 3 * i:5 + 3 * i], byteorder='little', signed=True)
            for i in range(16)
        ]

    def read_data(self):
        buffer_index = 0

        # --- CSV setup (NEW) ---
        filename = f"CurveData_{datetime.now().strftime('%H%M%S')}.csv"
        f = open(filename, "w", newline="", buffering=1)
        writer = csv.writer(f)

        writer.writerow(["time"] + [f"CH{i} " for i in range(16)])

        print(f"[INFO] Saving EEG data to {filename}")

        try:
            while buffer_index < self.buffer_size:
                channels = self.read_sample()
                if channels is None:
                    continue

                for i, value in enumerate(channels):
                    getattr(self, f"ch{i}")[buffer_index] = value

                # --- write CSV row ---
                timestamp = datetime.now().strftime("%y-%m-%d %H:%M:%S.%f")[:-3]
                writer.writerow([timestamp] + channels)

                # periodic flush
                if buffer_index % 202 == 0:
                    f.flush()

                buffer_index += 1
        finally:
            f.flush()
            f.close()
            print("Data collection complete")
        return None


if __name__ == '__main__':
    
    ################
    # Hardcode COM3
    ################
    eeg_driver = EEG_Driver(serial_port='COM3', baud_rate=3000000, buffer_size=6000)
    
    eeg_driver.read_data()
    print(eeg_driver.ch0)
    print(eeg_driver.ch1)
    print(eeg_driver.ch2)
    print(eeg_driver.ch3)
    print(eeg_driver.ch4)
    print(eeg_driver.ch5)
    print(eeg_driver.ch6)
    print(eeg_driver.ch7)
    print(eeg_driver.ch8)
    print(eeg_driver.ch9)
    print(eeg_driver.ch10)
    print(eeg_driver.ch11)
    print(eeg_driver.ch12)
    print(eeg_driver.ch13)
    print(eeg_driver.ch14)
    print(eeg_driver.ch15)
