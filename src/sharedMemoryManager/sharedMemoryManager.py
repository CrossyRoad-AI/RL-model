import struct

from multiprocessing import shared_memory

from constants.constants import *

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances: cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class SharedMemoryManager(metaclass = Singleton):
    def __init__(self):
        self._sharedMemory = self.initSharedMemory()

    def initSharedMemory(self):
        return shared_memory.SharedMemory(name = SHARED_MEMORY_FILENAME, size = SHARED_MEMORY_SIZE)
    
    def closeSharedMemory(self):
        self._sharedMemory.close()
    
    def isDataReady(self):
        return self._sharedMemory.buf[0]
        
    @property
    def buffer(self):
        return self._sharedMemory.buf
    
    @property
    def parsedBuffer(self):
        fieldsToQuery = ["grass", "trees", "water", "water-lily"]
        data = {
            "score": 0,
            "player": {
                "alive": 0,
                "x": 0,
                "y": 0
            }
        }

        data["score"] = struct.unpack('<I', self._sharedMemory.buf[1 : 5])[0]
        data["player"]["alive"] = self._sharedMemory.buf[5]
        data["player"]["x"] = struct.unpack('<f', self._sharedMemory.buf[6 : 10])[0]
        data["player"]["y"] = struct.unpack('<f', self._sharedMemory.buf[10 : 14])[0]

        offset = 14
        for field in fieldsToQuery:
            data[field] = {
                "count": 0,
                "positions": []
            }

            data[field]["count"] = struct.unpack('<I', self._sharedMemory.buf[offset : offset + 4])[0]
            for i in range(0, data[field]["count"] * 2):
                data[field]["positions"].append(struct.unpack('<f', self._sharedMemory.buf[offset + 4 + i * 4 : offset + 4 + i * 4 + 4])[0])
            offset += 4 + data[field]["count"] * 2 * 4

        return data
    
    def __del__(self):
        self.closeSharedMemory()