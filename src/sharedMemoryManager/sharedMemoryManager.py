import struct
from multiprocessing import shared_memory

from utils.singleton import Singleton

from constants.constants import *

class SharedMemoryManager(metaclass = Singleton):
    def __init__(self):
        self._sharedMemory = self.initSharedMemory()

        self._reloadData = False

        self._parsedBuffer = {}
        self._listBuffer = []

    def initSharedMemory(self):
        return shared_memory.SharedMemory(name = SHARED_MEMORY_FILENAME, size = SHARED_MEMORY_SIZE)
    
    def closeSharedMemory(self):
        self._sharedMemory.close()
    
    def isDataReady(self):
        self._reloadData = True
        return self._sharedMemory.buf[0]
    
    def readAndParseBuffer(self):
        self._reloadData = False

        listBuffer = []
        parsedBuffer = {
            "score": 0,
            "player": {
                "alive": 0,
                "x": 0,
                "y": 0
            }
        }

        parsedBuffer["score"] = struct.unpack('<I', self._sharedMemory.buf[1 : 5])[0]
        parsedBuffer["player"]["alive"] = self._sharedMemory.buf[5]
        parsedBuffer["player"]["x"] = struct.unpack('<f', self._sharedMemory.buf[6 : 10])[0]
        parsedBuffer["player"]["y"] = struct.unpack('<f', self._sharedMemory.buf[10 : 14])[0]

        listBuffer.append(parsedBuffer["score"])
        listBuffer.append(parsedBuffer["player"]["alive"])
        listBuffer.append(parsedBuffer["player"]["x"])
        listBuffer.append(parsedBuffer["player"]["y"])

        offset = 14
        for j, field in enumerate(FIELDS_TO_QUERY):
            parsedBuffer[field] = { "count": 0 }
            parsedBuffer[field]["count"] = struct.unpack('<I', self._sharedMemory.buf[offset : offset + 4])[0]

            for i in range(0, parsedBuffer[field]["count"] * 2):
                try:
                    listBuffer.append(struct.unpack('<f', self._sharedMemory.buf[offset + 4 + i * 4 : offset + 4 + i * 4 + 4])[0])
                except struct.error:
                    print(bytes(self._sharedMemory.buf[offset : offset + 4]), self._sharedMemory.buf[offset : offset + 4], offset)
                    print(i, parsedBuffer[field]["count"] * 2, j, field)
                    exit(150)

            # Pad array
            if parsedBuffer[field]["count"] * 2 < PAD_PER_FIELDS[j]: listBuffer = listBuffer + [0] * (PAD_PER_FIELDS[j] - parsedBuffer[field]["count"] * 2)

            offset += 4 + parsedBuffer[field]["count"] * 2 * 4

        self._parsedBuffer = parsedBuffer
        self._listBuffer = listBuffer
        
    @property
    def buffer(self):
        return self._sharedMemory.buf
    
    @property
    def parsedBuffer(self):
        if self._reloadData: self.readAndParseBuffer()
        return self._parsedBuffer
    
    @property
    def listBuffer(self):
        if self._reloadData: self.readAndParseBuffer()
        return self._listBuffer
    
    def __del__(self):
        self.closeSharedMemory()