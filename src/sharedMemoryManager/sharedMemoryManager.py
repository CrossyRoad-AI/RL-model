from multiprocessing import shared_memory

from constants.constants import *

# Global var for shared memory manager
shm_a = None

def initSharedMemoryReader():
    # Use global var
    global shm_a

    # Init shared memory space
    shm_a = shared_memory.SharedMemory(name = SHARED_MEMORY_FILENAME, size = SHARED_MEMORY_SIZE)

    print(shm_a.buf[0])

def isDataReady():
    return shm_a.buf[0]

def readBuffer():
    # May use bytes(shm_a.buf[0: SHARED_MEMORY_SIZE]), if need into bytes string
    return bytes(shm_a.buf[0: SHARED_MEMORY_SIZE])

def closeSharedMemory():
    global shm_a

    # Close shared memory space
    shm_a.close()