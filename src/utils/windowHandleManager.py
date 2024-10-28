import win32gui
import win32process as wproc
import win32api as wapi

from constants.constants import *

def winEnumHandler(hwnd, ctx, windowName):
    if win32gui.IsWindowVisible( hwnd ):
        print(hex(hwnd), win32gui.GetWindowText(hwnd))
        if win32gui.GetWindowText(hwnd) == GAME_WINDOW_NAME : win32gui.SetFocus(hwnd)

def focusGameWindow():
    handle = win32gui.FindWindow(None, GAME_WINDOW_NAME)
    if not handle:
        print("Invalid window handle")
        return
    
    remote_thread, _ = wproc.GetWindowThreadProcessId(handle)
    wproc.AttachThreadInput(wapi.GetCurrentThreadId(), remote_thread, True)
    prev_handle = win32gui.SetFocus(handle)