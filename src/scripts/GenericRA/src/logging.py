import inspect
Showlog = False
VISIBLE_TAGS=[]
import time

def TOGGLE_SHOWLOG(on):
    global Showlog
    Showlog = on

def REG_TAG(tag):
    VISIBLE_TAGS.append(tag)

def sourced_print(message, source):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for line in message.split("\n"):
        print(f"[{timestamp}] [{source}] {line}")

def LOG(message):
    caller = inspect.currentframe().f_back
    source = caller.f_code.co_name if caller else ""
    if Showlog:
        sourced_print(message, source)
        
def LOG_TAG(message,tag):
    caller = inspect.currentframe().f_back
    source = caller.f_code.co_name if caller else ""
    if Showlog and tag in VISIBLE_TAGS:
        sourced_print(message, source)