import resource
import platform
import sys
import numpy as np


def memory_limit(percentage: float):
    if platform.system() != "Linux":
        print('Only works on linux!')
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * percentage, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def memory(percentage=0.8):
    def decorator(function):
        def wrapper(*args, **kwargs):
            memory_limit(percentage)
            return function(*args, **kwargs)

            # try:
            #     return function(*args, **kwargs)
            # except MemoryError:
            #     mem = get_memory() / 1024 /1024
            #     print('Remain: %.2f GB' % mem)
            #     sys.stderr.write('\n\nERROR: Memory Exception\n')
            #     sys.exit(1)
        return wrapper
    return decorator

#@memory(percentage=0.01)
def create():
    #np.ones((5000, 5000))
    pass

print(get_memory() * 1.5)
create()
np.ones((5000, 5000))