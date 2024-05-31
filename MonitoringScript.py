import psutil
import GPUtil
from time import sleep
from datetime import datetime

def print_system_stats(proc=None):
    # Get current time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n--- System Stats at {current_time} ---")

    # Get RAM details
    memory_info = psutil.virtual_memory()
    print(f"Total RAM: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"Available RAM: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"Used RAM: {memory_info.used / (1024 ** 3):.2f} GB")
    print(f"RAM Usage Percentage: {memory_info.percent}%")

    # Get VRAM details
    print("\n--- GPU Stats ---")
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name}")
        print(f"  VRAM Total: {gpu.memoryTotal} MB")
        print(f"  VRAM Used: {gpu.memoryUsed} MB")
        print(f"  VRAM Free: {gpu.memoryFree} MB")
        print(f"  GPU Load: {gpu.load * 100:.2f}%")

    # If a specific process is provided, get its RAM usage
    if proc:
        try:
            print("\n--- Process Stats ---")
            proc_memory_info = proc.memory_info()
            print(f"Process RAM Usage: {proc_memory_info.rss / (1024 ** 3):.2f} GB")
            print(f"Process RAM Usage Percentage: {proc.memory_percent():.2f}%")
        except psutil.NoSuchProcess:
            print("\n--- Process Stats ---")
            print("Process no longer exists.")
    print("-" * 40)

def find_process_by_name(name):
    # Find process by name
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == name or proc.info['name'] == f"{name}.exe":
            return proc
    return None

def monitor_maindyn():
    while True:
        proc = find_process_by_name("maindyn.py")
        if proc:
            print(f"Monitoring process: {proc.pid}")
            print_system_stats(proc)
        else:
            print("Process 'maindyn.py' is not running.")
            print_system_stats()
        
        sleep(5)

if __name__ == "__main__":
    monitor_maindyn()
