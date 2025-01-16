from useq import MDAEvent
import queue
import pymmcore_plus

import time


# from multiprocessing import Process, Queue as MPQueue
# def run_mda_in_process(queue, stop_event):
#     mmc = pymmcore_plus.CMMCorePlus()
#     mmc.loadSystemConfiguration("E:\\MicroManagerConfigs\\Ti2CicercoConfig_w_DMD_21_w_ttl.cfg")
#     mmc.run_mda(iter(queue.get, stop_event))
#     mmc.mda.events.frameReady.disconnect()
#     @mmc.mda.events.frameReady.connect
#     def on_frame(img, event):
#         print(f'Frame {event.index} received: {img.shape}')

# if __name__ == "__main__":
#     STOP_EVENT = object()
#     event_queue = MPQueue()
#     process = Process(target=run_mda_in_process, args=(event_queue, STOP_EVENT))
#     process.start()

#     event_queue.put(MDAEvent(index={"t": 0}, exposure=2000, channel={"config": "BF", "group": "WF_TTL"}))
#     time.sleep(10)
#     event_queue.put(MDAEvent(index={"t": 1}, exposure=1000, channel={"config": "BF", "group": "WF_TTL"}))
#     time.sleep(10)
#     event_queue.put(MDAEvent(index={"t": 2}, exposure=1000, channel={"config": "BF", "group": "WF_TTL"}))
#     time.sleep(15)
#     event_queue.put(STOP_EVENT)
#     process.join()



# mmc = pymmcore_plus.CMMCorePlus()
# mmc.loadSystemConfiguration("E:\\MicroManagerConfigs\\Ti2CicercoConfig_w_DMD_21_w_ttl.cfg")

# def wakeup_laser(lumencore_ip="192.168.201.200"):
#     url = f"http://{lumencore_ip}/service/?command=WAKEUP"
#     requests.get(url)



# STOP_EVENT = object()
# queue = Queue()
# queue_sequence = iter(queue.get, STOP_EVENT)
# mmc.run_mda(queue_sequence)
# mmc.mda.events.frameReady.disconnect()
# @mmc.mda.events.frameReady.connect
# def on_frame(img, event):
#     print(f'Frame {event.index} received: {img.shape}')

# queue.put(MDAEvent(index={"t": 0}, exposure=2000, channel={"config": "BF", "group": "WF_TTL"}))
# time.sleep(10)
# queue.put(MDAEvent(index={"t": 1}, exposure=1000, channel={"config": "BF", "group": "WF_TTL"}))
# time.sleep(15)
# queue.put(MDAEvent(index={"t": 2}, exposure=1000, channel={"config": "BF", "group": "WF_TTL"}))
# time.sleep(15)
# queue.put(STOP_EVENT)



# def run_controller_in_process(event_queue, stop_event):
#     event_queue.put(MDAEvent(index={"t": 0}, exposure=2000, channel={"config": "BF", "group": "WF_TTL"}))
#     time.sleep(10)
#     event_queue.put(MDAEvent(index={"t": 1}, exposure=1000, channel={"config": "BF", "group": "WF_TTL"}))
#     time.sleep(10)
#     event_queue.put(MDAEvent(index={"t": 2}, exposure=1000, channel={"config": "BF", "group": "WF_TTL"}))
#     time.sleep(15)
#     event_queue.put(stop_event)

# if __name__ == "__main__":
#     STOP_EVENT = object()
#     event_queue = MPQueue()
#     mmc = pymmcore_plus.CMMCorePlus()
#     mmc.loadSystemConfiguration("E:\\MicroManagerConfigs\\Ti2CicercoConfig_w_DMD_21_w_ttl.cfg")
#     mmc.run_mda(iter(event_queue.get, STOP_EVENT))

#     mmc.mda.events.frameReady.disconnect()
#     @mmc.mda.events.frameReady.connect
#     def on_frame(img, event):
#         print(f'Frame {event.index} received: {img.shape}')
#     process = Process(target=run_controller_in_process, args=(event_queue, STOP_EVENT))
#     process.start()
#     print("before join")
#     process.join()

# import threading

# def run_controller_in_thread(event_queue, stop_event):
#     time.sleep(5)
#     print("gnom")
#     event_queue.put(MDAEvent(index={"t": 0}, exposure=2000, channel={"config": "BF", "group": "WF_TTL"}, min_start_time=0))
#     time.sleep(10)
#     print("gumba")
#     event_queue.put(MDAEvent(index={"t": 1}, exposure=1000, channel={"config": "BF", "group": "WF_TTL"}, min_start_time=0))
#     time.sleep(10)
#     print("Yoshi")
#     event_queue.put(MDAEvent(index={"t": 2}, exposure=1000, channel={"config": "BF", "group": "WF_TTL"}, min_start_time=0))
#     time.sleep(15)
#     event_queue.put(stop_event)

# def run_mda_in_thread(_mmc, event_queue, stop_event): 
#     _mmc.mda.run(iter(event_queue.get, stop_event))


# if __name__ == "__main__":
#     STOP_EVENT = object()
#     event_queue = queue.Queue()
#     event_queue.put(MDAEvent())
#     mmc = pymmcore_plus.CMMCorePlus()
#     mmc.loadSystemConfiguration("E:\\MicroManagerConfigs\\Ti2CicercoConfig_w_DMD_21_w_ttl.cfg")

#     mmc.mda.events.frameReady.disconnect()

#     @mmc.mda.events.frameReady.connect
#     def on_frame(img, event):
#         print(f'Frame {event.index} received: {img.shape}')
#     process = threading.Thread(target=run_controller_in_thread, args=(event_queue, STOP_EVENT))
#     process_mda_runner = threading.Thread(target=run_mda_in_thread, args=(mmc, event_queue, STOP_EVENT))
#     process_mda_runner.start()
#     process.start()
#     print("before join")
#     process.join()
#     process_mda_runner.join()
#     print("after join")



# import threading
# def run_controller_in_thread(event_queue, stop_event):
#     time.sleep(5)
#     print("1st")
#     event_queue.put(MDAEvent(index={"t": 0}, exposure=2000, min_start_time=0))
#     time.sleep(10)
#     print("2nd")
#     event_queue.put(MDAEvent(index={"t": 1}, exposure=1000, min_start_time=0))
#     time.sleep(10)
#     print("3rd")
#     event_queue.put(MDAEvent(index={"t": 2}, exposure=1000, min_start_time=0))
#     time.sleep(15)
#     event_queue.put(stop_event)

# def run_mda_in_thread(_mmc, event_queue, stop_event): 
#     _mmc.run_mda(iter(event_queue.get, stop_event), block=False)


# if __name__ == "__main__":
#     STOP_EVENT = object()
#     event_queue = queue.Queue()
#     mmc = pymmcore_plus.CMMCorePlus()
#     mmc.loadSystemConfiguration()
#     # mmc.loadSystemConfiguration("E:\\MicroManagerConfigs\\Ti2CicercoConfig_w_DMD_21_w_ttl.cfg")

#     mmc.mda.events.frameReady.disconnect()

#     @mmc.mda.events.frameReady.connect
#     def on_frame(img, event):
#         print(f'Frame {event.index} received: {img.shape}')
#     process = threading.Thread(target=run_controller_in_thread, args=(event_queue, STOP_EVENT))
#     process_mda_runner = threading.Thread(target=run_mda_in_thread, args=(mmc, event_queue, STOP_EVENT))
#     process_mda_runner.start()
#     process.start()
#     print("before join")
#     process.join()
#     process_mda_runner.join()
#     print("after join")


# own implementation
# import threading
# def run_controller_in_thread(event_queue, stop_event):
#     time.sleep(5)
#     print("1st")
#     event_queue.put(MDAEvent(index={"t": 0}, exposure=2000, min_start_time=0))
#     time.sleep(10)
#     print("2nd")
#     event_queue.put(MDAEvent(index={"t": 1}, exposure=1000, min_start_time=0))
#     time.sleep(10)
#     print("3rd")
#     event_queue.put(MDAEvent(index={"t": 2}, exposure=1000, min_start_time=0))
#     time.sleep(15)
#     event_queue.put(stop_event)

# def run_mda_in_thread(_mmc, event_queue, stop_event): 
#     while True: 
#         if event_queue.empty():
#             time.sleep(0.1)
#         else:
#             current_event = event_queue.get()
#             if current_event == stop_event:
#                 break
#             _mmc.mda.run([current_event])


# if __name__ == "__main__":
#     STOP_EVENT = object()
#     event_queue = queue.Queue()
#     mmc = pymmcore_plus.CMMCorePlus()
#     mmc.loadSystemConfiguration()

#     mmc.mda.events.frameReady.disconnect()

#     @mmc.mda.events.frameReady.connect
#     def on_frame(img, event):
#         print(f'Frame {event.index} received: {img.shape}')
#     process = threading.Thread(target=run_controller_in_thread, args=(event_queue, STOP_EVENT))
#     process_mda_runner = threading.Thread(target=run_mda_in_thread, args=(mmc, event_queue, STOP_EVENT))
#     process_mda_runner.start()
#     process.start()
#     process.join()
#     process_mda_runner.join()


# import threading
# def run_controller_in_thread(event_queue, stop_event):
#     time.sleep(5)
#     print("1st")
#     event_queue.put(MDAEvent(index={"t": 0}, exposure=2000, min_start_time=0))
#     time.sleep(10)
#     print("2nd")
#     event_queue.put(MDAEvent(index={"t": 1}, exposure=1000, min_start_time=0))
#     time.sleep(10)
#     print("3rd")
#     event_queue.put(MDAEvent(index={"t": 2}, exposure=1000, min_start_time=0))
#     time.sleep(15)
#     event_queue.put(stop_event)

# def run_mda_in_thread(event_queue, stop_event): 
#     _mmc = pymmcore_plus.CMMCorePlus()
#     _mmc.loadSystemConfiguration()

#     _mmc.mda.events.frameReady.disconnect()

#     @_mmc.mda.events.frameReady.connect
#     def on_frame(img, event):
#         print(f'Frame {event.index} received: {img.shape}')
#     _mmc.run_mda(iter(event_queue.get, stop_event), block=True).join()


# if __name__ == "__main__":
#     STOP_EVENT = object()
#     event_queue = queue.Queue()

#     process = threading.Thread(target=run_controller_in_thread, args=(event_queue, STOP_EVENT))
#     process_mda_runner = threading.Thread(target=run_mda_in_thread, args=(event_queue, STOP_EVENT))
#     process_mda_runner.start()
#     process.start()
#     process.join()
#     process_mda_runner.join()


# import multiprocessing

# def run_controller_in_thread(event_queue, stop_event):
#     time.sleep(5)
#     print("1st")
#     event_queue.put(MDAEvent(index={"t": 0}, exposure=2000, min_start_time=0))
#     time.sleep(10)
#     print("2nd")
#     event_queue.put(MDAEvent(index={"t": 1}, exposure=1000, min_start_time=0))
#     time.sleep(10)
#     print("3rd")
#     event_queue.put(MDAEvent(index={"t": 2}, exposure=1000, min_start_time=0))
#     time.sleep(15)
#     event_queue.put(stop_event)

# def run_mda_in_thread(event_queue, stop_event): 
#     _mmc = pymmcore_plus.CMMCorePlus()
#     _mmc.loadSystemConfiguration()

#     _mmc.mda.events.frameReady.disconnect()

#     @_mmc.mda.events.frameReady.connect
#     def on_frame(img, event):
#         print(f'Frame {event.index} received: {img.shape}')
#     _mmc.run_mda(iter(event_queue.get, stop_event), block=True).join()


# if __name__ == "__main__":
#     STOP_EVENT = object()
#     event_queue = multiprocessing.Queue()

#     process = multiprocessing.Process(target=run_controller_in_thread, args=(event_queue, STOP_EVENT))
#     process_mda_runner = multiprocessing.Process(target=run_mda_in_thread, args=(event_queue, STOP_EVENT))
#     process_mda_runner.start()
#     process.start()
#     print("before join")
#     process.join()
#     process_mda_runner.join()
#     print("after join")



# import threading
# class IterableQueue(queue.Queue):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._lock = threading.Lock()
    
#     def _iter(self): 
#         return self
    
#     def __next__(self): 
#         with self._lock:
#             return self.get(block=True)
        
# def run_controller_in_thread(event_queue, stop_event):
#     time.sleep(5)
#     print("1st")
#     event_queue.put(MDAEvent(index={"t": 0}, exposure=2000, min_start_time=0))
#     time.sleep(10)
#     print("2nd")
#     event_queue.put(MDAEvent(index={"t": 1}, exposure=1000, min_start_time=0))
#     time.sleep(10)
#     print("3rd")
#     event_queue.put(MDAEvent(index={"t": 2}, exposure=1000, min_start_time=0))
#     time.sleep(15)
#     event_queue.put(stop_event)

# def run_mda_in_thread(event_queue, stop_event): 
#     _mmc = pymmcore_plus.CMMCorePlus()
#     _mmc.loadSystemConfiguration()

#     _mmc.mda.events.frameReady.disconnect()

#     @_mmc.mda.events.frameReady.connect
#     def on_frame(img, event):
#         print(f'Frame {event.index} received: {img.shape}')
#     _mmc.run_mda(iter(event_queue.get, stop_event))


# if __name__ == "__main__":
#     STOP_EVENT = object()
#     event_queue = IterableQueue()

#     process = threading.Thread(target=run_controller_in_thread, args=(event_queue, STOP_EVENT))
#     process_mda_runner = threading.Thread(target=run_mda_in_thread, args=(event_queue, STOP_EVENT))
#     process_mda_runner.start()
#     process.start()
#     print("before join")
#     process.join()
#     process_mda_runner.join()
#     print("after join")


from useq import MDAEvent
import queue
import pymmcore_plus
import time
import threading
def run_controller_in_thread(event_queue, stop_event):
    time.sleep(5)
    print("1st")
    event_queue.put(MDAEvent(index={"t": 0}, exposure=2000, channel={"config": "BF", "group": "WF_TTL"}))
    time.sleep(10)
    event_queue.put(MDAEvent(index={"t": 1}, exposure=1000, channel={"config": "BF", "group": "WF_TTL"}))
    time.sleep(10)
    event_queue.put(MDAEvent(index={"t": 2}, exposure=1000, channel={"config": "BF", "group": "WF_TTL"}))
    event_queue.put(stop_event)


if __name__ == "__main__":
    STOP_EVENT = object()
    event_queue = queue.Queue()
    mmc = pymmcore_plus.CMMCorePlus()
    mmc.mda.engine.use_hardware_sequencing = False
    mmc.run_mda(iter(event_queue.get, STOP_EVENT))
    mmc.loadSystemConfiguration("E:\\MicroManagerConfigs\\Ti2CicercoConfig_w_DMD_21_w_ttl.cfg")
    mmc.mda.events.frameReady.disconnect()

    @mmc.mda.events.frameReady.connect
    def on_frame(img, event):
        print(f'Frame {event.index} received: {img.shape}')
    process = threading.Thread(target=run_controller_in_thread, args=(event_queue, STOP_EVENT))
    process.start()
    process.join()

