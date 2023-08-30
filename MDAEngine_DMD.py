from typing import Iterable, Iterator
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import MDAEngine
from useq import MDASequence
from useq._mda_sequence import MDASequence
import useq
import time
from dmd import DMD

class MDAEngine_DMD(MDAEngine):
    """An MDA engine that expects acquisition events to have a stim_mask property 
    in the metadata, that can be sent to the DMD. E.g. np.ones((800,600)) to get the whole FOV illuminated."""
    _mmc = None

    def __init__(self, dmd: DMD):
        self.dmd = dmd
        super().__init__(dmd.mmc)
        self._mmc = dmd.mmc

    def exec_event(self, event: useq.MDAEvent) -> object:
        # do some custom pre-execution
        if self.dmd.test_mode == False:
            if 'stim_mask' in event.metadata:
                stim_mask = event.metadata['stim_mask']
                #print(stim_mask.shape)
                #set the DMD pixels
                self.dmd.display_mask(stim_mask)
            else:
                print('No DMD mask found')
                self.dmd.all_on()

        result = super().exec_event(event=event)

        # my_post_camera_hook_fn() ... or implement `self.teardown_event()`
        return result
    
    def setup_event(self, event: useq.MDAEvent) -> None:
        # pre_hardware_hook_fn()

        super().setup_event(event)
        # post_hardware_hook_fn()

