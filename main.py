
#Add MDA event to acquisition queue, and get multi-channel frame back

from matplotlib import pyplot as plt
from napari import Viewer
import numpy as np
from queue import Queue
from pymmcore_plus import CMMCorePlus
from useq import MDAEvent
import useq
from useq._channel import Channel
from dmd import DMD
from MDAEngine_DMD import MDAEngine_DMD
from pymmcore_plus.mda import MDAEngine

from controller import Controller, Analyzer


def main():
    # Setup the MM Core
    mmc = CMMCorePlus()
    mmc.loadSystemConfiguration()
    #mmc.mda.set_engine(MDAEngine_DMD)
    mmc.mda.set_engine(MDAEngine_DMD(mmc))


    # Apply the custom acquisition engine
    # Register the custom engine with the runner
    core = CMMCorePlus.instance()
    dmd = DMD(mmc)
    #core.mda.set_engine(MDAEngine_DMD(core))

    # create the Queue that will hold the MDAEvents
    q = Queue()

    # Setup the controller and analyzers
    analyzer = Analyzer()
    controller = Controller(analyzer, mmc, q, dmd)

    # Start the acquisition
    controller.run()

if __name__ == "__main__":
    main()




