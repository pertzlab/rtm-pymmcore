import yaml
import pymmcore
# Load local_config.yaml file

def load_config(mmc:pymmcore.CMMCore, config_file='/local/local_config.yaml'):
    """
    Load the configuration from local_config.yaml file and set the default devices in pymmcore.

    Args:
        mmc (pymmcore.CMMCore): The CMMCore object.
        config_file (str, optional): The path to the configuration file. 
                                        Defaults to '/rtm-pymmcore-local/local_config.yaml'.
    Returns:
        None
    """

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Set default devices in pymmcore
    mmc.setCameraDevice(config['camera'])
    mmc.setShutterDevice(config['shutter'])
    mmc.setXYStageDevice(config['xy_stage'])
    mmc.setFocusDevice(config['z_stage'])
    mmc.setAutoFocusDevice(config['autofocus'])


    slm_device = config['slm_device']
    if slm_device is not None:

        mmc.setSLMDevice(slm_device)

    # Load 3-level settings in pymmcore
    for setting in config['settings']:
        device = setting['device']
        for prop in setting['properties']:
            property_label = prop['property']
            property_value = prop['value']
            mmc.setProperty(device, property_label, property_value)

