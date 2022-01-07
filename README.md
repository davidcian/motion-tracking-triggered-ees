# motion-tracking-triggered-ees
This software uses video and FLIR cameras to trigger relevant epidural electrical stimulation (EES) in a patient in a closed feedback loop. This allows spinal cord injury (SCI) patients to regain some mobility, provided they have an epidural implant.

## Using the software

### Setup

You can use the `requirements.txt` file with `pip` to install all required dependencies except OpenSim. OpenSim requires additional steps, which can be found [here](https://simtk-confluence.stanford.edu:8443/display/OpenSim/Scripting+in+Python).

### Running the GUI

In order to run the GUI, run the main GUI script like this: `python gui/native_gui.py`. Depending on the video source and on whether you provide a depth input as well, you will need to specify additional command line options. You can find examples on how to run the GUI script in the three Windows `.bat` scripts.
