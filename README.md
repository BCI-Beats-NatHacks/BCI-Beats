<p align="center">
  <img src="https://github.com/BCI-Beats-NatHacks/BCI-Beats/assets/140631194/d868c503-2faa-4f02-8415-f52d0247aa77" alt="Logo" width="500"/>
</p>

# BCI-Beats
A BCI music composition app, designed for NatHacks2023. Find us on [Devpost](https://devpost.com/software/bci-beats).

# Contributors
- Jared Gourley
- Jake Hennig
- Kaiden Mastel
- Mona Safari
- Nicholas Mellon
- Kai Luedemann

# Setup
1. Clone the BCI Beats GitHub repository
2. Clone the OpenBCI LSL repository using

  ```git submodule add https://github.com/openbci-archive/OpenBCI_LSL.git```

3. Create a virtual environment using
   
   ```python3 -m venv venv```
   
4. Activate the virtual environment and install the requirements with

   ```pip3 install -r requirements.txt```
   
5. Create a Unity project for the repository
6. Install BCI Essentials Unity into the Unity project by following these [instructions](https://github.com/kirtonBCIlab/bci-essentials-unity#install-into-unity-project)
7. Connect your OpenBCI headset to your computer

# Instructions
1. Run the LabStreamingLayer using

  ```python3 OpenBCI_LSL/openbci_lsl.py --stream```
  
2. Start your Unity project
3. Run the backend program (in virtual environment) with

   ```python3 p300_unity_backend.py```
