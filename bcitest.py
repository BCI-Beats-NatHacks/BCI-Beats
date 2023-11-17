import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams

parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str,
help='serial port', required=False, default='COM3')
parser.add_argument('--board-id', type=int,
        help='board id, check docs to get a list of supported boards', required=False, default=1) ## Cyton board ID = 0)
args = parser.parse_args()

params = BrainFlowInputParams()
params.serial_port = args.serial_port
board = BoardShim(args.board_id, params)

board.prepare_session()
board.start_stream()
time.sleep(1)
board.stop_stream()
data = board.get_board_data() ## (250 Hz @ 1sec) ##
# board.release_session()

eeg_channels = board.get_eeg_channels(args.board_id)
sfreq = board.get_sampling_rate(args.board_id)
# eeg_names = BoardShim.get_eeg_names(args.board_id)

df = pd.DataFrame(np.transpose(data[:,1:]))
df_eeg = df[eeg_channels]
# df_eeg.columns = eeg_names
df_eeg.to_csv('data.csv', sep=',', index = False)





def TestPlotter(eeg_data, eeg_channels, sfreq, ch_names, event_codes):
    '''
    Uses code from Jupyter Lab notebook in CMPUT 624 to do a test plot for raw EEG data coming in
    Params:
        -eeg_data: raw eeg data
        -eeg_channels: # of eeg_channels
        -sfreq: sampling frequency
        -ch_names: array of channel names
        -event_codes: array info about what events occured 
    '''
    # get data
    ch_types = ['eeg'] * len(eeg_channels)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)

    #plot without any modifications
    raw.plot_psd(average=True)

    # re-referencing and event attaching
    raw.set_eeg_reference(ref_channels='average') # subtract mean from data for re-referencing
    events = mne.find_events(raw) # find events (mne can look through stim channel, find timings and event codes and return matrix of results)
    events[:, 2] = event_codes  # add event codes to event structure

    # filter
    eeg_run1_filtered = raw.copy().filter(l_freq=0.01, h_freq=100) # bandpass 
    eeg_run1_filtered = eeg_run1_filtered.notch_filter(freqs=60) # take out 50 hz noise using notch filter
    eeg_run1_filtered.compute_psd(fmax=120).plot(); 

    # epoching
    epochs = mne.Epochs(eeg_run1_filtered, events, tmin=-0.1, tmax=0.7, picks=eeg_channels)
    condition_map = {'closed': 0, 'open': 1}
    epochs.event_id = condition_map # put events into human readable form

    # downsample to reduce temporal precision
    # not needed here because ours is already 250 Hz which is small enough

    # do we need to get rid of baseline? - mne.Epochs does it automatically

    # Averaging (of all trials)
    open_avg = epochs['open'].average()
    closed_avg = epochs['closed'].average()
    open_avg.plot(spatial_colors=True)
    closed_avg.plot(spatial_colors=True)

    # look at differences between resulting averages
    open_vs_closed = mne.combine_evoked([open_avg, closed_avg], weights=[1,-1])
    open_vs_closed.plot(spatial_colors=True)
    open_vs_closed.plot_joint()

