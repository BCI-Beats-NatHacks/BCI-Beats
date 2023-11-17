import time
import argparse
import numpy as np
import pandas as pd

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