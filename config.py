import time
'''
CONFIGS
'''
import torch.nn as nn

BATCH_SIZE = 128
BATCH_TEST = BATCH_SIZE
EPOCHS = 90
learning_rate = 0.0001
loss_function = nn.MSELoss()
CNN_CONFIG = 'cnn_configs/default.json'
REPORT_FILE = f"report-{int(time.time())}.csv"