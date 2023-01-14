import os
import csv
import json
import datetime
import argparse
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=pathlib.Path, required=False, default='messages/inbox/', help='Path to the messages/inbox/ folder')
parser.add_argument('-n', type=int, default=12, metavar='N', help='Plot the top %(metavar)s chats (default: %(default)s)')
parser.add_argument('--smooth', type=int, default=10, metavar='N', help='Smooth the data by averaging over %(metavar)s days (default: %(default)s)')

args = parser.parse_args()

if not args.folder.is_dir():
    print(f'Error: {args.folder} is not a directory')
    exit(1)

if args.n < 1:
    print(f'Error: n must be positive')
    exit(1)

if args.smooth < 1:
    print(f'Error: smoothing must be positive')
    exit(1)




chatlist = [p for p in args.folder.iterdir()]

time_div_ms = 1000 * 60 * 60 * 24   # 1 day

# Initialize array of # of messages by chat by time_div
num_messages = np.zeros((len(chatlist), 5000))

# Initialize array of chat names
chat_names = np.zeros(len(chatlist), dtype=object)

# Get current timestamp in milliseconds
curr_time = datetime.datetime.now().timestamp() * 1000

max_time = 0

# Get count of total files to read
total_files = 0
for chatfolder in chatlist:
    total_files += len([f for f in chatfolder.iterdir() if f.name.startswith('message')])

# Read all message files
with tqdm(total=total_files, desc='Reading message files') as pbar:
    # Iterate through chat folders
    for chat_idx, chatfolder in enumerate(chatlist):
        chatname = chatfolder.name.split('_')[0]

        # Get list of files with message data
        msg_files = [f for f in chatfolder.iterdir() if f.name.startswith('message')]
        if len(msg_files) == 0:
            continue

        # Iterate through JSON files
        for file_idx, file in enumerate(msg_files):
            data = json.load(open(file))

            # Ensure that chat is direct message, not group chat
            if len(data['participants']) != 2:
                pbar.update(1)
                continue

            # Record chat name
            chat_names[chat_idx] = data['participants'][0]['name']

            # Iterate through messages
            for msg in data['messages']:
                # Get timestamp
                timestamp = msg['timestamp_ms']
                # Get number of time_divs since message
                time = int((curr_time - timestamp) / time_div_ms)
                if time > max_time:
                    max_time = time
                # Add to count
                num_messages[chat_idx, time] += 1

            pbar.update(1)

# Truncate to actual number of time_divs
num_messages = num_messages[:, :max_time+1]

# Sum chats over time to get totals
chat_totals = np.sum(num_messages, axis=1)

# Sort chats and chat names by total messages
sort_idx = np.argsort(chat_totals)[::-1]
chat_totals = chat_totals[sort_idx]
chat_names = chat_names[sort_idx]
num_messages = num_messages[sort_idx, :]

# Smooth out num_messages by averaging over 10 time_divs
num_messages_smooth = np.zeros(num_messages.shape)
for i in range(num_messages.shape[0]):
    num_messages_smooth[i, :] = np.convolve(num_messages[i, :], np.ones(args.smooth)/args.smooth, mode='same')




print('Plotting...')

# Make notebook background white
plt.rcParams['figure.facecolor'] = 'white'

# Get actual timestamps for each week
timestamps = np.zeros(max_time+1)
for i in range(max_time+1):
    timestamps_ms = curr_time - (i * time_div_ms)
    # convert to datetime object
    dt = datetime.datetime.fromtimestamp(timestamps_ms / 1000)
    # convert to matplotlib timestamp
    timestamps[i] = mdates.date2num(dt)

# Plot top n chats over time
fig, ax = plt.subplots(1, 1, figsize=(20, 10), dpi=100)
for i in range(args.n):
    ax.plot(timestamps, num_messages_smooth[i, :], label=chat_names[i], linewidth=1)

# Set x-axis locator and formatter
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

ax.set_ylim(bottom=0)

ax.set_ylabel('Messages per day (running average)')
ax.set_title(f'Top {args.n} chats by number of messages')
ax.legend()

plt.savefig('top_chats.png', bbox_inches='tight')


# Plot total messages over time
fig, ax = plt.subplots(1, 1, figsize=(20, 10), dpi=100)
ax.plot(timestamps, np.sum(num_messages_smooth, axis=0), linewidth=1)

# Set x-axis locator and formatter
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

ax.set_ylim(bottom=0)

ax.set_ylabel('Messages per day (running average)')
ax.set_title(f'Total messages over time')

plt.savefig('total_messages.png', bbox_inches='tight')


# Get total messages from people outside of top n chats
other_messages = np.sum(chat_totals[args.n:])

# Plot pie chart of top n chats
fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=100)
ax.pie(
    np.concatenate([chat_totals[:args.n], [other_messages]]), labels=np.concatenate([chat_names[:args.n], ["Others"]]),
    rotatelabels=True,
)

plt.savefig('top_chats_pie.png', bbox_inches='tight')

print('Done')