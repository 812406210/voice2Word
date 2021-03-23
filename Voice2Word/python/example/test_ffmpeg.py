#!/usr/bin/env python3

from vosk import Model, KaldiRecognizer, SetLogLevel
import os
import subprocess
import  sys
import json
SetLogLevel(0)

if not os.path.exists("model"):
    print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit (1)

sample_rate=16000
model = Model("model")
rec = KaldiRecognizer(model, sample_rate)

# process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
#                             'demo2.mp3',
#                             '-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'],
#                             stdout=subprocess.PIPE)
process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                            sys.argv[1],
                            '-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'],
                            stdout=subprocess.PIPE)

totalResult = []
while True:
    data = process.stdout.read(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        jres = json.loads(rec.Result())
        print("jres:",jres['text'])
        totalResult.append(jres['text'])
    else:
        pass
        # print("result_two",rec.PartialResult())  # 实时结果

lastJres = json.loads(rec.FinalResult())
print("lastJres:",lastJres['text'])
totalResult.append(lastJres['text'])

print("totalResult:",totalResult)