#!/usr/bin/env python3

from vosk import Model, KaldiRecognizer, SetLogLevel
import os
import subprocess
import json
import re

class VoiceRecognition:


    def __init__(self,fileName):
        """
        初始化
        :param fileName:
        """
        SetLogLevel(0)
        self.rec , self.process = self.judgeCondition(fileName)

    def judgeCondition(self,fileName):
        """
        条件判断，初始化模型
        :param fileName: 文件名
        :return:
        """
        if not os.path.exists("model"):
            print(
                "Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
            exit(1)

        sample_rate = 16000
        model = Model("model")
        rec = KaldiRecognizer(model, sample_rate)
        process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                                    fileName,
                                    '-ar', str(sample_rate), '-ac', '1', '-f', 's16le', '-'],
                                   stdout=subprocess.PIPE)

        return rec,process

    def getRealData(self):

        """
        获取数据
        :return:
        """
        totalResult = []
        while True:
            data = self.process.stdout.read(4000)
            if len(data) == 0:
                break
            if self.rec.AcceptWaveform(data):
                jres = json.loads(self.rec.Result())
                print("jres:", jres['text'])
                totalResult.append(self.clean_space(jres['text']))
            else:
                pass
                # print("result_two",rec.PartialResult())  # 实时结果

        lastJres = json.loads(self.rec.FinalResult())
        print("lastJres:", lastJres['text'])
        totalResult.append(self.clean_space(lastJres['text']))
        print("totalResult:", totalResult)

        return ",".join(totalResult)

    def clean_space(self,originalData):
        match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
        should_replace_list = match_regex.findall(originalData)
        order_replace_list = sorted(should_replace_list, key=lambda i: len(i), reverse=True)
        for i in order_replace_list:
            if i == u' ':
                continue
            new_i = i.strip()
            originalData = originalData.replace(i, new_i)
        return originalData



