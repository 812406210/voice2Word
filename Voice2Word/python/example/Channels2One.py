import wave
import io
from pydub import AudioSegment
import pydub

pydub.AudioSegment.converter = r'D:\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe'
def mp3_to_wav(mp3_path, wav_path):
    """
    MP3(ID3开头的文件)格式转为wav(RIFF开头文件)格式，并为单声道
    :param mp3_path:
    :param wav_path:
    :return:
    """
    with open(mp3_path, 'rb') as fh:
        data = fh.read()
    aud = io.BytesIO(data)
    sound = AudioSegment.from_file(aud, format='mp3')
    raw_data = sound._data

    size = len(raw_data)
    f = wave.open(wav_path, 'wb')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    f.setnframes(size)
    f.writeframes(raw_data)
    f.close()

    return wav_path

mp3_to_wav('E:\\python\\Demo\\Voice2Word\\python\\voices\\mp3\\test_3.mp3', 'E:\\python\\Demo\\Voice2Word\\python\\example\\channel.wav')



###CMD将MP3转为单通道wav文件  ： ffmpeg -i test_one.mp3 -acodec pcm_s16le -ac 1 -ar 16k clock.wav