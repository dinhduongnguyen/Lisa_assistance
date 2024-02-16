import numpy as np
import pyaudio
import librosa
import wave 
import threading
from threading import Event
import time
import os
from tflite_runtime.interpreter import Interpreter
import lirc
import RPi.GPIO as GPIO
led_pin = 16
# GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(16, GPIO.OUT, initial=GPIO.LOW)
client = lirc.Client()
client.list_remotes()
client.list_remote_keys('RGBLED_REMOTE')
class Listener:

    def __init__(self, sample_rate=16000, record_seconds=1):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk , exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nWake Word Engine is now listening... \n")

def encode_single_sample(file):
    audio,rate = librosa.load(file,sr=16000)
    mfccs = librosa.feature.mfcc(y=audio,n_mfcc=12,sr=16000)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
    features = features.transpose()
    means = np.mean(features, axis=1, keepdims=True)
    stddevs = np.std(features, axis=1, keepdims=True)
    mfcc_feature = (features - means) / (stddevs + 1e-10)
    mfccs_feature = np.zeros((34,36))
    mfccs_feature[:mfcc_feature.shape[0], :mfcc_feature.shape[1]]=mfcc_feature
    mfccs_feature = mfccs_feature.reshape(1,34,36)
    return mfccs_feature

interpreter_wakeword = Interpreter(model_path='aug_wakeword_quantizev1.tflite')
interpreter_wakeword.allocate_tensors()
input_wakeword_details = interpreter_wakeword.get_input_details()
output_wakeword_details = interpreter_wakeword.get_output_details()

interpreter_command = Interpreter(model_path='command_quantize_aug_12_v0.tflite')
interpreter_command.allocate_tensors()
input_command_details = interpreter_command.get_input_details()
output_command_details = interpreter_command.get_output_details()

def decode_predictions(pred):
    index = list(pred).index(max(pred))
    if index ==0:
        return 'den bat'
    elif index == 1:
        return 'den chuyen mau'
    elif index ==2:
        return 'den giam sang'
    elif index ==3:
        return 'den tang sang'
    elif index == 4:
        return 'den tat'
    elif index ==5:
        return 'noise'

class VoiceAssistanceEngine:

    def __init__(self):
        self.listener = Listener(sample_rate=16000, record_seconds=1)
        self.towakeword = 0
        self.audio_q = list()
        self.flag = 0

    def save(self, waveforms, fname="wakeword_temp"):
        wf = wave.open(fname, "wb")
        # set the channels
        wf.setnchannels(1)
        # set the sample format
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        # set the sample rate
        wf.setframerate(16000)
        # write the frames as bytes
        wf.writeframes(b"".join(waveforms))
        # close the file
        wf.close()
        return fname


    def predict(self, audio):
        fname = self.save(audio)
        feature = encode_single_sample(fname)
        if self.flag == 0:
            in_tensor = np.float32(feature)
            interpreter_wakeword.set_tensor(input_wakeword_details[0]['index'], in_tensor)
            interpreter_wakeword.invoke()
            output_wakeword = interpreter_wakeword.get_tensor(output_wakeword_details[0]['index'])
            out_wakeword = output_wakeword[0][0]
            if out_wakeword>0.5:
                self.flag = 1
                GPIO.output(led_pin, GPIO.HIGH)
                time.sleep(1)
                GPIO.output(led_pin, GPIO.LOW)
                pred = 'lisa oi'
            else:
                pred = 'noise'
            return pred
        else:
            in_tensor = np.float32(feature) 
            interpreter_command.set_tensor(input_command_details[0]['index'], in_tensor)
            interpreter_command.invoke()
            output_data = interpreter_command.get_tensor(output_command_details[0]['index'])
            out = output_data
            results = decode_predictions(out[0])
            if results == 'noise':
                self.towakeword += 1
            else: 
                self.towakeword = 0
            if self.towakeword == 30:
                GPIO.output(led_pin, GPIO.HIGH)
                time.sleep(1)
                GPIO.output(led_pin, GPIO.LOW)
                self.flag = 0
                self.towakeword = 0
            return results
    
    
    def inference_loop(self, action):
        while True:
            if len(self.audio_q) > 15:  # remove part of stream
                diff = len(self.audio_q) - 15
                for _ in range(diff):
                    self.audio_q.pop(0)
                action(self.predict(self.audio_q))
            elif len(self.audio_q) == 15:
                action(self.predict(self.audio_q))
            time.sleep(0.7)

    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop,
                                    args=(action,), daemon=True)
        thread.start()

class DemoAction:

    def __init__(self):
        self.asr_results = ""

    def __call__(self, x):
        self.asr_results = x
        trascript = self.asr_results
        print(trascript)
        if trascript == 'den bat':
            client.send_once('RGBLED_REMOTE', 'POWER_ON', 0)
        elif trascript == 'den tat':
            client.send_once('RGBLED_REMOTE', 'POWER_OFF', 0)
        elif trascript == 'den tang sang':
            client.send_once('RGBLED_REMOTE', 'POWER_UP', 0)
            client.send_once('RGBLED_REMOTE', 'POWER_UP', 0)
            client.send_once('RGBLED_REMOTE', 'POWER_UP', 0)
        elif trascript == 'den giam sang':
            client.send_once('RGBLED_REMOTE', 'POWER_DOWN', 0)
            client.send_once('RGBLED_REMOTE', 'POWER_DOWN', 0)
            client.send_once('RGBLED_REMOTE', 'POWER_DOWN', 0)
        elif trascript == 'den chuyen mau':
            client.send_once('RGBLED_REMOTE', 'COLOR_WHITE', 0)
if __name__ == "__main__":
    GPIO.output(led_pin, GPIO.LOW)
    asr_engine = VoiceAssistanceEngine()
    action = DemoAction()

    asr_engine.run(action)
    threading.Event().wait()