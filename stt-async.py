import torch
import asyncio
import os
import numpy as np
import speech_recognition as sr
import time
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from queue import Queue
from time import sleep
from sys import platform, stdout
import threading

record_timeout = 2
phrase_timeout = 1
phrase_time = None
data_queue = Queue()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
	model_id,
	torch_dtype=torch_dtype,
	low_cpu_mem_usage=True,
	use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
	"automatic-speech-recognition",
	model=model,
	tokenizer=processor.tokenizer,
	feature_extractor=processor.feature_extractor,
	torch_dtype=torch_dtype,
	device=device,
	return_timestamps=True
)

print("Model loaded.\n")

def record_callback(_, audio:sr.AudioData) -> None:
	"""
	Threaded callback function to receive audio data when recordings finish.
	audio: An AudioData containing the recorded bytes.
	"""
	# Grab the raw bytes and push it into the thread safe queue.
	data = audio.get_raw_data()
	data_queue.put(data)

async def predict():
	start_time = time.time()
	audio_data = b''.join(data_queue.queue)
	data_queue.queue.clear()
	audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
	result = pipe(audio_np)
	text = result['text'].strip()
	
	print(text)
	print("--- %s seconds ---" % (time.time() - start_time))
	# stdout.write('\033[K' + text + '\r')

if __name__ == "__main__":
	recorder = sr.Recognizer()
	recorder.energy_threshold = 1000
	recorder.dynamic_energy_threshold = False

	source = sr.Microphone(sample_rate=16000)
	
	recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
	
	while True:
		asyncio.run(predict())

