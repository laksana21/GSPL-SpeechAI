import asyncio
import queue
import re
import sys
import threading
import time
from typing import Dict, List, Tuple

import pyaudio

import numpy as np
from faster_whisper import WhisperModel

MODEL_TYPE = "large-v3"
RUN_TYPE = "cpu"  # "cpu" or "gpu"

# For CPU usage (https://github.com/SYSTRAN/faster-whisper/issues/100#issuecomment-1492141352)
NUM_WORKERS = 10
CPU_THREADS = 4

# For GPU usage
GPU_DEVICE_INDICES = [0, 1, 2, 3]
VAD_FILTER = True

# Audio settings
STEP_IN_SEC: int = 1
LENGHT_IN_SEC: int = 6
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE

# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80

# This queue holds all the 1-second audio chunks
audio_queue = queue.Queue()

# This queue holds all the chunks that will be processed together
# If the chunk is filled to the max, it will be emptied
length_queue = queue.Queue(maxsize=LENGHT_IN_SEC)

def create_whisper_model() -> WhisperModel:
	if RUN_TYPE.lower() == "gpu":
		whisper = WhisperModel(
			MODEL_TYPE,
			device="cuda",
			compute_type="float16",
			device_index=GPU_DEVICE_INDICES,
			download_root="./models"
		)
		
	elif RUN_TYPE.lower() == "cpu":
		whisper = WhisperModel(
			MODEL_TYPE,
			device="cpu",
			compute_type="int8",
			num_workers=NUM_WORKERS,
			cpu_threads=CPU_THREADS,
			download_root="./models"
		)
		
	else:
		raise ValueError(f"Invalid model type: {RUN_TYPE}")
		
	print("Loaded model")
	
	return whisper
	
model = create_whisper_model()

def execute_blocking_whisper_prediction(model: WhisperModel, audio_data_array: np.ndarray, language_code: str = "") -> Tuple[str, str, float]:
	language_code = language_code.lower().strip()
	segments, info = model.transcribe(
		audio_data_array,
		language=language_code if language_code != "" else None,
		beam_size=5,
		vad_filter=VAD_FILTER,
		vad_parameters=dict(min_silence_duration_ms=500)
	)
	
	segments = [s.text for s in segments]
	transcription = " ".join(segments)
	transcription = transcription.strip()
	
	return transcription
	
def predict(audio_data: bytes, language_code: str = "") -> str:
	# Convert the audio bytes to a NumPy array
	audio_data_array: np.ndarray = np.frombuffer(audio_data, np.int16).astype(np.float32) / 255.0

	try:
		# Run the prediction on the audio data
		result = execute_blocking_whisper_prediction(model, audio_data_array, language_code)

	except Exception as e:
		print(e)
		result = e

	return result
	
def producer_thread():
	audio = pyaudio.PyAudio()
	stream = audio.open(
		format=pyaudio.paInt16,
		channels=NB_CHANNELS,
		rate=RATE,
		input=True,
		frames_per_buffer=CHUNK
	)

	print("-" * 80)
	print("Microphone initialized, recording started...")
	print("-" * 80)
	print("TRANSCRIPTION")
	print("-" * 80)

	while True:
		audio_data = b""
		for _ in range(STEP_IN_SEC):
			chunk = stream.read(RATE)
			audio_data += chunk

		audio_queue.put(audio_data)

def consumer_thread(stats):
	while True:
		if length_queue.qsize() >= LENGHT_IN_SEC:
			with length_queue.mutex:
				length_queue.queue.clear()
				print()

		audio_data = audio_queue.get()
		transcription_start_time = time.time()
		length_queue.put(audio_data)

		# Concatenate audio data in the lenght_queue
		audio_data_to_process = b""
		for i in range(length_queue.qsize()):
			# We index it so it won't get removed
			audio_data_to_process += length_queue.queue[i]

		try:
			transcription = str(predict(audio_data_to_process))
			# remove anything from the text which is between () or [] --> these are non-verbal background noises/music/etc.
			transcription = re.sub(r"\[.*\]", "", transcription)
			transcription = re.sub(r"\(.*\)", "", transcription)
		except Exception as e:
			transcription = e

		transcription_end_time = time.time()

		# We do this for the more clean visualization (when the next transcription we print would be shorter then the one we printed)
		transcription_to_visualize = transcription.ljust(MAX_SENTENCE_CHARACTERS, " ")

		transcription_postprocessing_end_time = time.time()

		sys.stdout.write('\033[K' + transcription_to_visualize + '\r')

		audio_queue.task_done()

		overall_elapsed_time = transcription_postprocessing_end_time - transcription_start_time
		transcription_elapsed_time = transcription_end_time - transcription_start_time
		postprocessing_elapsed_time = transcription_postprocessing_end_time - transcription_end_time
		stats["overall"].append(overall_elapsed_time)
		stats["transcription"].append(transcription_elapsed_time)
		stats["postprocessing"].append(postprocessing_elapsed_time)
		
if __name__ == "__main__":
	stats: Dict[str, List[float]] = {"overall": [], "transcription": [], "postprocessing": []}

	producer = threading.Thread(target=producer_thread)
	producer.start()

	consumer = threading.Thread(target=consumer_thread, args=(stats))
	consumer.start()

	try:
		producer.join()
		consumer.join()
	except KeyboardInterrupt:
		print("Exiting...")
		# print out the statistics
		print("Number of processed chunks: ", len(stats["overall"]))
		print(f"Overall time: avg: {np.mean(stats['overall']):.4f}s, std: {np.std(stats['overall']):.4f}s")
		print(
			f"Transcription time: avg: {np.mean(stats['transcription']):.4f}s, std: {np.std(stats['transcription']):.4f}s"
		)
		print(
			f"Postprocessing time: avg: {np.mean(stats['postprocessing']):.4f}s, std: {np.std(stats['postprocessing']):.4f}s"
		)
		# We need to add the step_in_sec to the latency as we need to wait for that chunk of audio
		print(f"The average latency is {np.mean(stats['overall'])+STEP_IN_SEC:.4f}s")
	
