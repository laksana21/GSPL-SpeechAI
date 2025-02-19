# IMPORTANT!
# Clone the models and voices from https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX

import os
import pyaudio
import time
import numpy as np
from onnxruntime import InferenceSession
from kokoro_onnx.tokenizer import Tokenizer
import scipy.io.wavfile as wavfile
import sounddevice as sd

def main():
	start_time = time.time()
	# Text you want to speak
	text_input = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book."

	tokenizer = Tokenizer()
	tokens = tokenizer.tokenize(tokenizer.phonemize(text_input))

	# Context length is 512, but leave room for the pad token 0 at the start & end
	assert len(tokens) <= 510, len(tokens)

	# Style vector based on len(tokens), ref_s has shape (1, 256)
	voices = np.fromfile('./Kokoro-82M-v1.0-ONNX/voices/af.bin', dtype=np.float32).reshape(-1, 1, 256)
	ref_s = voices[len(tokens)]

	# Add the pad ids, and reshape tokens, should now have shape (1, <=512)
	tokens = [[0, *tokens, 0]]

	model_name = './Kokoro-82M-v1.0-ONNX/onnx/model.onnx'
	sess = InferenceSession(model_name)

	audio = sess.run(None, dict(
		input_ids=tokens,
		style=ref_s,
		speed=np.ones(1, dtype=np.float32),
	))[0]

	print("--- %s seconds ---" % (time.time() - start_time))

	#sd.play(audio[0], 24000)
	#sd.wait()

	wavfile.write('audio.wav', 24000, audio[0])


if __name__ == "__main__":
	main()

