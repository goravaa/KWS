# main.py (Final Architecture)
#
# This script uses a queue to decouple audio capture from model inference,
# preventing 'input overflow' errors and creating a robust real-time system.

import sounddevice as sd
import numpy as np
import onnxruntime as ort
import yaml
import torchaudio
import torch
import logging
import os
import queue # Used to communicate between the audio callback and the main thread

def setup_logging():
    """Configures logging for the application."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler()])

def main():
    setup_logging()

    # --- 1. Load Configuration and Models ---
    logging.info("Loading configuration and models...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_dir = config['training']['model_output_path']
    
    # --- IMPORTANT: Use the quantized model for best performance ---
    use_quantized = False
    model_filename = 'kws_model.quant.onnx' if use_quantized else 'kws_model.onnx'
    onnx_model_path = os.path.join(model_dir, model_filename)
    checkpoint_path = os.path.join(model_dir, 'best_kws_model.pth')

    try:
        ort_session = ort.InferenceSession(onnx_model_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        class_labels = checkpoint['classes']
    except FileNotFoundError as e:
        logging.error(f"Error loading files: {e}. Please ensure train.py and quantize_model.py have been run.")
        return

    logging.info(f"Loaded ONNX model: {onnx_model_path}")
    logging.info(f"Loaded classes: {class_labels}")

    # --- 2. Setup Audio Processing ---
    audio_cfg = config['audio']
    detection_cfg = config['detection']
    n_mels = audio_cfg.get('n_mels', 2 * audio_cfg['n_mfcc'])
    
    input_details = ort_session.get_inputs()[0]
    input_shape = input_details.shape
    n_mfcc = input_shape[2]
    time_steps = input_shape[3]

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=audio_cfg['sample_rate'], n_mfcc=n_mfcc,
        melkwargs={'n_fft': audio_cfg['n_fft'], 'hop_length': audio_cfg['hop_length'], 
                   'win_length': audio_cfg['win_length'], 'n_mels': n_mels}
    )
    
    mfcc_buffer = np.zeros((n_mfcc, time_steps), dtype=np.float32)
    cooldown_counter = 0
    try:
        background_label_index = class_labels.index('_background_noise_')
    except ValueError:
        logging.error("'_background_noise_' not found in class labels.")
        return

    # --- 3. Setup Thread-Safe Queue and Audio Callback ---
    # The queue will hold chunks of audio data from the callback
    audio_queue = queue.Queue()

    # The new callback is extremely fast: it just puts data in the queue.
    def audio_callback(indata, frames, time, status):
        if status:
            logging.warning(status)
        audio_queue.put(indata.copy())

    # --- 4. Start Audio Stream ---
    blocksize = audio_cfg['hop_length'] * 20 # Give a larger buffer
    stream = sd.InputStream(samplerate=audio_cfg['sample_rate'], channels=1, dtype='float32',
                            blocksize=blocksize, callback=audio_callback)
    
    logging.info("Listening for keywords... Press Ctrl+C to stop.")

    # --- 5. Main Processing Loop ---
    with stream:
        try:
            while True:
                # Pull a chunk of audio from the queue
                audio_chunk = audio_queue.get()
                
                # --- This is the same processing logic as before, but now in the main loop ---
                waveform = torch.from_numpy(audio_chunk.T.astype(np.float32))
                new_mfccs_3d = mfcc_transform(waveform)
                new_mfccs_2d = new_mfccs_3d.squeeze(0)
                
                num_new_frames = new_mfccs_2d.shape[1]
                if num_new_frames > mfcc_buffer.shape[1]:
                    mfcc_buffer[:] = new_mfccs_2d.numpy()[:, -mfcc_buffer.shape[1]:]
                else:
                    mfcc_buffer[:] = np.concatenate(
                        (mfcc_buffer[:, num_new_frames:], new_mfccs_2d.numpy()), axis=1
                    )
                
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    continue
                
                model_input = np.expand_dims(mfcc_buffer, axis=(0, 1)).astype(np.float32)
                
                ort_inputs = {ort_session.get_inputs()[0].name: model_input}
                ort_outs = ort_session.run(None, ort_inputs)
                
                probabilities = torch.nn.functional.softmax(torch.tensor(ort_outs[0]), dim=1).squeeze().numpy()
                
                top_prob_idx = np.argmax(probabilities)
                top_prob = probabilities[top_prob_idx]

                if top_prob_idx != background_label_index and top_prob > detection_cfg['threshold']:
                    detected_keyword = class_labels[top_prob_idx]
                    print(f"\033[92m--- DETECTED '{detected_keyword.upper()}' (Confidence: {top_prob:.2f}) ---\033[0m")
                    cooldown_counter = detection_cfg['cooldown_frames']
        
        except KeyboardInterrupt:
            logging.info("Stopping...")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()