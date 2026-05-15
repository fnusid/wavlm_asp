import torch
import torchaudio
from pvad_model import pVAD_module, PreEmphasis
import argparse
import queue
import time
import threading
from collections import deque

import numpy as np
import soundfile as sf

import torch.nn.functional as F
import torch.nn as nn
import onnxruntime as ort

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

RATE=16_000

class LogMelFrontend(nn.Module):
    """
    Input:
        wav: [B, T]
    Output:
        logmel: [B, 80, Frames]
    """
    def __init__(self):
        super().__init__()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,
                hop_length=160,
                f_min=20,
                f_max=7600,
                window_fn=torch.hamming_window,
                n_mels=80,
            ),
        )
    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.torchfbank(wav) + 1e-6
        x = x.log()
        x = x - torch.mean(x, dim=-1, keepdim=True)
        return x
    
def load_pvad_from_lightning_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    pvad_state = {}
    for k, v in state.items():
        if k.startswith("pvad."):
            pvad_state[k.replace("pvad.", "", 1)] = v
        
    if len(pvad_state) == 0:
        print("No pVAD.* found in the checkpoint.")
        print("Available keys in ckpt:")
        for k in state.keys():
            if "pvad" in k.lower():
                print(" ", k)
        raise RuntimeError("pVAD state not found in checkpoint.")
    

    model = pVAD_module(hidden_dim=256+80)
    missing, unexpected = model.load_state_dict(pvad_state, strict=False)

    print(f"[pVAD CKPT] loaded. missing={len(missing)} unexpected={len(unexpected)}")

    if missing:
        print("Missing keys:")
        for k in missing:
            print(" ", k)

    if unexpected:
        print("Unexpected keys:")
        for k in unexpected:
            print(" ", k)

    
    model.to(device)
    model.eval()
    return model

class QuantizedECAPAEmbedder:
    def __init__(self, onnx_path,
                 frontend_device="cpu",
                 chunk_sec=3.0,
                 providers=None,):
        self.onnx_path = onnx_path
        self.chunk_sec = float(chunk_sec)
        self.chunk_samples = int(self.chunk_sec * RATE)
        self.frontend_device = torch.device(frontend_device)
        self.frontend = LogMelFrontend().to(self.frontend_device).eval()

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.sess = ort.InferenceSession(self.onnx_path, providers=providers)
        print(f"[ONNX ECAPA Loaded: {onnx_path}] Providers: {self.sess.get_providers()}")

    @torch.no_grad()
    def wav_to_features(self, wav_1d):
        '''
        wav_1d: numpy [T]
        returns:
            feat_np : [1, 80, Frames]
        '''
        wav = torch.from_numpy(wav_1d.astype(np.float32))
        wav = wav.unsqueeze(0).to(self.frontend_device)  # [1, T]

        feat = self.frontend(wav)  # [1, 80, Frames]
        feat_np = feat.cpu().numpy().astype(np.float32)
        return feat_np
    
    def run_features(self, feat_np):
        '''
        feat_np: [1, 80, Frames]
        returns:
            emb_np: [1, 256]
        '''
        emb_np = self.sess.run(['embeddings'], {'features': feat_np.astype(np.float32)})[0]  # [1, 256]
        return emb_np.astype(np.float32)
    
    def compute_embeddings_from_buffer(self, audio_buffer, num_chunks=0):
        '''
        Since my ONNX model was exported for ~3s features, I samples several
        3s chunks from the rolling buffer, run quantized ECAPA, normalize and avg slot wise

        audio_buffer: [T] numpy, last 60s

        returns:
            avg_emb: [1, 2, 256] torch
        '''

        audio = np.asarray(audio_buffer, dtype=np.float32)

        if len(audio) < self.chunk_samples:
            # Not enough audio for even one chunk
            audio = np.pad(audio, (0, self.chunk_samples - len(audio)), mode='constant')

        max_start = max(0, len(audio) - self.chunk_samples)

        if num_chunks <= 1 or max_start == 0:
            starts = [max_start]
        else:
            starts = np.linspace(0, max_start, num_chunks).astype(int).tolist()
        
        embs = []
        for st in starts:
            chunk = audio[st:st + self.chunk_samples]
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)), mode='constant')
            
            feat_np = self.wav_to_features(chunk)  # [1, 80, Frames]
            emb_np = self.run_features(feat_np)     # [1, 2, 256]

            emb = torch.from_numpy(emb_np).float() # [1, 2, 256]
            emb = F.normalize(emb, dim=-1)             # L2 normalize
            embs.append(emb)

        embs = torch.cat(embs, dim=0)  # [N, 2, 256]

        emb_avg = embs.mean(dim=0, keepdim=True)  # [1, 2, 256]
        emb_avg = F.normalize(emb_avg, dim=-1)  # L2 normalize
        return emb_avg
    


class MicAudioStream:
    def __init__(self, sample_rate=16_000, block_sec=0.25, device_id=None):
        import sounddevice as sd
        self.sd = sd
        self.sample_rate = sample_rate
        self.block_size = int(block_sec * sample_rate)
        self.device_id = device_id
        self.q = queue.Queue()
        self.stream = None

    def callback(self, indata, frames, time_info, status):
        if status:
            print("[sounddevice callback] Status:", status)
        x = indata[:, 0].copy().astype(np.float32)  # Use the first channel
        self.q.put(x)

    def start(self):
        self.stream = self.sd.InputStream(
            channels = 1,
            samplerate = self.sample_rate,
            blocksize = self.block_size,
            callback = self.callback,
            device = self.device_id,
        )
        self.stream.start()
        print(f"[MicAudioStream] Started with sample_rate={self.sample_rate}, block_sec={self.block_size / self.sample_rate:.2f}s, device_id={self.device_id}")

    def read_available(self):
        chunks = []

        while True:
            try:
                chunks.append(self.q.get_nowait())
            except queue.Empty:
                break
        
        if len(chunks) == 0:
            return None
        
        return np.concatenate(chunks).astype(np.float32)
    
    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            print("[MicAudioStream] Stopped.")


class WavAudioStream:
    '''
    Simulates real time streaming from a wav file.
    Useful on GPU servers with no microphone
    '''

    def __init__(self, wav_path, sample_rate=16_000, block_sec=0.25, loop=True):
        self.wav_path = wav_path
        self.sample_rate = sample_rate
        self.block_size = int(block_sec * sample_rate)
        self.loop = loop

        wav, sr = sf.read(wav_path, dtype='float32')

        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # Convert to mono by averaging channels
        
        if sr != sample_rate:
           
            wav_t = torch.from_numpy(wav).float()

            wav_t = torchaudio.functional.resample(wav_t, sr, sample_rate)

            wav = wav_t.numpy().astype(np.float32)

        self.wav = wav.astype(np.float32)
        self.pos = 0
        self.last_time = None

        print(f"[WavAudioStream] Loaded '{wav_path}' with sample_rate={sample_rate}, block_sec={block_sec:.2f}s, total_duration={len(wav) / sample_rate:.2f}s, loop={loop}")

    def start(self):
        self.last_time = time.time()
        print(f"[WavAudioStream] Started streaming from '{self.wav_path}'")

    def read_available(self):
        now = time.time()

        if self.last_time is None:
            self.last_time = now
            return None

        elapsed = now - self.last_time

        if elapsed < 0.2:
            return None

        self.last_time = now
        n = self.block_size
        if self.pos >= len(self.wav):
            if self.loop:
                self.pos = 0
            else:
                return None
            
        end = min(self.pos + n, len(self.wav))
        chunk = self.wav[self.pos:end]
        self.pos = end

        if len(chunk) < n:
            if self.loop:
                rem = n - len(chunk)
                self.pos = rem
                chunk = np.concatenate([chunk, self.wav[:rem]])
            else:
                chunk = np.pad(chunk, (0, n - len(chunk)), mode='constant')

        return chunk.astype(np.float32)
    
    def stop(self):
        pass


class DemoState:
    def __init__(self, buffer_sec=60):
        self.buffer_sec = buffer_sec
        self.max_samples = int(self.buffer_sec * RATE)
        self.audio = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()    

        self.embeddings = None #[1,2,256]
        self.last_embedding_update = 0.0

        self.spk1_prob = 0.0
        self.spk2_prob  = 0.0

        self.spk1_curve = np.zeros(200, dtype=np.float32)
        self.spk2_curve = np.zeros(200, dtype=np.float32)

        self.status = "initializing.."
        self.num_samples_seen = 0

    def append_audio(self, chunk):
        with self.lock:
            self.audio.extend(chunk.tolist())
            self.num_samples_seen += len(chunk)

    def get_audio_np(self):
        with self.lock:
            if len(self.audio) == 0:
                return np.zeros(1, dtype=np.float32)
            return np.array(self.audio, dtype=np.float32)
        
    def update_embeddings(self, emb):
        with self.lock:
            self.embeddings = emb.detach().cpu()
            self.last_embedding_update = time.time()

    def get_embeddings(self):
        with self.lock:
            if self.embeddings is None:
                return None
            return self.embeddings.clone()
        
    def update_vad_probs(self, p1, p2):
        with self.lock:
            self.spk1_prob = float(p1)
            self.spk2_prob = float(p2)

            self.spk1_curve = np.roll(self.spk1_curve, -1)
            self.spk1_curve[-1] = float(p1)

            self.spk2_curve = np.roll(self.spk2_curve, -1)
            self.spk2_curve[-1] = float(p2)

    def snapshot(self):
        with self.lock:
            return {
                "spk1_prob": self.spk1_prob,
                "spk2_prob": self.spk2_prob,
                "spk1_curve": self.spk1_curve.copy(),
                "spk2_curve": self.spk2_curve.copy(),
                "embeddings": None if self.embeddings is None else self.embeddings.clone(),
                "last_embedding_update": self.last_embedding_update,
                "status": self.status,
                "num_samples_seen": self.num_samples_seen,
                "buffer_len": len(self.audio),
            }
    def set_status(self, status):
        with self.lock:
            self.status = status
    

def inference_worker(
        state,
        audio_source,
        embedder,
        pvad_model,
        pvad_device,
        vad_window_sec=5.0,
        embedding_update_sec=5.0,
        embedding_num_chunks=8,
):
    audio_source.start()
    last_embed_time = 0.0
    last_vad_time = 0.0

    try:
        while True:
            chunk = audio_source.read_available()
            if chunk is not None:
                state.append_audio(chunk)

            now = time.time()
            audio_np = state.get_audio_np()

            enough_for_embedding = len(audio_np) >= int(3.0 * RATE)

            need_initial_embedding = state.get_embeddings() is None and enough_for_embedding
            need_periodic_embedding = (
                enough_for_embedding and
                now - last_embed_time >= embedding_update_sec
            )

            # -------------------------
            # Embedding update
            # -------------------------
            if need_initial_embedding or need_periodic_embedding:
                state.set_status("Updating embeddings")

                with torch.no_grad():
                    emb = embedder.compute_embeddings_from_buffer(
                        audio_np,
                        num_chunks=embedding_num_chunks,
                    )

                state.update_embeddings(emb)

                last_embed_time = now
                state.set_status("Embeddings ready")

                e1_norm = emb[:, 0, :].norm(dim=-1).item()
                e2_norm = emb[:, 1, :].norm(dim=-1).item()

                print(
                    f"\n[Embedding Update] "
                    f"t={time.strftime('%H:%M:%S', time.localtime(now))} "
                    f"e1_norm={e1_norm:.3f} e2_norm={e2_norm:.3f}",
                    flush=True,
                )

            # -------------------------
            # pVAD update every second
            # -------------------------
            if now - last_vad_time >= 1.0:
                last_vad_time = now

                emb = state.get_embeddings()
                if emb is None:
                    state.set_status("Waiting for 3s audio")
                    time.sleep(0.05)
                    continue

                window_samples = int(vad_window_sec * RATE)

                if len(audio_np) < window_samples:
                    wav = np.pad(audio_np, (window_samples - len(audio_np), 0))
                else:
                    wav = audio_np[-window_samples:]

                wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(pvad_device)

                emb = emb.to(pvad_device)
                emb1 = emb[:, 0, :]
                emb2 = emb[:, 1, :]

                with torch.no_grad():
                    logit1 = pvad_model(wav_t, emb1)
                    logit2 = pvad_model(wav_t, emb2)

                    prob1 = torch.sigmoid(logit1).mean().item()
                    prob2 = torch.sigmoid(logit2).mean().item()

                state.update_vad_probs(prob1, prob2)

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass

    finally:
        audio_source.stop()


#Matplotlib UI
def run_ui(state):

    fig = plt.figure(figsize=(11, 6))

    ax_bar = plt.subplot2grid((2, 1), (0, 0))

    ax_curve = plt.subplot2grid((2, 1), (1, 0))

    bars = ax_bar.bar(["Speaker 1", "Speaker 2"], [0.0, 0.0])

    ax_bar.set_ylim(0, 1)

    ax_bar.set_ylabel("pVAD probability")

    ax_bar.set_title("Real-time Quantized-ECAPA + pVAD Demo")

    ax_curve.set_ylim(0, 1)

    ax_curve.set_xlim(0, 200)

    ax_curve.set_ylabel("Probability")

    ax_curve.set_xlabel("Recent pVAD updates")

    line1, = ax_curve.plot(np.zeros(200), label="Speaker 1")

    line2, = ax_curve.plot(np.zeros(200), label="Speaker 2")

    ax_curve.legend(loc="upper right")

    status_text = fig.text(0.02, 0.02, "", fontsize=10)

    def update(_):

        snap = state.snapshot()

        p1 = snap["spk1_prob"]

        p2 = snap["spk2_prob"]

        bars[0].set_height(p1)

        bars[1].set_height(p2)

        line1.set_ydata(snap["spk1_curve"])

        line2.set_ydata(snap["spk2_curve"])

        emb = snap["embeddings"]

        if emb is not None:

            e1 = emb[0, 0, :].detach().cpu().numpy()

            e2 = emb[0, 1, :].detach().cpu().numpy()

            emb_info = (

                f"emb_norms: spk1={np.linalg.norm(e1):.3f}, "

                f"spk2={np.linalg.norm(e2):.3f}"

            )

        else:

            emb_info = "emb_norms: not ready"

        if snap["last_embedding_update"] > 0:

            age = time.time() - snap["last_embedding_update"]

            age_str = f"{age:.1f}s"

        else:

            age_str = "N/A"

        status_text.set_text(

            f"Status: {snap['status']} | "

            f"buffer={snap['buffer_len'] / RATE:.1f}s | "

            f"samples_seen={snap['num_samples_seen']} | "

            f"embedding_age={age_str} | "

            f"{emb_info}"

        )

        return bars[0], bars[1], line1, line2, status_text

    anim = FuncAnimation(

        fig,

        update,

        interval=250,

        blit=False,

        cache_frame_data=False,

    )

    plt.tight_layout()

    plt.show()

    return anim

def run_headless_ui(state):
    print("[Headless UI] Running terminal pVAD display. Press Ctrl+C to stop.")

    last_line_len = 0

    try:
        while True:
            snap = state.snapshot()

            p1 = snap["spk1_prob"]
            p2 = snap["spk2_prob"]

            n1 = int(p1 * 30)
            n2 = int(p2 * 30)

            bar1 = "█" * n1 + "░" * (30 - n1)
            bar2 = "█" * n2 + "░" * (30 - n2)

            if snap["last_embedding_update"] > 0:
                emb_age = time.time() - snap["last_embedding_update"]
                emb_age_str = f"{emb_age:.1f}s"
            else:
                emb_age_str = "N/A"

            status = snap["status"][:18]

            msg = (
                f"spk1={p1:.3f} [{bar1}] | "
                f"spk2={p2:.3f} [{bar2}] | "
                f"buf={snap['buffer_len'] / RATE:.1f}s | "
                f"emb_age={emb_age_str:<6} | "
                f"{status:<18}"
            )

            # Clear previous line fully before printing new one.
            clear = " " * max(0, last_line_len - len(msg))
            print("\r" + msg + clear, end="", flush=True)
            last_line_len = len(msg)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[Headless UI] stopped.")
# ============================================================

# Main

# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--ecapa_onnx",

        type=str,

        required=True,

        help="Path to quantized ECAPA ONNX model.",

    )

    parser.add_argument(

        "--pvad_ckpt",

        type=str,

        required=True,

        help="Path to trained Lightning pVAD checkpoint.",

    )

    parser.add_argument(

        "--wav",

        type=str,

        default="",

        help="Optional WAV file. If provided, simulates realtime streaming.",

    )

    parser.add_argument(

        "--device",

        type=str,

        default="cpu",

        choices=["cpu", "cuda"],

        help="Device for pVAD and frontend.",

    )

    parser.add_argument(

        "--frontend_device",

        type=str,

        default="cpu",

        choices=["cpu", "cuda"],

        help="Device for ECAPA log-Mel frontend.",

    )

    parser.add_argument(

        "--mic_device",

        type=int,

        default=None,

        help="Optional sounddevice input device id.",

    )

    parser.add_argument(

        "--buffer_sec",

        type=float,

        default=60.0,

        help="Rolling audio buffer length.",

    )

    parser.add_argument(

        "--vad_window_sec",

        type=float,

        default=5.0,

        help="Recent audio window used for pVAD display.",

    )
    parser.add_argument(

    "--headless",

    action="store_true",

    help="Run terminal-only demo without matplotlib GUI.",

)

    parser.add_argument(

        "--embedding_update_sec",

        type=float,

        default=60.0,

        help="How often to update ECAPA embeddings.",

    )

    parser.add_argument(

        "--embedding_chunks",

        type=int,

        default=8,

        help="Number of 3-sec chunks sampled from buffer to average embeddings.",

    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():

        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")

        args.device = "cpu"

    if args.frontend_device == "cuda" and not torch.cuda.is_available():

        print("[WARN] CUDA frontend requested but unavailable. Falling back to CPU.")

        args.frontend_device = "cpu"

    pvad_device = torch.device(args.device)

    print("[Demo] pVAD device:", pvad_device)

    print("[Demo] frontend device:", args.frontend_device)

    pvad = load_pvad_from_lightning_ckpt(

        args.pvad_ckpt,

        device=pvad_device,

    )

    # For QDQ quantized ONNX, CPUExecutionProvider is safest.

    embedder = QuantizedECAPAEmbedder(

        onnx_path=args.ecapa_onnx,

        frontend_device=args.frontend_device,

        chunk_sec=3.0,

        providers=["CPUExecutionProvider"],

    )

    if args.wav:

        audio_source = WavAudioStream(

            wav_path=args.wav,

            sample_rate=RATE,

            block_sec=0.25,

            loop=True,

        )

    else:

        audio_source = MicAudioStream(

            sample_rate=RATE,

            block_sec=0.25,

            device_id=args.mic_device,

        )

    state = DemoState(buffer_sec=args.buffer_sec)

    worker = threading.Thread(

        target=inference_worker,

        args=(

            state,

            audio_source,

            embedder,

            pvad,

            pvad_device,

            args.vad_window_sec,

            args.embedding_update_sec,

            args.embedding_chunks,

        ),

        daemon=True,

    )

    worker.start()

    if args.headless:

        run_headless_ui(state)

    else:

        anim = run_ui(state)

if __name__ == "__main__":

    main()