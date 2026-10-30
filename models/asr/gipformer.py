from .base import BaseASR

REPO_IDS = {
    "1": "g-group-ai-lab/gipformer-68M-rnnt",
    "1.5": "g-group-ai-lab/gipformer1.5-68M-rnnt",
}
DEFAULT_VERSION = "1.5"

SAMPLE_RATE = 16000
FEATURE_DIM = 80

ONNX_FILES = {
    "fp32": {
        "encoder": "encoder.onnx",
        "decoder": "decoder.onnx",
        "joiner": "joiner.onnx",
    },
    "int8": {
        "encoder": "encoder.int8.onnx",
        "decoder": "decoder.int8.onnx",
        "joiner": "joiner.int8.onnx",
    },
}

TOKENS_FILE = "tokens.txt"

class GipformerASR(BaseASR):
    '''
    Reference: https://github.com/ggroup-ai-lab/gipformer
    '''
    def __init__(
        self,
        model_version: str = "1.5",
        quantize: str = "fp32",
        num_threads: int = 4,
        decoding_method: str = "greedy_search",
    ):
        try:
            import sherpa_onnx
            import soundfile as sf
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                f"Missing required dependencies for Gipformer ASR ({e}).\n"
                "To use Gipformer, please install requirements by the following command:\n"
                "  uv pip install -r models/asr/requirements-gipformer.txt"
            ) from e

        self.sf = sf
        self.sherpa_onnx = sherpa_onnx

        repo_id = REPO_IDS.get(model_version, REPO_IDS[DEFAULT_VERSION])
        encoder_name = ONNX_FILES[quantize]["encoder"]
        decoder_name = ONNX_FILES[quantize]["decoder"]
        joiner_name = ONNX_FILES[quantize]["joiner"]

        encoder = hf_hub_download(repo_id=repo_id, filename=encoder_name)
        decoder = hf_hub_download(repo_id=repo_id, filename=decoder_name)
        joiner = hf_hub_download(repo_id=repo_id, filename=joiner_name)
        tokens = hf_hub_download(repo_id=repo_id, filename=TOKENS_FILE)

        # Create a sherpa-onnx OfflineRecognizer for transducer decoding.
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=FEATURE_DIM,
            decoding_method=decoding_method,
        )

    def read_audio(self, filename: str) -> tuple:
        """Read an audio file and return float32 samples and sample rate.
        Supports WAV, FLAC, OGG, and other formats via soundfile.
        """
        samples, sample_rate = self.sf.read(filename, dtype="float32")

        # Convert stereo to mono if needed
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        return samples, sample_rate

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file.
        Returns the recognized text.
        """
        samples, sample_rate = self.read_audio(audio_path)

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self.recognizer.decode_streams([stream])

        return stream.result.text.strip()

if __name__ == "__main__":
    # Test on a sample audio file
    audio_file = "data/train+public_test/train+public_test/train/audio/train_0001/train_0001.wav"  # Replace with your audio file
    model = GipformerASR()
    
    import time
    import soundfile as sf
    start = time.time()
    text = model.transcribe(audio_file)
    elapsed = time.time() - start

    audio_info = sf.info(audio_file)
    duration = audio_info.duration
    rtf = elapsed / duration if duration > 0 else 0

    print(f"\n  File: {audio_file}")
    print(f"  Text: {text}")
    print(f"  Time: {elapsed:.2f}s | Audio: {duration:.2f}s | RTF: {rtf:.3f}")