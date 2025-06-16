from auricle import constants


def test_window_and_hop_match_fft_sizes():
    # The documented seconds must reproduce the sample counts at 16 kHz.
    assert int(constants.WINDOW_SECONDS * constants.SAMPLE_RATE) == constants.N_FFT
    assert int(constants.HOP_SECONDS * constants.SAMPLE_RATE) == constants.HOP_LENGTH


def test_sample_rate_is_16k():
    assert constants.SAMPLE_RATE == 16_000


def test_hop_smaller_than_window():
    # A hop below the window length gives the STFT overlap it relies on.
    assert 0 < constants.HOP_LENGTH < constants.N_FFT


def test_chunk_window_yields_whisper_frame_count():
    # 30 s of audio at a 10 ms hop is the canonical 3000-frame window.
    frames = int(constants.CHUNK_SECONDS * constants.SAMPLE_RATE) // constants.HOP_LENGTH
    assert frames == 3000


def test_n_mels_positive_and_reasonable():
    assert 20 <= constants.N_MELS <= 256
