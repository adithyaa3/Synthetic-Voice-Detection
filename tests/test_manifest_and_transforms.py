import numpy as np
import pandas as pd
import pytest

from voicedetect import transforms as T
from voicedetect.manifest import assign_partitions, common_voice_runs, parse_user_ai_name

from .conftest import requires_data, speech_like


def test_parse_user_ai_name_is_case_insensitive():
    assert parse_user_ai_name("Obama-to-Biden.wav") == ("obama", "biden")
    assert parse_user_ai_name("obama-to-linus.wav") == ("obama", "linus")
    assert parse_user_ai_name("weird name.wav") == ("unknown", "unknown")


def test_common_voice_runs_split_on_gaps():
    names = [f"common_voice_en_{i}.mp3" for i in (100, 101, 102, 110, 111, 500)]
    g = common_voice_runs(names, gap=5)
    assert g[names[0]] == g[names[1]] == g[names[2]]
    assert g[names[3]] == g[names[4]] != g[names[0]]
    assert g[names[5]] not in {g[names[0]], g[names[3]]}


def test_assign_partitions_keeps_groups_whole_and_is_deterministic():
    rows = []
    for s in "abcdefgh":
        for t in range(3):
            rows.append({"file_id": f"ai/{s}{t}", "origin": "user", "label": 1, "group": f"vc_src_{s}", "official_partition": ""})
    for g in range(20):
        for k in range(5):
            rows.append({"file_id": f"h/{g}_{k}", "origin": "user", "label": 0, "group": f"cv_{g}", "official_partition": ""})
    rows.append({"file_id": "asv/1", "origin": "public", "label": 1, "group": "asv_x", "official_partition": "eval"})
    df = pd.DataFrame(rows)
    a, b = assign_partitions(df), assign_partitions(df)
    assert a.partition.tolist() == b.partition.tolist()
    assert (a.groupby("group").partition.nunique() == 1).all()
    assert a.loc[(a.label == 1) & (a.origin == "user") & (a.partition == "test"), "group"].nunique() == 2
    assert 0.2 <= (a[(a.label == 0)].partition == "test").mean() <= 0.35
    assert a.loc[a.file_id == "asv/1", "partition"].item() == "test"


def test_paths_through_a_linked_data_folder_stay_project_relative(tmp_path):
    """Regression: a data/ folder that is a junction/symlink to elsewhere must not break the manifest."""
    import sys
    from voicedetect.manifest import _rel
    real = tmp_path / "elsewhere" / "human"
    real.mkdir(parents=True)
    (real / "a.mp3").write_bytes(b"x")
    proj = tmp_path / "proj"
    (proj / "data").mkdir(parents=True)
    link = proj / "data" / "human"
    try:
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(str(real), str(link))
        else:
            link.symlink_to(real, target_is_directory=True)
    except OSError as e:
        pytest.skip(f"cannot create a directory link here: {e}")
    assert _rel(link / "a.mp3", proj) == "data/human/a.mp3"


@requires_data
def test_real_manifest_is_consistent():
    from voicedetect.config import ROOT
    from voicedetect.manifest import load_manifest
    m = load_manifest()
    assert m.file_id.is_unique
    assert all((ROOT / p).exists() for p in m.path)
    assert ((m.class_name == "ai") == (m.label == 1)).all()
    assert (m.supplementary == m.origin.ne("user")).all()
    assert set(m.origin) <= {"user", "public", "self_generated"}
    assert m[["source", "group", "generator", "partition", "sha256"]].notna().all().all()
    # user AI groups are the 8 source speakers; every source has 7 conversions
    vc = m[m.source == "user_voice_conversion"]
    assert vc.group.nunique() == 8 and (vc.groupby("group").size() == 7).all()


def test_mp3_roundtrip_preserves_length_and_changes_signal():
    y = speech_like(4, 22050, seed=2)
    z = T.mp3_roundtrip(y, 22050, kbps=48, codec_sr=32000)
    assert z.shape == y.shape and z.dtype == np.float32
    assert 0 < np.mean((z - y) ** 2) < np.mean(y ** 2)
    level, kbps = T.mp3_level_for(48, 32000)
    assert kbps == 48  # the Common Voice format is reproducible exactly
    # and the encoded size really corresponds to ~48 kbps (6 s, like a Common Voice clip)
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, speech_like(6, 32000, seed=5), 32000, format="MP3", compression_level=level, bitrate_mode="CONSTANT")
    assert abs(buf.tell() * 8 / 6 / 1000 - 48) < 1.5


@pytest.mark.parametrize("snr", [30, 10, 0])
def test_add_noise_hits_requested_snr(snr):
    y = speech_like(4, 22050, seed=3)
    z = T.add_noise(y, snr, "white", seed=1)
    measured = 10 * np.log10(np.mean(y.astype(float) ** 2) / np.mean((z - y).astype(float) ** 2))
    assert measured == pytest.approx(snr, abs=0.2)


def test_simple_transforms():
    y = speech_like(4, 22050, seed=4)
    assert len(T.truncate(y, 22050, 1.0)) == 22050
    assert np.max(np.abs(T.gain(y, 10))) <= 1.0
    assert T.resample_roundtrip(y, 22050, 8000).shape == y.shape
    assert T.mic_like(y, 22050, seed=0).shape == y.shape
    assert T.reverb(y, 22050).shape == y.shape
