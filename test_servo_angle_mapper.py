import contextlib
import io

from servo_angle_mapper import _demo


def test_demo_runs_without_keyerror():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _demo()

    out = buf.getvalue()
    assert "Calibration X range" in out
    assert "duplicate" in out
