import pytest

from rrbench import resource_reservation as reservation

GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def test_administrative_context_must_not_share_the_benchmark_device():
    reservation.validate_scope(7, 8, GPU_UUID, GPU_UUID, benchmark_device=0)
    with pytest.raises(ValueError, match="unused"):
        reservation.validate_scope(0, 8, GPU_UUID, GPU_UUID, benchmark_device=0)
    with pytest.raises(ValueError, match="visible"):
        reservation.validate_scope(7, 8, "0,7", GPU_UUID, benchmark_device=0)
    with pytest.raises(ValueError, match="UUID"):
        reservation.validate_scope(7, 8, "7", "7", benchmark_device=0)


def test_cooperative_admission_requires_the_owned_context_and_no_other_client():
    assert reservation.cooperative_ready([50], 50)
    assert not reservation.cooperative_ready([], 50)
    assert not reservation.cooperative_ready([50, 80], 50)
    assert not reservation.cooperative_ready([80], 50)
    with pytest.raises(ValueError, match="PID"):
        reservation.cooperative_ready([-1], 50)


def test_reservation_deadline_is_absolute_aware_and_finite():
    value = reservation.deadline_timestamp("2026-10-13T03:03:01.929Z")
    assert value > 0
    with pytest.raises(ValueError, match="timezone"):
        reservation.deadline_timestamp("2026-10-13T03:03:01")
