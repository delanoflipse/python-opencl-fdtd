import pytest


def _has_opencl_device() -> bool:
  try:
    import pyopencl as cl
    return any(platform.get_devices() for platform in cl.get_platforms())
  except Exception:
    return False


HAS_OPENCL = _has_opencl_device()

requires_opencl = pytest.mark.skipif(
    not HAS_OPENCL, reason="no OpenCL device available")
