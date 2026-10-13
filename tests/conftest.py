import pytest


@pytest.fixture(autouse=True)
def _fresh_pipeline_state():
    """Each test gets its own shared stores / keyword indexes (they are process-wide caches)."""
    from src.pipeline import reset_shared_state

    reset_shared_state()
    yield
    reset_shared_state()
