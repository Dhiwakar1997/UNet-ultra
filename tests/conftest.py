import pytest

from unetultra.fake_session import make_fake_session

SMALL = dict(n_samples=8, image_size=(512, 384))


@pytest.fixture(scope="session")
def fake_zip(tmp_path_factory):
    """A small single-class fake session with true masks."""
    return make_fake_session(tmp_path_factory.mktemp("fake") / "session_fake.zip", with_masks=True, **SMALL)
