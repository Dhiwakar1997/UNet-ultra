import pytest

from unetultra.fake_session import make_fake_session

SMALL = dict(n_samples=8, image_size=(512, 384))


@pytest.fixture(scope="session")
def fake_zip(tmp_path_factory):
    """A small single-class fake session with true masks."""
    return make_fake_session(tmp_path_factory.mktemp("fake") / "session_fake.zip", with_masks=True, **SMALL)


@pytest.fixture(scope="session")
def fake_zips(tmp_path_factory):
    """Three small training sessions and one test session, all with true masks."""
    root = tmp_path_factory.mktemp("sessions")
    train = [make_fake_session(root / f"train{i}.zip", seed=i, with_masks=True, n_samples=10, image_size=(512, 384))
             for i in range(3)]
    test = make_fake_session(root / "test.zip", seed=9, split="test", with_masks=True, n_samples=10,
                             image_size=(512, 384))
    return {"train": train, "test": test}
