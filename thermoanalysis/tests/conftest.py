from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def this_dir(request):
    path = Path(request.fspath)
    return path.parent
