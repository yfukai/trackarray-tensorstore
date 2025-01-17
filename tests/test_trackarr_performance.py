import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
import pytest
import itertools
import tensorstore as ts

@pytest.fixture
def circular_blob_image(request):
    frame_count, image_size, num_blobs, blob_diameter, dtype = request.param
    image = np.zeros((frame_count, image_size, image_size), dtype=dtype)
    radius = blob_diameter // 2
    for frame in range(frame_count):
        for i in range(num_blobs):
            # Randomly select the center of the circle
            center_x = np.random.randint(radius, image.shape[1] - radius)
            center_y = np.random.randint(radius, image.shape[2] - radius)

            # Get the coordinates of the circle
            rr, cc = disk((center_x, center_y), radius, shape=image.shape[1:])

            # Draw the circle
            image[frame, rr, cc] = i
    return image 
        
@pytest.mark.parametrize(
    "circular_blob_image",
    list(itertools.product([10, 20], [2048, 8192*2], [300, 3000], [30], [np.uint16, np.uint32])),
    indirect=True,
)
def test_break_track_performance(circular_blob_image, trackarr, benchmark):
    ta = trackarr(circular_blob_image, {})
    with ts.Transaction() as txn:
        benchmark(ta.break_track,ta.array.shape[0]//2, 1, True, txn)
    assert ta.is_valid()