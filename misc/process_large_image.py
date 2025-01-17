import numpy as np
from skimage.draw import disk
from copy import deepcopy
import tensorstore as ts
import trackarray_tensorstore as tta


def get_spec(ndims):
    return {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": None},
        "metadata": {
            "shape": None,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": ([1] * (ndims - 2)) + [32768, 32768]},
            },
            "chunk_key_encoding": {"name": "default"},
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": ([1] * (ndims - 2)) + [512, 512],
                        "codecs": [
                            {
                                "name": "blosc",
                                "configuration": {"cname": "lz4", "clevel": 5},
                            }
                        ],
                    },
                }
            ],
            "data_type": "uint32",
        },
        "context": {"cache_pool": {"total_bytes_limit": 100_000_000}},
        "recheck_cached_data": "open",
    }


def get_read_spec(filename):
    read_spec = deepcopy(get_spec(3))
    read_spec["kvstore"]["path"] = str(filename)
    del read_spec["metadata"]["shape"]
    return read_spec


def get_write_spec(filename, shape):
    write_spec = deepcopy(get_spec(3))
    write_spec["create"] = True
    write_spec["delete_existing"] = True
    write_spec["kvstore"]["path"] = str(filename)
    write_spec["metadata"]["shape"] = list(shape)
    return write_spec


def circular_blob_image(frame_count, image_size, num_blobs, blob_diameter, dtype):
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


def test_break_track_performance(ta):
    with ts.Transaction() as txn:
        ta.break_track(ta.array.shape[0] // 2, 1, True, txn)


def main():
    # Create a circular blob image
    labels = circular_blob_image(10, 8192 * 2, 3000, 30, np.uint16)
    ts.open(get_write_spec("/tmp/test.zarr", labels.shape)).result().write(
        labels
    ).result()
    labels_ts = ts.open(get_read_spec("/tmp/test.zarr")).result()
    ta = tta.TrackArray(labels_ts, {}, {})
    test_break_track_performance(ta)


if __name__ == "__main__":
    main()
