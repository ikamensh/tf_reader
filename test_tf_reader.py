import tensorflow as tf
import os

from tf_reader import records

import math


def test_writes(tmpdir):
    files_before = set(os.listdir(tmpdir))

    summary_writer = tf.summary.create_file_writer( str(tmpdir) )

    with summary_writer.as_default():
        tf.summary.scalar('loss', 0.1, step=42)
        tf.summary.scalar('loss', 0.2, step=43)
        tf.summary.scalar('loss', 0.3, step=44)

    new_file = next(iter(set(os.listdir(tmpdir)) - files_before))

    path = os.path.join(tmpdir, new_file)

    data = list(records(path))

    assert len(data) == 3

    assert math.isclose(data[0].value, 0.1, rel_tol=1e-4)
    assert data[1].tag == 'loss'
    assert data[2].step == 44
    assert data[0].value != data[1].value
    assert data[0].step != data[1].step



