import os
from unittest import TestCase

import numpy as np

from pyvox.parser import VoxParser
from pyvox.writer import VoxWriter


class TestParser(TestCase):
    def test_read_from_file(self):
        filename = os.path.sep.join([os.getcwd(), 'pyvox', 'test', 'howdy.vox'])
        try:
            voxels = VoxParser(filename).parse()
        except Exception as exc:
            self.fail(f"Failed to read from file {filename} :: {exc}")

    def test_read_write(self):
        in_filename = os.path.sep.join([os.getcwd(), 'pyvox', 'test', 'howdy.vox'])
        out_filename = os.path.sep.join([os.getcwd(), 'pyvox', 'test', 'howdy2.vox'])
        entity = VoxParser(in_filename)
        try:
            entity = entity.parse()
        except Exception as exc:
            self.fail(f"Failed to read from file {in_filename} :: {exc}")

        try:
            VoxWriter(out_filename, entity).write()
        except Exception as exc:
            self.fail(f"Failed to write file {out_filename} :: {exc}")

        entity2 = None
        try:
            entity2 = VoxParser(out_filename).parse()
        except Exception as exc:
            self.fail(f"Failed to read from file {out_filename} :: {exc}")

        self.assertTrue(np.array_equal(entity.to_dense(), entity2.to_dense()),
                        "Entities should be the same before and after writing.")

        try:
            os.remove(out_filename)
        except Exception as exc:
            print(f"Failed to clean up after unit test. File: {out_filename}")
        # TODO: add comparisons for materials and such.
