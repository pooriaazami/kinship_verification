import torch
import unittest

from models.VGGFace import VGGFace

class VGGFaceTest(unittest.TestCase):
    def test_model(self):
        model = VGGFace(2)

        sample_input = torch.zeros((10, 3, 64, 64))
        sample_output = model(sample_input)

        print(sample_output.shape)

if __name__ == '__main__':
    unittest.main()