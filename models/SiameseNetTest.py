import torch
import unittest

from SiameseNet import SiameseNet


class SiameseNetTest(unittest.TestCase):
    def test_network(self):
        model = SiameseNet()

        test_input = torch.zeros((10, 3, 64, 64), dtype=torch.float32)
        test_output = model(test_input)

        print(test_output.shape)

if __name__ == '__main__':
    unittest.main()