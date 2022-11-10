import torch
import unittest

from BaseCNN import BaseCNN

class BaseCNNTest(unittest.TestCase):
    
    def test_shapes(self):
        model = BaseCNN()

        sample_input = torch.zeros((10, 3, 64, 64), dtype=torch.float32)
        test_output = model(sample_input)

        self.assertTrue(test_output.shape == (10, 128))

if __name__ == '__main__':
    unittest.main()