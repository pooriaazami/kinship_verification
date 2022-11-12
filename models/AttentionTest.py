import torch
import unittest

from Attention import ChannelAttention, SpatialAttention


class AttentionTest(unittest.TestCase):

    def test_channel_attention(self):
        attention_layer = ChannelAttention(3)

        sample_input = torch.zeros((10, 3, 64, 64), dtype=torch.float32)
        test_output = attention_layer(sample_input)

        print(test_output.shape)

    def test_spatial_attention(self):
        attention_layer = SpatialAttention()

        sample_input = torch.zeros((10, 3, 64, 64), dtype=torch.float32)
        test_output = attention_layer(sample_input)

        print(test_output.shape)


if __name__ == '__main__':
    unittest.main()
