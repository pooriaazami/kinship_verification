import torch
import unittest

from Attention import ChannelAttention, SpatialAttention, AttentionLayer, SpatialAttn


class AttentionTest(unittest.TestCase):

    def test_channel_attention(self):
        attention_layer = ChannelAttention(3)

        sample_input = torch.zeros((10, 3, 64, 64), dtype=torch.float32)
        test_output = attention_layer(sample_input)

        print('channel_attention: ', test_output.shape)

    def test_spatial_attention(self):
        attention_layer = SpatialAttention()

        sample_input = torch.zeros((10, 3, 64, 64), dtype=torch.float32)
        test_output = attention_layer(sample_input)

        print('spetial_attention: ', test_output.shape)

    def test_attention_layer(self):
        attention_layer = AttentionLayer(3)

        sample_input = torch.zeros((10, 3, 64, 64), dtype=torch.float32)
        test_output = attention_layer(sample_input)

        print('attention_layer: ', test_output.shape)

    def test_spatial_attn(self):
        attention_layer = SpatialAttn(in_features=3)

        sample_input = torch.zeros((10, 3, 64, 64), dtype=torch.float32)
        test_output = attention_layer(sample_input)

        print('spatial_attn: ', test_output.shape)


if __name__ == '__main__':
    unittest.main()
