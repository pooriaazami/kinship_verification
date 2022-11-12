import torch
import torch.nn as nn

# Main Source: https://github.com/ifding/seq2seq-pytorch/blob/master/examples/adaptive-attention-model-for-image-captioning.md

#Attention Block for C_hat calculation
class Atten( nn.Module ):
    def __init__( self, hidden_size ):
        super( Atten, self ).__init__()

        self.affine_v = nn.Linear( hidden_size, 49, bias=False ) # W_v
        self.affine_g = nn.Linear( hidden_size, 49, bias=False ) # W_g
        self.affine_s = nn.Linear( hidden_size, 49, bias=False ) # W_s
        self.affine_h = nn.Linear( 49, 1, bias=False ) # w_h
        
        self.dropout = nn.Dropout( 0.5 )
        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        nn.init.xavier_uniform( self.affine_v.weight )
        nn.init.xavier_uniform( self.affine_g.weight )
        nn.init.xavier_uniform( self.affine_h.weight )
        nn.init.xavier_uniform( self.affine_s.weight )
        
    def forward( self, V, h_t, s_t ):
        '''
        Input: V=[v_1, v_2, ... v_k], h_t, s_t from LSTM
        Output: c_hat_t, attention feature map
        '''
        
        #W_v * V + W_g * h_t * 1^T
        content_v = self.affine_v( self.dropout( V ) ).unsqueeze( 1 ) \
                    + self.affine_g( self.dropout( h_t ) ).unsqueeze( 2 )
        
        #z_t = W_h * tanh( content_v )
        z_t = self.affine_h( self.dropout( F.tanh( content_v ) ) ).squeeze( 3 )
        alpha_t = F.softmax( z_t.view( -1, z_t.size( 2 ) ) ).view( z_t.size( 0 ), z_t.size( 1 ), -1 )
        
        #Construct c_t: B x seq x hidden_size
        c_t = torch.bmm( alpha_t, V ).squeeze( 2 )
        
        #W_s * s_t + W_g * h_t
        content_s = self.affine_s( self.dropout( s_t ) ) + self.affine_g( self.dropout( h_t ) )
        #w_t * tanh( content_s )
        z_t_extended = self.affine_h( self.dropout( F.tanh( content_s ) ) )
        
        #Attention score between sentinel and image content
        extended = torch.cat( ( z_t, z_t_extended ), dim=2 )
        alpha_hat_t = F.softmax( extended.view( -1, extended.size( 2 ) ) ).view( extended.size( 0 ), extended.size( 1 ), -1 )
        beta_t = alpha_hat_t[ :, :, -1 ]
        
        #c_hat_t = beta * s_t + ( 1 - beta ) * c_t
        beta_t = beta_t.unsqueeze( 2 )
        c_hat_t = beta_t * s_t + ( 1 - beta_t ) * c_t

        return c_hat_t, alpha_t, beta_t
