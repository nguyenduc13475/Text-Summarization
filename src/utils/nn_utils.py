import torch.nn as nn
import torch.nn.init as init


def init_weights(m: nn.Module) -> None:
    """
    Initialize standard weights for Neural Network layers.
    Apply Xavier Uniform for Linear/Attention, Orthogonal for LSTM/RNN.
    """
    if isinstance(m, nn.Embedding):
        init.uniform_(m.weight, -0.1, 0.1)

    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                init.orthogonal_(param.data)
            elif "bias" in name:
                init.zeros_(param.data)
                # Setting the forget gate bias to 1.0 helps the model remember better in the initial stages.
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            init.xavier_uniform_(m.in_proj_weight)
        if m.out_proj.weight is not None:
            init.xavier_uniform_(m.out_proj.weight)
        if m.in_proj_bias is not None:
            init.zeros_(m.in_proj_bias)
        if m.out_proj.bias is not None:
            init.zeros_(m.out_proj.bias)

    elif isinstance(m, nn.LayerNorm):
        init.ones_(m.weight)
        init.zeros_(m.bias)
