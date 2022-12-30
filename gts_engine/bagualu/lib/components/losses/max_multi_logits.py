import torch

class MaxMultiLogits(torch.nn.Module):
    
    def __init__(self, class_num: int, hidden_size: int, multiply: int = 1):
        super().__init__()
        
        self.inf = -1e7
        self.class_num = class_num
        self.hidden_size = hidden_size
        self.multiply = multiply

        multi_classes = self.class_num * self.multiply
        self.linear = torch.nn.Linear(in_features=hidden_size,
                                      out_features=multi_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_hidden_states: torch.Tensor) -> torch.Tensor:
        cls_input = input_hidden_states[:, 0, :]
        logits: torch.Tensor = self.linear(cls_input).view(-1, self.class_num, self.multiply)
        max_logits, _ = torch.max(logits, dim=-1)
        return max_logits