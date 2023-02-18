import torch.nn as nn

class TableNetLoss(nn.Module):
    def __init__(self):
        super(TableNetLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, table_prediction, table_target, column_prediction = None, column_target = None,):
        table_loss = self.bce(table_prediction, table_target)
        column_loss = self.bce(column_prediction, column_target)
        return table_loss, column_loss
