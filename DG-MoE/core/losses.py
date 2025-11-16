from torch import nn

class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.gamma = args['base']['gamma']
        self.sigma = args['base']['sigma']
        self.beta = args['base']['beta']
        self.alph = args['base']['alpha']

        self.CE_Fn = nn.CrossEntropyLoss()
        self.MSE_Fn = nn.MSELoss() 

    def forward(self, out, label):

        share_sim_loss = out['share_sim_loss']
        diff_loss = out['diff_loss']
        l_sp = self.MSE_Fn(out['sentiment_preds'],label['sentiment_labels'])
        sup_loss = out['sup_sim_loss']
        
        loss = self.sigma * l_sp + self.gamma * share_sim_loss + self.beta * diff_loss + self.alph * sup_loss

        return {'loss': loss, 'l_sp': l_sp, 'share_sim_loss':share_sim_loss, 'diff_loss':diff_loss, 'sup_loss':sup_loss}
