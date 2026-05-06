import torch
from torch import nn
from torch.nn import functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        B = z1.size(0)
        logits = torch.matmul(z1, z2.T) / self.temperature  # [B, B]
        labels = torch.arange(B, device=z1.device)
        loss = (F.cross_entropy(logits, labels) + 
                F.cross_entropy(logits.T, labels)) / 2
        return loss

class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args['base']['alpha']
        self.beta = args['base'].get('beta', 0.1)
        self.Rec_Fn = ReconLoss(type=args['base']['rec_loss'])
        self.MSE_Fn = nn.MSELoss()
        self.InfoNCE_Fn = InfoNCELoss(
            temperature=args['base'].get('nce_temp', 0.1)
        )

    def forward(self, out, label, mask):
        l_sp = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels'])

        l_rec_low = self.Rec_Fn(
            out['rec_text'][0], out['complete_text'][0], mask
        ) if out['rec_text'] is not None and out['complete_text'] is not None else torch.tensor(0.0, device=l_sp.device)

        l_nce = torch.tensor(0.0, device=l_sp.device)
        if out.get('text_rep') is not None:
            t_rep = out['text_rep']
            a_rep = out['audio_rep']
            v_rep = out['video_rep']
            l_nce = (self.InfoNCE_Fn(t_rep, a_rep) + 
                     self.InfoNCE_Fn(t_rep, v_rep))
            # skip (a,v) — corrupted reps fighting each other hurts more than helps

        loss = l_sp + self.alpha * l_rec_low + self.beta * l_nce
        return {'loss': loss, 'l_sp': l_sp, 
                'l_rec': l_rec_low, 'l_nce': l_nce}

class ReconLoss(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.eps = 1e-6
        self.type = type
        if type == 'L1Loss':
            self.loss = nn.L1Loss(reduction='sum')
        elif type == 'SmoothL1Loss':
            self.loss = nn.SmoothL1Loss(reduction='sum')
        elif type == 'MSELoss':
            self.loss = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2]).float()

        loss = self.loss(pred*mask, target*mask) / (torch.sum(mask) + self.eps)

        return loss