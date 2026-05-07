import torch
from torch import nn
from models.bert import BertTextEncoder
from einops import rearrange, repeat
from models.tmm import EnhanceSubNet
from models.mamba import TCMamba,TQMamba,Crossattn
class TFMamba(nn.Module):
    def __init__(self, args):
        super(TFMamba, self).__init__()

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args['model']['feature_extractor']['bert_pretrained'])

        # Learnable missing token embeddings
        D_v = args['model']['tmm']['input_dim'][1]  # visual input dim
        D_a = args['model']['tmm']['input_dim'][2]  # audio input dim
        D_t = args['model']['tmm']['hidden_dim']    # text hidden dim (post-BERT proj)

        self.missing_embedding_v = nn.Parameter(torch.zeros(1, 1, D_v))
        self.missing_embedding_a = nn.Parameter(torch.zeros(1, 1, D_a))
        self.missing_embedding_t = nn.Parameter(torch.zeros(1, 1, 768))  # BERT dim

        # Initialize with small random values, not zeros
        nn.init.normal_(self.missing_embedding_v, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_embedding_a, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_embedding_t, mean=0.0, std=0.02)

        #input seq t a v
        # TME
        self.text_modality_mixup = EnhanceSubNet(
            input_length=args['model']['tmm']['input_length'],
            input_dim=args['model']['tmm']['input_dim'],
            hidden_dim=args['model']['tmm']['hidden_dim'])
        # feature reconstruction
        self.recon_text_low = nn.Sequential(
            nn.Linear(args['model']['tmr']['input_dim_high'] * 3, args['model']['tmr']['input_dim_high']),
            nn.ReLU(),
            nn.Dropout(args['model']['tmr']['dropout']),
            nn.Linear(args['model']['tmr']['input_dim_high'], args['model']['tmr']['input_dim_low'])
        )
        #TC-Mamba
        self.text_based_context_mamba = TCMamba(
            num_layers=args['model']['tc_mamba']['num_layers'],
            d_model=args['model']['tc_mamba']['d_model'],
            d_ffn=args['model']['tc_mamba']['d_model'] * 4,
            activation=args['model']['tc_mamba']['activation'],
            dropout=args['model']['tc_mamba']['dropout'],
            causal=args['model']['tc_mamba']['causal'],
            mamba_config=args['model']['tc_mamba']['mamba_config']
        )
        D = args['model']['tc_mamba']['d_model']

        # Attention pooling
        self.attn_pool_t = nn.Linear(D, 1)
        self.attn_pool_a = nn.Linear(D, 1)
        self.attn_pool_v = nn.Linear(D, 1)

        # Projection head
        self.contrastive_proj = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )
        #TQ-Mamba
        self.text_guided_attention = Crossattn(
            num_heads=args['model']['tq_mamba']['attn_heads'],
            d_model=args['model']['tq_mamba']['d_model'],

        )
        self.text_based_query_mamba = TQMamba(
            num_layers=args['model']['tq_mamba']['num_layers'],
            d_model=args['model']['tq_mamba']['d_model'],
            d_ffn=args['model']['tq_mamba']['d_model'] * 4,
            activation=args['model']['tq_mamba']['activation'],
            dropout=args['model']['tq_mamba']['dropout'],
            causal=args['model']['tq_mamba']['causal'],
            mamba_config=args['model']['tq_mamba']['mamba_config']
        )


        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output = nn.Linear(args['model']['regression']['input_dim'], args['model']['regression']['out_dim'])

    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        def _apply_missing_embedding(self, x, missing_mask, learned_emb):
            """
            x: [B, L, D] — corrupted input (zeros where missing)
            missing_mask: [B, L] — 1 where token is PRESENT, 0 where MISSING
            learned_emb: [1, 1, D] — learnable missing token
            returns: [B, L, D] — zeros replaced with learned embedding
            """
            mask = missing_mask.unsqueeze(-1).float()  # [B, L, 1]
            # where mask=1 keep x, where mask=0 use learned embedding
            return x * mask + learned_emb.expand(x.size(0), x.size(1), -1) * (1 - mask)


        b = vision_m.size(0)

        # Get vision missing mask: where all features are zero = missing
        # vision_m is [B, L, D_v], zeros where missing
        v_missing_mask = (vision_m.abs().sum(dim=-1) > 1e-6).float()  # [B, L]
        a_missing_mask = (audio_m.abs().sum(dim=-1) > 1e-6).float()   # [B, L]

        h_0_v = self._apply_missing_embedding(vision_m, v_missing_mask, self.missing_embedding_v)
        h_0_a = self._apply_missing_embedding(audio_m, a_missing_mask, self.missing_embedding_a)

        # For text: BERT output, then replace [UNK] positions with learned embedding
        h_0_t_raw = self.bertmodel(language_m)  # [B, L, 768]
        # text_missing_mask from dataset: 1 where present, 0 where [UNK] was inserted
        # We don't have direct access here, so detect via language_m input_ids
        # input_ids are in language_m[:, 0, :], UNK token id = 100
        text_input_ids = language_m[:, 0, :].long()  # [B, L]
        t_missing_mask = (text_input_ids != 100).float()  # [B, L], 0 where UNK
        h_0_t = self._apply_missing_embedding(h_0_t_raw, t_missing_mask, self.missing_embedding_t)
        # text-aware mixup #t v a
        h_tmm_t, h_tmm_v, h_tmm_a = self.text_modality_mixup(h_0_t,h_0_v,h_0_a)

        # tc-mamba a v t
        h_tc_mamba_a, h_tc_mamba_v, h_tc_mamba_t = self.text_based_context_mamba(h_tmm_a,h_tmm_v,h_tmm_t)

        def attn_pool(h, pooler):
            w = torch.softmax(pooler(h), dim=1)  # [B, L, 1]
            return (w * h).sum(dim=1)            # [B, D]

        t_rep, a_rep, v_rep = None, None, None

        if self.training:
            t_rep = self.contrastive_proj(attn_pool(h_tc_mamba_t, self.attn_pool_t))
            a_rep = self.contrastive_proj(attn_pool(h_tc_mamba_a, self.attn_pool_a))
            v_rep = self.contrastive_proj(attn_pool(h_tc_mamba_v, self.attn_pool_v))

        # tq-mamaba
        h_tm_attn = self.text_guided_attention(h_tc_mamba_t,torch.cat([h_tc_mamba_a,h_tc_mamba_v],dim=1))
        h_tm_mamba = self.text_based_query_mamba(h_tm_attn)

        #regression
        h_m_pool = self.pool(h_tm_mamba.permute(0,2,1)).squeeze(-1)
        output = self.output(h_m_pool)

        rec_text_feats, com_text_feats = None, None
        if (vision is not None) and (audio is not None) and (language is not None):
        #text modal recon
            h_t_o = self.bertmodel(language)
            text_recon_low = self.recon_text_low(torch.cat([h_tmm_t, h_tmm_v, h_tmm_a], dim=-1))
            rec_text_feats= [text_recon_low]
            com_text_feats = [h_t_o]

        return {'sentiment_preds': output,
                'rec_text': rec_text_feats,
                'complete_text': com_text_feats,
                'text_rep': t_rep,
                'audio_rep': a_rep,
                'video_rep': v_rep,}




def build_model(args):
    return TFMamba(args)