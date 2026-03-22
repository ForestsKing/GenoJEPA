import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedModel, ModernBertConfig, ModernBertModel
from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutput


class GenoJEPAConfig(PretrainedConfig):
    model_type = "GenoJEPA"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.vocab_num = kwargs.get("vocab_num", 5)
        self.pad_value = kwargs.get("pad_value", 4)

        self.global_num = kwargs.get("global_num", 2)
        self.global_scale = kwargs.get("global_scale", [0.65, 0.8])
        self.local_num = kwargs.get("local_num", 6)
        self.local_scale = kwargs.get("local_scale", [0.35, 0.4])

        self.patch_size = kwargs.get("patch_size", 16)
        self.embed_size = kwargs.get("embed_size", 16)
        self.head_size = kwargs.get("head_size", 16)

        self.hidden_size = kwargs.get("hidden_size", 512)
        self.intermediate_size = kwargs.get("intermediate_size", 2048)
        self.layer_num = kwargs.get("layer_num", 12)
        self.head_num = kwargs.get("head_num", 16)

        self.slice_num = kwargs.get("slice_num", 256)
        self.point_num = kwargs.get("point_num", 17)
        self.max_t = kwargs.get("max_t", 3.0)
        self.alpha = kwargs.get("alpha", 0.05)

        self.auto_map = {
            "AutoConfig": "model.GenoJEPAConfig",
            "AutoModel": "model.GenoJEPAForSequenceEmbedding",
            "AutoModelForSequenceClassification": "model.GenoJEPAForSequenceClassification",
        }

    def to_dict(self):
        output = super().to_dict()
        output["auto_map"] = self.auto_map

        return output


class Patch(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pad_value = config.pad_value
        self.patch_size = config.patch_size

    @torch.no_grad()
    def forward(self, data):
        batch_size, seq_len = data.shape

        pad_len = (self.patch_size - seq_len % self.patch_size) % self.patch_size
        pad_data = F.pad(data, (0, pad_len), value=self.pad_value)
        patch_data = rearrange(pad_data, "B (N P) -> B N P", P=self.patch_size)
        patch_mask = (patch_data != self.pad_value).any(dim=-1).long()

        return patch_data.detach(), patch_mask.detach()


class Augment(nn.Module):
    def __init__(self, config):
        super().__init__()

    @torch.no_grad()
    def _crop(self, data, scale):
        batch_size, seq_len = data.shape

        sample_scale = np.random.uniform(scale[0], scale[1])
        sample_len = max(int(seq_len * sample_scale), 1)
        start_idx = torch.randint(0, seq_len - sample_len + 1, (batch_size, 1), device=data.device)
        gather_idx = start_idx + torch.arange(sample_len, device=data.device).unsqueeze(0)
        aug_data = torch.gather(data, 1, gather_idx)

        return aug_data.detach()

    @torch.no_grad()
    def forward(self, data, scale):
        aug_data = self._crop(data, scale)

        return aug_data.detach()


class SIGReg(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.slice_num = config.slice_num

        t = torch.linspace(0, config.max_t, config.point_num)
        window = torch.exp(-t.square() / 2.0)
        dt = config.max_t / (config.point_num - 1)
        weight = torch.full((config.point_num,), 2 * dt)
        weight[[0, -1]] = dt

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weight", weight * window)

    def forward(self, embed):
        view_num, batch_size, hidden_size = embed.shape

        A = torch.randn((hidden_size, self.slice_num)).to(embed.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (embed @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(dim=-3) - self.phi).square() + x_t.sin().mean(dim=-3).square()
        loss = (err @ self.weight) * batch_size
        loss = loss.mean()

        return loss


class Embedder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_num, config.embed_size)
        self.embedder = nn.Conv1d(
            in_channels=config.embed_size,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )

    def forward(self, data):
        batch_size, patch_num, patch_size = data.shape

        embed = self.embedding(data)
        embed = rearrange(embed, "B N P D -> (B N) D P")
        embed = self.embedder(embed)
        embed = rearrange(embed, "(B N) D 1 -> B N D", B=batch_size)

        return embed


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size), requires_grad=True)

        self.patch = Patch(config)
        self.embedder = Embedder(config)
        self.encoder = ModernBertModel(
            ModernBertConfig(
                vocab_size=config.vocab_num,
                pad_token_id=config.pad_value,
                global_attn_every_n_layers=1,

                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.layer_num,
                num_attention_heads=config.head_num,
            )
        )

    def forward(self, data):
        batch_size, seq_len = data.shape

        patch_data, patch_mask = self.patch(data)
        init_patch_embed = self.embedder(patch_data)

        init_cls_embed = self.cls_token.expand(batch_size, -1, -1)
        cls_mask = torch.ones(batch_size, 1, device=data.device).long()
        input_embed = torch.cat((init_cls_embed, init_patch_embed), dim=1)
        input_mask = torch.cat((cls_mask, patch_mask), dim=1)

        output = self.encoder(inputs_embeds=input_embed, attention_mask=input_mask).last_hidden_state
        cls_embed, patch_embed = output[:, 0, :], output[:, 1:, :]

        patch_mask_expanded = patch_mask.unsqueeze(-1)
        mask_sum = patch_mask_expanded.sum(dim=1)
        sum_embed = (patch_embed * patch_mask_expanded).sum(dim=1)
        avg_embed = sum_embed / mask_sum.clamp(min=1)
        raw_max = patch_embed.masked_fill(patch_mask_expanded == 0, float("-inf")).max(dim=1)[0]
        max_embed = torch.where(mask_sum > 0, raw_max, torch.zeros_like(raw_max))

        return ModelOutput(patch_embed=patch_embed, cls_embed=cls_embed, avg_embed=avg_embed, max_embed=max_embed)


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(config.hidden_size, config.intermediate_size, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm1d(config.intermediate_size),
            nn.GELU(),
            nn.Conv1d(config.intermediate_size, config.head_size, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, embed):
        embed = rearrange(embed, "V B D -> B D V")
        embed = self.head(embed)
        embed = rearrange(embed, "B D V -> V B D")

        return embed


class GenoJEPAForPreTraining(PreTrainedModel):
    config_class = GenoJEPAConfig

    def __init__(self, config):
        super().__init__(config)

        self.alpha = config.alpha
        self.pred_loss = 0.0
        self.sigreg_loss = 0.0

        self.global_num = int(config.global_num)
        self.global_scale = config.global_scale
        self.local_num = int(config.local_num)
        self.local_scale = config.local_scale

        self.augment = Augment(config)
        self.model = Model(config)
        self.head = Head(config)

        self.pred_criterion = nn.MSELoss()
        self.sigreg_criterion = SIGReg(config)

        self.post_init()

    @torch.no_grad()
    def get_detailed_metric(self):
        return ModelOutput(pred_loss=self.pred_loss, sigreg_loss=self.sigreg_loss)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if not self.training:
            sequence_embed = self.model(input_ids.long()).avg_embed
            return SequenceClassifierOutput(loss=sequence_embed.std(dim=0).mean(), logits=sequence_embed)

        embed_list = []
        for _ in range(self.global_num):
            global_input_ids = self.augment(input_ids.long(), self.global_scale)
            embed_list.append(self.model(global_input_ids).avg_embed)
        for _ in range(self.local_num):
            local_input_ids = self.augment(input_ids.long(), self.local_scale)
            embed_list.append(self.model(local_input_ids).avg_embed)

        embed = self.head(torch.stack(embed_list, dim=0))
        global_embed, local_embed = torch.split(embed, [self.global_num, self.local_num], dim=0)

        pred_loss = self.pred_criterion(embed, global_embed.mean(dim=0, keepdim=True))
        sigreg_loss = self.sigreg_criterion(embed)
        self.pred_loss, self.sigreg_loss = pred_loss.item(), sigreg_loss.item()

        loss = (1 - self.alpha) * pred_loss + self.alpha * sigreg_loss

        return ModelOutput(loss=loss)


class GenoJEPAForSequenceEmbedding(PreTrainedModel):
    config_class = GenoJEPAConfig

    def __init__(self, config):
        super().__init__(config)

        self.model = Model(config)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        embed = self.model(input_ids.long()).avg_embed
        return embed

    def encode(self, sequence, tokenizer, batch_size=64):
        original_training = self.training

        self.eval()
        embed_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequence), batch_size)):
                batch_sequence = sequence[i:i + batch_size]
                batch_input_dict = tokenizer(batch_sequence, return_tensors="pt", padding=True)
                batch_input_ids = batch_input_dict["input_ids"].to(self.model.cls_token.device)
                batch_embed = self.forward(input_ids=batch_input_ids)
                embed_list.append(batch_embed.detach().cpu())

        embed = torch.concat(embed_list, dim=0).numpy()

        if original_training:
            self.train()

        return embed


class GenoJEPAForSequenceClassification(PreTrainedModel):
    config_class = GenoJEPAConfig

    def __init__(self, config):
        super().__init__(config)

        self.model = Model(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.LayerNorm(config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.num_labels),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        embeds = self.model(input_ids.long()).cls_embed
        logits = self.classifier(embeds)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
