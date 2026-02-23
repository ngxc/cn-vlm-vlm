import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModel, CLIPProcessor, BertTokenizerFast
from tqdm import tqdm
import random
import math
from PIL import Image
import collections
import itertools
from torch.cuda.amp import autocast, GradScaler

IMAGE_DIR = "/root/lanyun-tmp/flickr30k/Images"
TOKEN_FILE = "translated_pro.txt"
MINILLM_MODEL_PATH = "large.pt"

BATCH_SIZE = 20
MAX_LEN = 512
NUM_PLACEHOLDER = 128
LR = 8e-5
LLM_LR_RATIO = 0.1
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
GRAD_ACCUM_STEPS = 4
DATA_RATIO = 1.0
CONTRASTIVE_WEIGHT = 0.5

SAVE_STEP_INTERVAL = 500
KEEP_LATEST_N = 3

PLACEHOLDER_TOKEN = "<image>"

CKPT_SAVE_DIR = "checkpoints"
os.makedirs(CKPT_SAVE_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CKPT_SAVE_DIR, "latest_ckpt.pt")

class MiniLLMBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden_dim = dim * 4
        self.ff_linear1 = nn.Linear(dim, hidden_dim * 2)
        self.ff_act = nn.GLU()
        self.ff_linear2 = nn.Linear(hidden_dim, dim)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        x = x + attn_out
        res = x
        x = self.ln2(x)
        x = self.ff_linear1(x)
        x = self.ff_act(x)
        x = self.ff_linear2(x)
        x = self.ff_dropout(x)
        x = x + res
        return x

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, max_len=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.layers = nn.ModuleList([MiniLLMBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.max_len = max_len

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        x = inputs_embeds if inputs_embeds is not None else self.embed(input_ids)
        bsz, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embed(pos[:, :seq_len])
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        for blk in self.layers:
            x = blk(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        return self.head(self.ln(x))

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )

    def forward(self, x): return x + self.net(self.ln(x))

class VisionToTextProj(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=8, dropout=0.1):
        super().__init__()
        self.initial_proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim))
        self.layers = nn.ModuleList([ResidualBlock(out_dim, dropout) for _ in range(num_layers)])
        self.final_ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.initial_proj(x)
        for layer in self.layers: x = layer(x)
        return self.final_ln(x)

class TinyMiniLLMCLIPVLM(nn.Module):
    def __init__(self, text_model, vision_model, proj, placeholder_id):
        super().__init__()
        self.text_model, self.vision_model, self.proj = text_model, vision_model, proj
        self.vis_norm = nn.LayerNorm(vision_model.config.hidden_size).to(DEVICE)
        self.placeholder_id = placeholder_id
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, vision_inputs, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            for k in vision_inputs: vision_inputs[k] = vision_inputs[k].to(DEVICE)
            vis_feats = self.vis_norm(self.vision_model(**vision_inputs).last_hidden_state)

        vis_proj = self.proj(vis_feats)
        text_emb = self.text_model.embed(input_ids.to(DEVICE))
        placeholder_mask = (input_ids == self.placeholder_id)

        vis_seq_len = vis_proj.size(1)
        chunk_indices = torch.linspace(0, vis_seq_len, steps=NUM_PLACEHOLDER + 1).long()

        for b in range(text_emb.size(0)):
            idxs = torch.nonzero(placeholder_mask[b], as_tuple=True)[0]
            for i, idx in enumerate(idxs):
                if i >= NUM_PLACEHOLDER: break
                start, end = chunk_indices[i], chunk_indices[i + 1]
                text_emb[b, idx] = vis_proj[b, start:end].mean(dim=0) if start < end else vis_proj[
                    b, min(start, vis_seq_len - 1)]

        logits = self.text_model(inputs_embeds=text_emb, attention_mask=attention_mask.to(DEVICE))
        lm_loss = torch.tensor(0.0, device=DEVICE)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels.to(DEVICE)[..., 1:].contiguous()
            lm_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        vis_mean = nn.functional.normalize(vis_proj.mean(dim=1), dim=-1)
        mask_txt = attention_mask.to(DEVICE).bool() & (~placeholder_mask.to(DEVICE))
        denom = mask_txt.sum(dim=1, keepdim=True).float().placeholder_mask = (~placeholder_mask.to(DEVICE))
        denom = mask_txt.sum(dim=1, keepdim=True).float().clamp(min=1)
        txt_mean = nn.functional.normalize((text_emb * mask_txt.unsqueeze(-1)).sum(dim=1) / denom, dim=-1)
        sim = vis_mean @ txt_mean.t() / 0.07
        lbl = torch.arange(sim.size(0), device=DEVICE)
        cont_loss = (nn.CrossEntropyLoss()(sim, lbl) + nn.CrossEntropyLoss()(sim.t(), lbl)) / 2
        return lm_loss + CONTRASTIVE_WEIGHT * cont_loss, lm_loss, cont_loss

class FlickrDataset(Dataset):
    def __init__(self, data_list, tokenizer, processor, max_len=512):
        self.data_list, self.tokenizer, self.processor, self.max_len = data_list, tokenizer, processor, max_len
        self.header_ids = tokenizer.encode("<|im_start|><|user|>", add_special_tokens=False)
        self.mid_ids = tokenizer.encode("<|im_end|><|im_start|><|assistant|>", add_special_tokens=False)
        self.eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.img_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        try:
            image = Image.open(item['image_path']).convert("RGB")
            vis = self.processor(images=image, return_tensors="pt")
        except:
            vis = self.processor(images=Image.new('RGB', (224, 224), 0), return_tensors="pt")

        img_tokens = [self.img_id] * NUM_PLACEHOLDER
        caption_ids = self.tokenizer.encode(item['caption'], add_special_tokens=False)
        input_ids = self.header_ids + img_tokens + self.mid_ids + caption_ids + [self.eos_id]

        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            input_ids[-1] = self.eos_id

        labels = torch.tensor(input_ids).clone()
        labels[:min(len(self.header_ids) + NUM_PLACEHOLDER + len(self.mid_ids), len(labels))] = -100

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.ones(len(input_ids))
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len)])

        return {k: v.squeeze(0) for k, v in vis.items()}, {'input_ids': input_ids.long(),
                                                           'attention_mask': attention_mask.long(),
                                                           'labels': labels.long()}

if __name__ == "__main__":
    print(f"🔹 Config: Tokens={NUM_PLACEHOLDER}, Proj LR={LR}, LLM LR={LR * LLM_LR_RATIO}, SAVE_INTERVAL={SAVE_STEP_INTERVAL}")

    data_list = []
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    img_path = os.path.join(IMAGE_DIR, parts[0].split("#")[0].strip())
                    if os.path.exists(img_path): data_list.append({"image_path": img_path, "caption": parts[1].strip()})

    if DATA_RATIO < 1.0:
        random.shuffle(data_list)
        data_list = data_list[:int(len(data_list) * DATA_RATIO)]
    print(f" Samples: {len(data_list)}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", PLACEHOLDER_TOKEN]})

    text_model = MiniLLM(len(tokenizer), 512, 24, 8, MAX_LEN).to(DEVICE)
    if os.path.exists(MINILLM_MODEL_PATH):
        st = torch.load(MINILLM_MODEL_PATH, map_location=DEVICE)
        if st['embed.weight'].shape[0] != len(tokenizer):
            new_emb = text_model.embed.weight.data.clone()
            new_emb[:min(st['embed.weight'].shape[0], len(tokenizer))] = st['embed.weight'][
                                                                         :min(st['embed.weight'].shape[0],
                                                                              len(tokenizer))]
            st['embed.weight'] = new_emb
            if 'head.weight' in st: st['head.weight'] = new_emb
        text_model.load_state_dict(st, strict=False)
        text_model.head.weight = text_model.embed.weight

    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    for p in vision_model.parameters(): p.requires_grad = False

    proj = VisionToTextProj(vision_model.config.hidden_size, 512, num_layers=8).to(DEVICE)
    vlm = TinyMiniLLMCLIPVLM(text_model, vision_model, proj, tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)).to(DEVICE)

    if torch.__version__.startswith("2."):
        print(" Compiling model with torch.compile()...")
        vlm = torch.compile(vlm)

    print(" Freezing text_model layers except the last two...")
    for param in text_model.parameters(): param.requires_grad = False
    num_layers_to_unfreeze = 2
    for i in range(len(text_model.layers) - num_layers_to_unfreeze, len(text_model.layers)):
        print(f" Unfreezing layer {i}")
        for param in text_model.layers[i].parameters(): param.requires_grad = True
    print(" Unfreezing final LayerNorm and output head.")
    for param in text_model.ln.parameters(): param.requires_grad = True
    for param in text_model.head.parameters(): param.requires_grad = True

    unfrozen_llm_params = itertools.chain(text_model.layers[-2].parameters(), text_model.layers[-1].parameters(),
                                          text_model.ln.parameters(), text_model.head.parameters())
    optimizer = torch.optim.AdamW([
        {'params': proj.parameters(), 'lr': LR},
        {'params': unfrozen_llm_params, 'lr': LR * LLM_LR_RATIO}
    ])
    print(f" Optimizer configured with 2 parameter groups:")
    print(f"  - Group 1 (proj): LR = {LR}")
    print(f"  - Group 2 (unfrozen LLM): LR = {LR * LLM_LR_RATIO}")

    dataloader = DataLoader(
        FlickrDataset(data_list, tokenizer, vision_processor, MAX_LEN),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    start_epoch, global_step = 0, 0
    step_checkpoint_queue = collections.deque(maxlen=KEEP_LATEST_N)
    scaler = GradScaler()
    print("️ Mixed Precision Training Enabled.")

    if os.path.exists(CKPT_PATH):
        print(f" Resuming from {CKPT_PATH}...")
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        proj.load_state_dict(ckpt['proj'])
        text_model.load_state_dict(ckpt['text_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', start_epoch * len(dataloader))

    vlm.train()
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, (vis, txt) in enumerate(loop):
            with autocast():
                loss, lm, cont = vlm(vis, txt['input_ids'], txt['attention_mask'], txt['labels'])

            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vlm.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                if global_step > 0 and global_step % SAVE_STEP_INTERVAL == 0:
                    step_path = os.path.join(CKPT_SAVE_DIR, f"step_{global_step}.pt")
                    state = {'proj': proj.state_dict(), 'text_model': text_model.state_dict(),
                             'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
                             'global_step': global_step}
                    torch.save(state, step_path)

                    if len(step_checkpoint_queue) == KEEP_LATEST_N:
                        old_path = step_checkpoint_queue.popleft()
                        if os.path.exists(old_path): os.remove(old_path)
                    step_checkpoint_queue.append(step_path)
                    print(f"\n Saved & Cleaned at step {global_step}")

            loop.set_postfix(lm=f"{lm.item():.4f}", cont=f"{cont.item():.4f}", gs=global_step)

        epoch_state = {'proj': proj.state_dict(), 'text_model': text_model.state_dict(),
                       'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
                       'global_step': global_step}
        torch.save(epoch_state, os.path.join(CKPT_SAVE_DIR, f"vlm_epoch_{epoch}.pt"))
        torch.save(epoch_state, CKPT_PATH)