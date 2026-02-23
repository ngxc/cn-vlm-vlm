import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPProcessor, BertTokenizerFast
from PIL import Image
import collections

IMAGE_PATH = r"E:\deep-learning\llm\vlm\test\0.jpg"
CKPT_PATH = r"E:\deep-learning\cn-vlm\step_21000.pt"
MINILLM_PATH = "large.pt"

MAX_LEN = 512
NUM_PLACEHOLDER = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMPERATURE = 0.55
TOP_K = 70
TOP_P = 0.75
MAX_GEN_LEN = 128

PLACEHOLDER_TOKEN = "<image>"

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
                                attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + attn_out
        res = x
        x = self.ln2(x)
        x = self.ff_linear1(x)
        x = self.ff_act(x)
        x = self.ff_linear2(x)
        x = self.ff_dropout(x)
        return x + res


class MiniLLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, max_len=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.layers = nn.ModuleList([MiniLLMBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, inputs_embeds):
        seq_len = inputs_embeds.size(1)
        pos = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        x = inputs_embeds + self.pos_embed(pos)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        for blk in self.layers:
            x = blk(x, attn_mask=causal_mask)
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


@torch.no_grad()
def run_inference(image_path):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    special_tokens = ["<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", PLACEHOLDER_TOKEN]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    text_model = MiniLLM(len(tokenizer), 512, 24, 8, MAX_LEN).to(DEVICE)
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    proj = VisionToTextProj(vision_model.config.hidden_size, 512, num_layers=8).to(DEVICE)

    print(f" 正在从 {CKPT_PATH} 加载权重...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    if ckpt['text_model']['embed.weight'].shape[0] != len(tokenizer):
        if os.path.exists(MINILLM_PATH):
            st = torch.load(MINILLM_PATH, map_location=DEVICE)
            new_emb = text_model.embed.weight.data.clone()
            new_emb[:st['embed.weight'].shape[0]] = st['embed.weight']
            text_model.load_state_dict(st, strict=False)
            text_model.embed.weight.data = new_emb
            text_model.head.weight = text_model.embed.weight
        else:
            raise ValueError("词表不匹配且找不到原始权重文件进行初始化！")

    text_model.load_state_dict(ckpt['text_model'], strict=False)
    proj.load_state_dict(ckpt['proj'])

    text_model.eval()
    vision_model.eval()
    proj.eval()
    print(" 模型准备就绪。")

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    user_id = tokenizer.encode("<|user|>", add_special_tokens=False)
    assist_id = tokenizer.encode("<|assistant|>", add_special_tokens=False)
    img_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)

    input_ids = [im_start_id] + user_id + [img_id] * NUM_PLACEHOLDER + [im_end_id] + [im_start_id] + assist_id
    input_ids = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)

    image = Image.open(image_path).convert("RGB")
    vis_inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    vis_norm = nn.LayerNorm(vision_model.config.hidden_size).to(DEVICE)
    vis_feats = vis_norm(vision_model(**vis_inputs).last_hidden_state)
    vis_proj = proj(vis_feats)

    current_embeds = text_model.embed(input_ids)
    mask = (input_ids == img_id)[0]
    idxs = torch.nonzero(mask, as_tuple=True)[0]

    v_len = vis_proj.size(1)
    chunk_indices = torch.linspace(0, v_len, steps=NUM_PLACEHOLDER + 1).long()
    for i, idx in enumerate(idxs):
        if i >= NUM_PLACEHOLDER: break
        start, end = chunk_indices[i], chunk_indices[i + 1]
        current_embeds[0, idx] = vis_proj[0, start:end].mean(dim=0) if start < end else vis_proj[0, min(start, v_len - 1)]

    generated_tokens = []
    print("\n 模型正在描述图片...")

    for _ in range(MAX_GEN_LEN):
        outputs = text_model(inputs_embeds=current_embeds)
        next_token_logits = outputs[0, -1, :]

        if TEMPERATURE > 0:
            logits = next_token_logits / TEMPERATURE

            if TOP_K > 0:
                v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
                logits[logits < v[-1]] = -float('Inf')

            if TOP_P < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > TOP_P
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                logits[sorted_indices[sorted_indices_to_remove]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        token_id = next_token_id.item()
        if token_id == im_end_id: break

        generated_tokens.append(token_id)

        next_emb = text_model.embed(next_token_id.unsqueeze(0))
        current_embeds = torch.cat([current_embeds, next_emb], dim=1)

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f" 找不到图片: {IMAGE_PATH}")
    else:
        result = run_inference(IMAGE_PATH)
        print("\n 模型回答:")
        print(result)