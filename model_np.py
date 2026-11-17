"""Er Translator (nn.Transformer, teckennivå) skriven i ren NumPy.
Körs i webbläsaren via Pyodide. Ingen dropout (den används bara vid träning)."""
import json
import numpy as np

NHEAD = 8          # måste vara samma som NHEAD i notebooken
MAX_LEN = 120      # max antal tecken i indata (som i träningen)
LN_EPS = 1e-5      # PyTorchs standardvärde för LayerNorm


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def layer_norm(x, w, b):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + LN_EPS) * w + b


class Translator:
    def __init__(self, weights_path, vocab_path):
        npz = np.load(weights_path)
        self.w = {k: npz[k].astype(np.float32) for k in npz.files}
        with open(vocab_path, encoding="utf-8") as f:
            self.token2idx = json.load(f)
        self.idx2token = {i: t for t, i in self.token2idx.items()}
        self.n_enc = 1 + max(int(k.split(".")[3]) for k in self.w
                             if k.startswith("transformer.encoder.layers."))
        self.n_dec = 1 + max(int(k.split(".")[3]) for k in self.w
                             if k.startswith("transformer.decoder.layers."))
        self.emb = self.w["embedding.input_embedding.weight"]   # delas med fc
        self.pe = self.w["embedding.pos_embedding.weight"]
        self.d_model = self.emb.shape[1]
        self.max_pos = self.pe.shape[0]

    # ---- byggstenar ----
    def embed(self, ids):
        x = self.emb[ids] * np.sqrt(self.d_model)
        return x + self.pe[: len(ids)]

    def attention(self, prefix, q_in, kv_in, mask=None):
        w = self.w
        Wq, Wk, Wv = np.split(w[prefix + ".in_proj_weight"], 3)
        bq, bk, bv = np.split(w[prefix + ".in_proj_bias"], 3)
        q = q_in @ Wq.T + bq
        k = kv_in @ Wk.T + bk
        v = kv_in @ Wv.T + bv
        dk = self.d_model // NHEAD

        def split_heads(x):  # (T, d) -> (huvuden, T, dk)
            return x.reshape(x.shape[0], NHEAD, dk).transpose(1, 0, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(dk)
        if mask is not None:
            scores = scores + mask
        out = softmax(scores) @ v                       # (huvuden, T, dk)
        out = out.transpose(1, 0, 2).reshape(q_in.shape[0], self.d_model)
        return out @ w[prefix + ".out_proj.weight"].T + w[prefix + ".out_proj.bias"]

    def ffn(self, prefix, x):
        w = self.w
        h = np.maximum(0, x @ w[prefix + ".linear1.weight"].T + w[prefix + ".linear1.bias"])
        return h @ w[prefix + ".linear2.weight"].T + w[prefix + ".linear2.bias"]

    def norm(self, name, x):
        return layer_norm(x, self.w[name + ".weight"], self.w[name + ".bias"])

    # ---- encoder / decoder ----
    def encode(self, src_ids):
        x = self.embed(src_ids)
        for i in range(self.n_enc):
            p = f"transformer.encoder.layers.{i}"
            x = self.norm(p + ".norm1", x + self.attention(p + ".self_attn", x, x))
            x = self.norm(p + ".norm2", x + self.ffn(p, x))
        return self.norm("transformer.encoder.norm", x)

    def decode(self, tgt_ids, memory):
        x = self.embed(tgt_ids)
        T = len(tgt_ids)
        causal = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
        for i in range(self.n_dec):
            p = f"transformer.decoder.layers.{i}"
            x = self.norm(p + ".norm1", x + self.attention(p + ".self_attn", x, x, causal))
            x = self.norm(p + ".norm2", x + self.attention(p + ".multihead_attn", x, memory))
            x = self.norm(p + ".norm3", x + self.ffn(p, x))
        return self.norm("transformer.decoder.norm", x)

    def logits(self, src_ids, tgt_ids):
        """Samma sak som model(src, tgt)[0] i PyTorch (för jämförelse)."""
        out = self.decode(np.array(tgt_ids), self.encode(np.array(src_ids)))
        return out @ self.emb.T + self.w["fc.bias"]

    # ---- översättning (greedy, som i notebooken) ----
    def translate(self, sentence):
        t2i = self.token2idx
        sentence = sentence.strip()[:MAX_LEN]
        src = [t2i["<s>"]] + [t2i.get(c, t2i["<unk>"]) for c in sentence] + [t2i["</s>"]]
        memory = self.encode(np.array(src))
        tgt = [t2i["<s>"]]
        for _ in range(min(len(src) + 20, self.max_pos) - 1):
            out = self.decode(np.array(tgt), memory)
            nxt = int((out[-1] @ self.emb.T + self.w["fc.bias"]).argmax())
            tgt.append(nxt)
            if nxt == t2i["</s>"]:
                break
        chars = [self.idx2token[i] for i in tgt[1:]]
        if chars and chars[-1] == "</s>":
            chars = chars[:-1]
        return "".join(chars)


def load():
    """Anropas från webbsidan."""
    return Translator("weights.npz", "pico_translator.vocab").translate
