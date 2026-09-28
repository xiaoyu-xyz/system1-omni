//! The embeddings client: the engine's half of the split.
//!
//! CLM does not compute embeddings. A frozen Qwen3-8B encoder runs as its own process
//! behind an OpenAI-compatible `/v1/embeddings` endpoint, and this module is the client
//! to it. Everything the engine owns happens after the vectors come back.
//!
//! The wire shape follows `src/clm/embedder.py` in the CLM reference: a POST of
//! `{model, input, encoding_format}`, a base64 `f32` payload per input in `index` order,
//! and an `l2` normalisation applied on receipt — the encoder is asked for raw vectors
//! and the client normalises, so a server that already normalises is harmless.
use anyhow::{Context, Result, bail, ensure};
use std::time::Duration;

/// Sends texts to the encoder and returns one row per text, L2-normalised.
///
/// A trait rather than a struct so the engine can be driven without an encoder; the
/// tests use [`HashingEncoder`], and a deployment uses [`HttpEncoder`].
pub trait Encoder: Send + Sync {
    /// One `hidden_size`-wide vector per input, in the order given.
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Whether the encoder is reachable, for readiness.
    fn healthy(&self) -> bool;
}

/// L2-normalise one row, the way `torch.nn.functional.normalize` does.
pub fn normalize(row: &mut [f32]) {
    let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
}

/// A deterministic encoder with no server behind it.
///
/// The vector is derived from a SHA-256 of the text filled to `dim` and then normalised,
/// which is exactly what `recipe/clm/native/head_oracle.py` does — so a decision taken
/// against this encoder can be compared with the Python oracle, and the whole engine can
/// be exercised on a machine with no GPU and no weights. The values carry no meaning as
/// model output; the point is that both implementations see the same numbers.
pub struct HashingEncoder {
    dim: usize,
}

impl HashingEncoder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// The vector for one text, as both this encoder and the oracle compute it.
    pub fn vector(text: &str, dim: usize) -> Vec<f32> {
        use sha2::{Digest, Sha256};

        let mut out: Vec<f32> = Vec::with_capacity(dim);
        let mut counter = 0u32;
        while out.len() < dim {
            let digest = Sha256::digest(format!("{counter}:{text}").as_bytes());
            for chunk in digest.as_chunks::<4>().0 {
                if out.len() == dim {
                    break;
                }
                out.push(u32::from_be_bytes(*chunk) as f64 as f32 / 2f64.powi(31) as f32 - 1.0);
            }
            counter += 1;
        }
        normalize(&mut out);
        out
    }
}

impl Encoder for HashingEncoder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| Self::vector(t, self.dim)).collect())
    }

    fn healthy(&self) -> bool {
        true
    }
}

/// An OpenAI-compatible `/v1/embeddings` endpoint, which is what `vllm serve --runner
/// pooling` exposes.
pub struct HttpEncoder {
    url: String,
    model: String,
    client: reqwest::blocking::Client,
    /// Batching, as `embedder.py` does: a request carries at most this many inputs.
    batch: usize,
}

impl HttpEncoder {
    pub fn new(
        url: impl Into<String>,
        model: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()
            .context("build the embeddings HTTP client")?;
        Ok(Self {
            url: url.into(),
            model: model.into(),
            client,
            batch: 512,
        })
    }

    /// The endpoint that serves `/v1/models`, which readiness checks.
    fn models_url(&self) -> String {
        match self.url.rsplit_once("/v1/") {
            Some((base, _)) => format!("{base}/v1/models"),
            None => format!("{}/v1/models", self.url.trim_end_matches('/')),
        }
    }

    fn fetch(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, u64)> {
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
            "encoding_format": "base64",
        });
        let response = self
            .client
            .post(&self.url)
            .json(&body)
            .send()
            .with_context(|| format!("embeddings request to {}", self.url))?;
        let status = response.status();
        let payload: serde_json::Value = response
            .json()
            .with_context(|| format!("embeddings response from {} was not JSON", self.url))?;
        if !status.is_success() {
            bail!("embeddings endpoint returned {status}: {payload}");
        }

        let data = payload["data"]
            .as_array()
            .context("embeddings response has no data array")?;
        let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        for row in data {
            let index = row["index"].as_u64().context("a data row has no index")? as usize;
            ensure!(index < out.len(), "data index {index} is out of range");
            let encoded = row["embedding"]
                .as_str()
                .context("embedding is not a base64 string")?;
            let raw = base64_decode(encoded).context("embedding is not valid base64")?;
            ensure!(
                raw.len() % 4 == 0,
                "embedding payload is {} bytes, not a whole number of f32",
                raw.len()
            );
            let mut row: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            normalize(&mut row);
            out[index] = Some(row);
        }
        let tokens = payload["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
        out.into_iter()
            .enumerate()
            .map(|(i, v)| v.with_context(|| format!("no embedding for input {i}")))
            .collect::<Result<Vec<_>>>()
            .map(|v| (v, tokens))
    }
}

impl Encoder for HttpEncoder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch) {
            let (rows, _tokens) = self.fetch(chunk)?;
            out.extend(rows);
        }
        Ok(out)
    }

    fn healthy(&self) -> bool {
        self.client
            .get(self.models_url())
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

/// Minimal base64 decode, so the client does not pull a dependency for one function.
fn base64_decode(input: &str) -> Result<Vec<u8>> {
    const fn table() -> [i8; 256] {
        let mut t = [-1i8; 256];
        let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0;
        while i < 64 {
            t[alphabet[i] as usize] = i as i8;
            i += 1;
        }
        t
    }
    const T: [i8; 256] = table();
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for &b in bytes {
        if b == b'=' || b == b'\n' || b == b'\r' {
            continue;
        }
        let v = T[b as usize];
        if v < 0 {
            bail!("invalid base64 byte {b:#x}");
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}
