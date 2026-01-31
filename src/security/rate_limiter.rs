use std::sync::Arc;
use parking_lot::Mutex;
use wasm_bindgen::prelude::*;

#[derive(Clone, Debug)]
pub struct RateLimiter {
    tokens: Arc<Mutex<f64>>,
    last_update: Arc<Mutex<f64>>,
    rate: f64,        // tokens per second
    max_tokens: f64,
}

impl RateLimiter {
    pub fn new(rate: f64, max_tokens: f64) -> Self {
        Self {
            tokens: Arc::new(Mutex::new(max_tokens)),
            last_update: Arc::new(Mutex::new(js_sys::Date::now())),
            rate,
            max_tokens,
        }
    }

    pub fn check(&self) -> Result<(), String> {
        let now = js_sys::Date::now();
        let mut tokens_lock = self.tokens.lock();
        let mut last_lock = self.last_update.lock();
        
        // Refill tokens based on elapsed time
        let elapsed_sec = (now - *last_lock) / 1000.0;
        let added_tokens = elapsed_sec * self.rate;
        
        *tokens_lock = (*tokens_lock + added_tokens).min(self.max_tokens);
        *last_lock = now;

        if *tokens_lock >= 1.0 {
            *tokens_lock -= 1.0;
            Ok(())
        } else {
            Err("Rate limit exceeded".to_string())
        }
    }
}
