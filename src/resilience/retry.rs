use js_sys::Promise;
use wasm_bindgen_futures::JsFuture;
use web_sys::console;

pub struct RetryConfig {
    pub max_retries: usize,
    pub base_delay_ms: u32,
    pub max_delay_ms: u32,
    pub factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 2000,
            factor: 2.0,
        }
    }
}

pub async fn retry_async<F, Fut, T, E>(config: RetryConfig, mut op: F) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut attempt = 0;
    let mut delay = config.base_delay_ms;

    loop {
        match op().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                attempt += 1;
                if attempt > config.max_retries {
                     // Log failure?
                     return Err(e);
                }
                
                // Wait (Sleep)
                let promise = Promise::new(&mut |resolve, _| {
                    if let Ok(wins) = web_sys::window().ok_or("no window") {
                        let _ = wins.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, delay as i32);
                    } else {
                        // Fallback/No-op if no window (Node env?)
                        // In node, setTimeout is global but different signature?
                        // Actually web_sys window works in many checks, but not in worker/node potentially.
                        // We assume strict wasm-bindgen env for async wait.
                         let _ = resolve.call0(&wasm_bindgen::JsValue::NULL);
                    }
                });
                let _ = JsFuture::from(promise).await;

                // Backoff
                delay = (delay as f64 * config.factor) as u32;
                if delay > config.max_delay_ms { delay = config.max_delay_ms; }
            }
        }
    }
}
