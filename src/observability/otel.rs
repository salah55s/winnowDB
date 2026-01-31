//! OpenTelemetry Support for WinnowDB
//! 
//! Provides integration with OpenTelemetry for distributed tracing and metrics export.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// ============================================================================
// Trace Data Models
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpanContext {
    pub trace_id: String,
    pub span_id: String,
    pub is_sampled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SpanKind {
    Internal,
    Server,
    Client,
    Producer,
    Consumer,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Span {
    pub name: String,
    pub context: SpanContext,
    pub parent_span_id: Option<String>,
    pub start_time: f64,
    pub end_time: Option<f64>,
    pub kind: SpanKind,
    pub attributes: HashMap<String, String>,
    pub events: Vec<SpanEvent>,
    pub status: SpanStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpanEvent {
    pub name: String,
    pub timestamp: f64,
    pub attributes: HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SpanStatus {
    Unset,
    Ok,
    Error(String),
}

// ============================================================================
// OTel Exporter
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OTelConfig {
    pub service_name: String,
    pub collector_endpoint: String,
    pub export_interval_ms: u64,
    pub sampling_ratio: f64,
}

#[wasm_bindgen]
pub struct OTelManager {
    config: OTelConfig,
    active_spans: HashMap<String, Span>, // span_id -> Span
}

#[wasm_bindgen]
impl OTelManager {
    #[wasm_bindgen(constructor)]
    pub fn new(service_name: &str, collector_url: &str) -> Self {
        Self {
            config: OTelConfig {
                service_name: service_name.to_string(),
                collector_endpoint: collector_url.to_string(),
                export_interval_ms: 10000,
                sampling_ratio: 1.0, // 100% sampling by default
            },
            active_spans: HashMap::new(),
        }
    }

    /// Start a new span
    pub fn start_span(&mut self, name: &str, trace_id: &str, parent_span_id: Option<String>) -> String {
        let span_id = format!("span_{}", js_sys::Math::random().to_string().replace("0.", ""));
        let span = Span {
            name: name.to_string(),
            context: SpanContext {
                trace_id: trace_id.to_string(),
                span_id: span_id.clone(),
                is_sampled: true,
            },
            parent_span_id,
            start_time: js_sys::Date::now(),
            end_time: None,
            kind: SpanKind::Internal,
            attributes: HashMap::new(),
            events: Vec::new(),
            status: SpanStatus::Unset,
        };
        
        self.active_spans.insert(span_id.clone(), span);
        span_id
    }

    /// End a span
    pub fn end_span(&mut self, span_id: &str) -> JsValue {
        if let Some(mut span) = self.active_spans.remove(span_id) {
            span.end_time = Some(js_sys::Date::now());
            // In a real implementation, this would queue for export
            serde_wasm_bindgen::to_value(&span).unwrap_or(JsValue::NULL)
        } else {
            JsValue::NULL
        }
    }

    /// Add attribute to span
    pub fn add_attribute(&mut self, span_id: &str, key: &str, value: &str) {
        if let Some(span) = self.active_spans.get_mut(span_id) {
            span.attributes.insert(key.to_string(), value.to_string());
        }
    }

    /// Add event to span
    pub fn add_event(&mut self, span_id: &str, name: &str) {
        if let Some(span) = self.active_spans.get_mut(span_id) {
            span.events.push(SpanEvent {
                name: name.to_string(),
                timestamp: js_sys::Date::now(),
                attributes: HashMap::new(),
            });
        }
    }
    
    /// Export traces (Mock) returns JSON for now
    pub fn export_batch(&self) -> String {
        // Mock export logic
        format!("Exporting to {}", self.config.collector_endpoint)
    }
}
