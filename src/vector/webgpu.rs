use wasm_bindgen::prelude::*;

use std::borrow::Cow;
use std::collections::HashMap;
use crate::vector::metric::MetricType;

// Uniforms struct matching WGSL
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    dim: u32,
    count: u32,
    extra_param: f32, // Used for query_norm (Cosine) or others
    _pad: u32,       // Alignment to 16 bytes
}

#[wasm_bindgen]
pub struct IndexWebGPU {
    device: wgpu::Device,
    queue: wgpu::Queue,
    
    // Pipelines for each metric
    pipelines: HashMap<i32, wgpu::ComputePipeline>, // Keyed by MetricType discriminant
    
    // State
    dim: usize,
    capacity: usize,
    count: usize,
    
    // Buffers
    vector_buffer: wgpu::Buffer,
    result_buffer: wgpu::Buffer,
    query_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    staging_mapped: bool, // Guard state
    
    bind_group: Option<wgpu::BindGroup>,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[wasm_bindgen]
impl IndexWebGPU {
    pub async fn new(dim: usize, initial_capacity: usize) -> Result<IndexWebGPU, JsValue> {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or(JsValue::from_str("No WebGPU adapter found"))?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("WinnowDB Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ).await.map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Helper to generate shader source
        let create_shader_src = |metric: MetricType| -> String {
            let core_logic = match metric {
                MetricType::L2 => r#"
                    var dist_sq: f32 = 0.0;
                    for (var i: u32 = 0u; i < dim; i = i + 1u) {
                        let diff = vectors[index * dim + i] - query[i];
                        dist_sq = dist_sq + (diff * diff);
                    }
                    results[index] = sqrt(dist_sq);
                "#,
                MetricType::InnerProduct => r#"
                    var dot: f32 = 0.0;
                    for (var i: u32 = 0u; i < dim; i = i + 1u) {
                        dot = dot + (vectors[index * dim + i] * query[i]);
                    }
                    results[index] = dot;
                "#,
                MetricType::Cosine => r#"
                    var dot: f32 = 0.0;
                    var norm_v_sq: f32 = 0.0;
                    for (var i: u32 = 0u; i < dim; i = i + 1u) {
                        let v = vectors[index * dim + i];
                        let q = query[i];
                        dot = dot + (v * q);
                        norm_v_sq = norm_v_sq + (v * v);
                    }
                    let norm_v = sqrt(norm_v_sq);
                    let norm_q = uniforms.extra_param; // Pre-calculated query norm
                    
                    if (norm_v == 0.0 || norm_q == 0.0) {
                        results[index] = 0.0;
                    } else {
                        results[index] = dot / (norm_v * norm_q);
                    }
                "#,
            };
            
            format!(r#"
                struct Uniforms {{
                    dim: u32,
                    count: u32,
                    extra_param: f32, // query_norm or unused
                }};
                @group(0) @binding(0) var<uniform> uniforms: Uniforms;
                @group(0) @binding(1) var<storage, read> vectors: array<f32>;
                @group(0) @binding(2) var<storage, read> query: array<f32>;
                @group(0) @binding(3) var<storage, read_write> results: array<f32>;

                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                    let index = global_id.x;
                    if (index >= uniforms.count) {{
                        return;
                    }}
                    let dim = uniforms.dim;
                    {}
                }}
            "#, core_logic)
        };

        // Create Pipelines
        let metrics = [MetricType::L2, MetricType::InnerProduct, MetricType::Cosine];
        let mut pipelines = HashMap::new();
        let mut bind_group_layout = None;

        for metric in metrics {
             let shader_src = create_shader_src(metric);
             let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("Shader {:?}", metric)),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_src)),
            });
            
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Pipeline {:?}", metric)),
                layout: None,
                module: &shader,
                entry_point: "main",
            });
            
            if bind_group_layout.is_none() {
                bind_group_layout = Some(pipeline.get_bind_group_layout(0));
            }
            pipelines.insert(metric as i32, pipeline);
        }

        // Initialize Buffers
        let (vector_buffer, result_buffer, query_buffer, uniform_buffer, staging_buffer) = 
            Self::create_buffers(&device, dim, initial_capacity);

        Ok(IndexWebGPU {
            device,
            queue,
            pipelines,
            bind_group_layout: bind_group_layout.unwrap(),
            dim,
            capacity: initial_capacity,
            count: 0,
            vector_buffer,
            result_buffer,
            query_buffer,
            uniform_buffer,
            staging_buffer,
            staging_mapped: false,
            bind_group: None,
        })
    }

    // Helper to create buffers
    fn create_buffers(device: &wgpu::Device, dim: usize, capacity: usize) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let vector_size = (capacity * dim * 4) as u64;
        let vector_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vector Buffer"),
            size: vector_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Buffer"),
            size: (dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let result_size = (capacity * 4) as u64;
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Buffer"),
            size: result_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: result_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        (vector_buffer, result_buffer, query_buffer, uniform_buffer, staging_buffer)
    }

    fn resize_buffers(&mut self, new_capacity: usize) {
        let (vb, rb, qb, ub, sb) = Self::create_buffers(&self.device, self.dim, new_capacity);
        self.vector_buffer = vb;
        self.result_buffer = rb;
        self.query_buffer = qb;
        self.uniform_buffer = ub;
        self.staging_buffer = sb;
        self.staging_mapped = false; // New buffer starts unmapped
        self.capacity = new_capacity;
        // Need to recreate bind group immediately? 
        // No, init() calls recreate_bind_group anyway.
    }

    pub fn init(&mut self, flat_vectors: &[f32], count: usize) -> Result<(), JsValue> {
        if flat_vectors.len() != count * self.dim {
             return Err(JsValue::from_str("Vector size mismatch"));
        }
        
        // Auto-Resize
        if count > self.capacity {
            let new_cap = (count * 3) / 2; // Grow 1.5x
            self.resize_buffers(new_cap);
        }

        self.queue.write_buffer(&self.vector_buffer, 0, bytemuck::cast_slice(flat_vectors));
        self.count = count;
        self.update_uniforms(0.0); 
        self.recreate_bind_group();
        Ok(())
    }
    
    fn update_uniforms(&self, extra_param: f32) {
        let uniforms = Uniforms {
            dim: self.dim as u32,
            count: self.count as u32,
            extra_param,
            _pad: 0,
        };
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
    
    fn recreate_bind_group(&mut self) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.vector_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.query_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.result_buffer.as_entire_binding() },
            ],
        });
        self.bind_group = Some(bind_group);
    }

    pub async fn search(&mut self, query: &[f32], k: usize, metric: MetricType) -> Result<JsValue, JsValue> {
        // Guard: If buffer was left mapped (e.g. previous future dropped), unmap it now.
        if self.staging_mapped {
             self.staging_buffer.unmap();
             self.staging_mapped = false;
        }

        if self.count == 0 { return Ok(JsValue::NULL); }
        
        // 1. Prepare Query & Uniforms
        self.queue.write_buffer(&self.query_buffer, 0, bytemuck::cast_slice(query));
        
        let mut extra_param = 0.0;
        if let MetricType::Cosine = metric {
             let mut sum = 0.0;
             for v in query { sum += v * v; }
             extra_param = sum.sqrt();
        }
        self.update_uniforms(extra_param);
        
        // 2. Select Pipeline
        let pipeline = self.pipelines.get(&(metric as i32))
            .ok_or(JsValue::from_str("Unsupported metric"))?;
        
        // 3. Dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            let workgroups = (self.count as u32 + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        let result_size = (self.count * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.result_buffer, 0, &self.staging_buffer, 0, result_size);
        self.queue.submit(Some(encoder.finish()));
        
        // 4. Readback
        let buffer_slice = self.staging_buffer.slice(0..result_size);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| { tx.send(v).unwrap(); });
        
        // Mark as mapped ASAP (before await point)
        self.staging_mapped = true;
        
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);
        
        rx.await
            .map_err(|_| JsValue::from_str("GPU mapping failed"))?
            .map_err(|e| JsValue::from_str(&format!("Buffer Async Error: {:?}", e)))?;
        
        let data = buffer_slice.get_mapped_range();
        let scores: &[f32] = bytemuck::cast_slice(&data);
        
        // 5. Sort via CPU
        let mut indexed_scores: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        drop(data);
        self.staging_buffer.unmap();
        self.staging_mapped = false; // Safely unmapped
            
        match metric {
            MetricType::L2 => indexed_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)),
            _ => indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)),
        }

        let top_k_indices: Vec<(usize, f32)> = indexed_scores.into_iter().take(k).collect();
        Ok(serde_wasm_bindgen::to_value(&top_k_indices)?)
    }
    pub fn dispose(&mut self) {
        self.vector_buffer.destroy();
        self.result_buffer.destroy();
        self.query_buffer.destroy();
        self.uniform_buffer.destroy();
        self.staging_buffer.destroy();
        self.bind_group = None;
    }
}
