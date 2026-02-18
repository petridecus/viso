//! Dynamic GPU buffer management with automatic resizing
//!
//! Provides buffers that grow automatically when data exceeds capacity,
//! using a 2x growth strategy to minimize reallocations.

use wgpu::util::DeviceExt;

/// A GPU buffer that can grow dynamically
///
/// Uses a 2x growth strategy when capacity is exceeded.
/// Never shrinks (GPU buffers cannot be resized in place).
pub struct DynamicBuffer {
    buffer: wgpu::Buffer,
    capacity: usize, // Capacity in bytes
    len: usize,      // Current data length in bytes
    usage: wgpu::BufferUsages,
    label: String,
}

impl DynamicBuffer {
    /// Buffer with the given initial byte capacity.
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        initial_capacity: usize,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let capacity = initial_capacity.max(64); // Minimum 64 bytes

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            capacity,
            len: 0,
            usage,
            label: label.to_string(),
        }
    }

    /// Buffer initialized from existing data.
    pub fn new_with_data<T: bytemuck::Pod>(
        device: &wgpu::Device,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> Self {
        let data_bytes = bytemuck::cast_slice(data);
        let capacity = data_bytes.len().max(64);

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data_bytes,
            usage: usage | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            buffer,
            capacity,
            len: data_bytes.len(),
            usage,
            label: label.to_string(),
        }
    }

    /// Write data to buffer, growing if necessary
    ///
    /// Returns `true` if buffer was reallocated (bind groups need recreation)
    pub fn write<T: bytemuck::Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[T],
    ) -> bool {
        let data_bytes = bytemuck::cast_slice(data);
        let needed = data_bytes.len();

        let reallocated = if needed > self.capacity {
            // 2x growth, minimum 1KB
            let new_capacity = (needed * 2).max(self.capacity + 1024);

            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&self.label),
                size: new_capacity as u64,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.capacity = new_capacity;
            true
        } else {
            false
        };

        if needed > 0 {
            queue.write_buffer(&self.buffer, 0, data_bytes);
        }
        self.len = needed;

        reallocated
    }

    /// Write raw bytes to buffer, growing if necessary.
    ///
    /// Returns `true` if buffer was reallocated (bind groups need recreation).
    pub fn write_bytes(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> bool {
        let needed = data.len();

        let reallocated = if needed > self.capacity {
            let new_capacity = (needed * 2).max(self.capacity + 1024);

            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&self.label),
                size: new_capacity as u64,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.capacity = new_capacity;
            true
        } else {
            false
        };

        if needed > 0 {
            queue.write_buffer(&self.buffer, 0, data);
        }
        self.len = needed;

        reallocated
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Typed wrapper for DynamicBuffer with cleaner API
///
/// Tracks item count rather than byte length.
pub struct TypedBuffer<T> {
    inner: DynamicBuffer,
    count: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> TypedBuffer<T> {
    /// Default initial capacity: 1000 items.
    pub fn new(device: &wgpu::Device, label: &str, usage: wgpu::BufferUsages) -> Self {
        let initial_capacity = std::mem::size_of::<T>() * 1000;
        Self {
            inner: DynamicBuffer::new(device, label, initial_capacity, usage),
            count: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Specified initial capacity (in items).
    pub fn with_capacity(
        device: &wgpu::Device,
        label: &str,
        capacity: usize,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let initial_capacity = std::mem::size_of::<T>() * capacity;
        Self {
            inner: DynamicBuffer::new(device, label, initial_capacity, usage),
            count: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Typed buffer initialized from existing data.
    pub fn new_with_data(
        device: &wgpu::Device,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> Self {
        Self {
            inner: DynamicBuffer::new_with_data(device, label, data, usage),
            count: data.len(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Write data to buffer, growing if necessary
    ///
    /// Returns `true` if buffer was reallocated (bind groups need recreation)
    pub fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[T]) -> bool {
        self.count = data.len();
        self.inner.write(device, queue, data)
    }

    /// Write raw bytes to buffer, growing if necessary.
    ///
    /// Infers item count from byte length / type size.
    /// Returns `true` if buffer was reallocated (bind groups need recreation).
    pub fn write_bytes(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> bool {
        self.count = data.len() / std::mem::size_of::<T>();
        self.inner.write_bytes(device, queue, data)
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        self.inner.buffer()
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity() / std::mem::size_of::<T>()
    }
}
