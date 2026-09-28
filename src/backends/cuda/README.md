# CUDA backend

Planned home for high-performance NVIDIA GPU operations and kernel integration. Implement the operations required by the first model, with hardware-specific optimizations where needed.

Model orchestration, batching policy, state management, and kernel selection remain with the model engine. CUDA and Metal implementations do not need identical internal structures or a universal tensor abstraction.

Status: planned; no CUDA implementation or validated hardware coverage yet.
