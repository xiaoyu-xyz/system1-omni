# Metal backend

Planned home for high-performance Apple GPU operations and Metal kernel integration. Implement the operations required by the first model, with hardware-specific optimizations where needed.

Model orchestration, batching policy, state management, and kernel selection remain with the model engine. CUDA and Metal implementations do not need identical internal structures or a universal tensor abstraction.

Status: planned; no Metal implementation or validated hardware coverage yet. Reference results from other runtimes do not establish native Metal backend support.
