---
title: Learn Tauri and Tokio
tags: Tech; Rust
---

If we want to create some async task outside of Tauri, then we need to own and manage the Tokio runtime.

```rust
#[tokio::main]
async fn main() {
    tauri::async_runtime::set(tokio::runtime::Handle::current());

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```
