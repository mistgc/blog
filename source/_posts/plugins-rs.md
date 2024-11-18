---
title: Plugins in Rust
subtitle: dlopen / libloading
date: 2022/10/18 21:31:00
tags: tech
---
![pixiv/artworks/78511187](https://img1.imgtp.com/2022/10/18/jwMRosrr.jpg)
## Rust plugins

This document ia a guide for setting the Rust application
up with Rust plugins that can be loaded dynamically
at runtime.

Additionally, these plugins can make calls to the 
application's public API, so that it can be invoked with same data structures and utilities for extending the
application.

## The first compile App

Rust has packed everything that you app needs to run inside of one executable.

So, by default, Rust will link all dependencies statically into the final executable.

**Thus our plugins can't use the functions and utilities in our application's library.**

In order to solve it, we should use `dynamic linking`.

## Modify the Cargo.toml file

We should tell compiler to compile a dynamic library and a rust library.

Open and edit Cargo.toml to add following code:

```toml
[lib]
crate-type = ["dylib", "rlib"]
```

**The `dylib` makes a `.so` file that contain the machine code.**

**The `rlib` makes a '.rlib' file which is like a header file that provide rust with extra metadata. Without `.rlib`, our application couldn't know that which functions and utilities in the library.**

Meanwhile we need to tell cargo to add some flags to the its rust compiler calls.
These settings go in a `.cargo/config` file:

```toml
[build]
rustflags = ["-C", "prefer-dynamic", "-C", "rpath"]
```

## Time to use extra crate

Now, we create a new project named `plugin1` alongside the app project.

And add the following code in the `plugin1`'s `Cargo.toml`.

```toml
[lib]
crate-type = ["dylib"]
```

And insert another code in `.cargo/config.toml`

```toml
 rustflags = ["-C", "prefer-dynamic", "-C", "rpath"]
```

Here we can edit `Cargo.toml` to associate it to `app` library. But the `app` library would also re-compile, when we already have the app compiled. There is no reason to compile the app library twice.

**So, here we specify `app` as an external crate.**

The `#[no_mangle]` attribute on a function tells the compiler not to add any extra metadata to that symbol in the compiled output, and this allows us to call the function by name when we later load it into our app dynamically.

For example:

```rust
#[no_mangle]
pub fn run() {
    println!("Running plugin1");
    app::test_app_func("Hello from plugin 1");
}
```

Now we attempt to `cargo build` the crate but it will tells us that it can't find the `app` crate. This is we didn't tell the cargo where is the `app` library.

Therefor we should create a `build.rs` script that can be used to do any kind of necessary setup to compile a library. In this case we just need to feed cargo some specially understood flags that tell it how to find our pre-compiled `app` library.

Like this:

```rust
// build.rs

fn main() {
    // Add our app's build directory to the lib search path.
    println!("cargo:rustc-link-search=../app/target/debug");
    // Add the app's dependency directory to the lib search path.
    // This is may be required if the app depends on any external "derive"
    // crates like the `dlopen_derive` crate that we add later.
    println!("cargo:rustc-link-search=../app/target/debug/deps");
    // Link to the `app` crate library. This tells cargo to actually link
    // to the `app` crate that we include using `extern crate app;`.
    println!("cargo:rustc-link-lib=app");
}
```

Now we can run `cargo build` and we will get a new `libplugin1.so` file in our `target/debug`.

> Note: If you run cargo build and get an error like `error[E0464]: multiple matching crates for 'app'`, change directory to your app directory and run `cargo clean` followed by `cargo build`.
> This will get rid of any extra `rlib` file that may have been left over from when we first bulit our app as a standalone binnary.
> After doing that you should be able to come back to your plugin and successfully run `cargo build` to build the library.

## Loading a Plugin

Now we can load the plugin into our app. To load plugins we are going to use the 
`[dlopen](https://crates.io/crates/dlopen)` crate. The `dlopen` crate will do the actual loading of the shared libraries and takes care of the lower level stuff so we don't have to. Then, let's add that crate to the `Cargo.toml` for our app.

```toml
dlopen = "0.1.6"
dlopen_derive = "0.1.3"
```

And update our app's code. For example:

```rust
#[macro_use]
extern crate dlopen_derive;
use dlopen::wrapper::{Container, WrapperApi};

#[derive(WrapperApi)]
struct PluginApi {
    run: extern fn(),
}

pub fn run() {
    println!("Starting App");

    let plugin_api_wrapper: Container<PluginApi> = unsafe { Container::load("plugins/libplugin1.so") }.unwrap();
    plugin_api_wrapper.run();
}

pub fn test_app_func(message: &str) {
    println!("test_app_func(\"{}\")", message);
}
```

## References

[rust-plugins](https://zicklag.github.io/rust-tutorials/rust-plugins.html)

[rust-dlopen](https://github.com/szymonwieloch/rust-dlopen)
