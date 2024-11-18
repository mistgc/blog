---
title: Static & Inline in C
subtitle: static inline const char* foo() { return "foo"; }
date: 2022/12/31 19:40:00
tags: tech
---

![pixiv/artworks/91518899](https://i.postimg.cc/pTSx9Y13/91518899-p0.png)

# Static & Inline in C

- static global variable
- static function
- inline function

## Static global variable

Just a variable whose lifetime thoughout the whole program.

```c
static int count = 0;
```

## Static function

Because the function in C are by **Default Global**, a static function in C is a function that has a scope that is limited to its object file. This measn that the static function is only visible in its object file.

This feature is used for reusing the same function name in other files.

```c
static void foo() {
    printf("Hello World!\n");
}
```

## Inline function

Keyword `inline` is just an advice for the compiler. It doesn't guarantee that a function is inlined, nor actually that a symbol is generated, if it is needed.

### C99 inline rules

In C99, a function defined `inline` will never, and a function defined `extern inline` will always, emit an  externally visible function.

### inline

If a function definition mentioned `inline` without the declaration, the compiler will issue an error: "undefined reference to 'func'".

```c
// without declaration: int func(inti);
inline int func(int i) {
    return i+1;
}

i = func(i);
```

Therefore a function with `inline` its own needs a declaration (A sensible approach is puting it into a header file).

### static inline

A function defined `static inline`. A local definition may be emitted if required. A program can have mutiple definition for it in different translation units.

A sensible approach would be to put the static inline functions in either a header file if they are to be widely used or just in the source files that use them if they are only ever used from one file.

### extern inline

A function where at least one declaration mentions `inline`, but where some declaration doesn't mention `inline` or does mention `extern`. There must be a definition in the same translation unit. Stand-alone object code is emitted and can be called from other translation units in the program.
