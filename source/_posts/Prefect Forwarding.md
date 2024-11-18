---
title: Perfect Forwarding in C++
subtitle: Perfect!!!
date: 2023/8/29 14:28:00
tags: tech
---

## Universal Reference

There is only two reference: *lvalue* reference *rvalue* reference.

The **Universal Reference** is not a type of reference. It just exists in the "Template".

```cpp
#include <iostream>

void process(int &i) {
    std::cout << i << " lvalue\n";
}

void process(int &&i) {
    std::cout << i << " rvalue\n";
}

template <typename T>
void test(T&& v) {
    process(v);
}

int main() {
    int i = 0;

    test(i);            // output: 0 lvalue
    test(1);            // output: 1 lvalue

    return 0;
}
```

Let's take a closer look at the code above. We will find that those both output as the "*lvalue*".
Because the *rvalue* passed into the function will own the **name**, then it could be got its *address* (Aka it's a *lvalue* now.).

## Reference Collapsing

Declaring the reference of a reference is illegal, but the compiler can generate a referenced reference during **Template Instantiation**.
During **Template Instantiation**, **Reference Collapsing** can occur in this situation. If either of the references is an *lvalue* reference,
the result will be an *lvalue* reference. If both are *rvalue* references, the result will be an *rvalue* reference.

```cpp
template <typename T>
void test_(T&& t) {
    std::cout << "is int&" << std::is_same_v<T, int&> << '\n';
    std::cout << "is int" << std::is_same_v<T, int> << '\n';
}

int main() {
    std::cout << std::boolalpha;
    int i = 1;
    test_(i);           // void test_(T &&t) => void test_(int & &&t) => void test_(int &t): T = int &
    test_(3);           // void test_(T &&t) => void test_(int &&t)                        : T = int

    return 0;
}
```
If we use the `static_cast<T>()` to cast `T` to `T&&`.

```cpp
#include <iostream>

void process(int &i) {
    std::cout << i << " lvalue\n";
}

void process(int &&i) {
    std::cout << i << " rvalue\n";
}

template <typename T>
void test(T&& v) {
    // test(i) => T: int& => T&& = int& &&  = int&
    // output: 0 lvalue

    // test(1) => T: int  => T&& = int&&
    // output: 1 rvalue

    process(static_cast<T&&>(v));
}

int main() {
    int i = 0;

    test(i);            // output: 0 lvalue
    test(1);            // output: 1 rvalue

    return 0;
}

```

## Function `forward<T>()`

Now, we attempt to encapsulate this functionality to a function.
This function returns an *lvalue* reference when passed an *lvalue* reference, and returns an *rvalue* reference when passed an *rvalue* reference.

```cpp
#include <iostream>

// receive the rvalue ref
template <typename T>
T&& forward(T&& v) {
    return static_cast<T&&>(v);
}

// receive the lvalue ref
template <typename T>
T&& forward(T& v) {
    return static_cast<T&&>(v);
}

void process(int &i) {
    std::cout << i << " lvalue\n";
}

void process(int &&i) {
    std::cout << i << " rvalue\n";
}

template <typename T>
void test(T&& v) {
    process(forward<T&&>(v));
}

int main() {
    int i = 0;

    test(i);            // output: 0 lvalue
    test(1);            // output: 1 rvalue

    return 0;
}
```

Note: Code in Godbolt: https://godbolt.org/z/zo85Exnez

## Reference

https://www.bilibili.com/video/BV1rG4y1V7ia
