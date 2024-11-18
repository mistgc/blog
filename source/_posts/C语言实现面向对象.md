---
title: C语言实现面向对象
subtitle: Object Oriented Programming.
data: 2021/12/17 13:49:00
tags: tech
---
# C语言实现面向对象
## 0x00.面向对象(OO)是一种思想
> OO: Object Oriented
>
> OOP: Object Oriented Programming



面向对面编程（OOP）的概念我最早是从C++中了解到的，最开始看的那本C++教程书里面说：OOP不是一个编程语言的特性，而是一种思想。 （大概是这样说的）。我认为所谓支持面向对象的编程语言，都是对OOP这种思想加上了语法的限制。  

本文将举一个关于**宠物进食**的例子进行分享。
> 本文中将使用大量指针：
> const type \*var 是定义一个指向常量内容的指针var， 指针本身不是常量。
> type const \*var 和 const type \*var 是一回事。
> type \*const var 是常量指针，初始化是指向哪，就指向哪，以后不变了。但内容不必常量。
> const type \*const var 一个指向常量内容的常量指针。

## 0x01.封装

**封装**就是把一个类中的所有`数据`和`函数（方法）`都打包在一起。

#### `Pet.h`

```C
#ifndef PET_H
#define PET_H
//Pet的属性
typedef struct{
    char *name;
}Pet;
//Pet的成员函数
void Pet_ctor(Pet * const me, char *name);                  //Pet构造函数（Constructor）
void Pet_rename(Pet * const me, char *new_name);            //Pet数据操作函数
char *Pet_getName(Pet * const me);							//Pet接口函数
#endif	//PET_H
```

这里的`Pet *me`就像python中对成员函数传参是第一传入的`self`一样。

> C++、Java中默认传入了`this`

#### `Pet.c`

```c
#include "Pet.h"

//构造函数
void Pet_ctor(Pet *me, char *name){
    me->name = name;
}
//Pet数据操作函数
void Pet_rename(Pet *me, char *new_name){
    me->name = new_name;
}
//Pet接口函数
char *Pet_getName(Pet *me){
    return me->name;
}
```

#### `mian_test1.c`

```c
#include <stdio.h>
#include "Pet.h"

int main(){

	Pet pet1, pet2;
	Pet_ctor(&pet1, "多多");
	Pet_ctor(&pet2, "煤球");

	Pet_rename(&pet1, "花生");

	printf("%s\n",Pet_getName(&pet1));		//输出结果：花生
	printf("%s\n",Pet_getName(&pet2));		//输出结果：煤球

	return 0;
}
```

注：可以通过变量的命名规则来实现访问权限规则，但因为C语言中没有对面向对象编程的语法支持，对于访问权限规则的遵守，全靠程序员的自觉。

## 0x02.继承

继承就是通过一个已有的类来派生出新的类。C语言实现，直接将需要继承的父类对象放在子类的最上面即可。

#### `Dog.h`

```c
#ifndef DOG_H
#define DOG_H
#include "Pet.h"

typedef struct{
	Pet super;
	//子类自己拥有的属性
	int color;		//毛色: 1:白色, 2:黑色, 3:黄色
}Dog;

void Dog_ctor(Dog * const me, char *name, int color);		//Dog构造函数
char *Dog_getColor(Dog * const me);							//Dog接口函数
#endif	//DOG_H
```
#### `Dog.c`
```c
#include "Dog.h"
#include <stdlib.h>

void Dog_ctor(Dog * const me, char *name, int color){
	//首先初始化父类对象
	Pet_ctor((&me->super), name);

	me->color = color;
}

char *Dog_getColor(Dog * const me){
	char *temp = (char *)malloc(sizeof(char) * 5);
	switch(me->color){
		case 1:
			temp = "白色";
			break;
		case 2:
			temp = "黑色";
			break;
		case 3:
			temp = "黄色";
			break;
	}
	return temp;
}
```
#### `main_test_2.c`
```c
#include <stdio.h>
#include "Pet.h"
#include "Dog.h"

int main(){
	Dog dog1, dog2;

	Dog_ctor(&dog1, "来福", 1);		//白色
	Dog_ctor(&dog2, "煤球", 2);		//黑色

	printf("%s: %s\n", Pet_getName(&dog1.super), Dog_getColor(&dog1));		//输出结果：来福:白色
	printf("%s: %s\n", Pet_getName(&dog2.super), Dog_getColor(&dog2));		//输出结果：煤球:黑色

	return 0;
}
```
## 0x03.多态
C++中的多态是由虚函数、虚函数指针和虚函数表实现的。C语言中也能够实现。
在实现多态之前，我们需要先实现`虚函数指针`和`虚函数表`。我们需要对我们的前面编写的代码进行修改：
#### `Pet.h`
```c
#ifndef PET_H
#define PET_H
//Pet的属性
typedef struct{
	struct PetVTBL const * vptr;
    char *name;
}Pet;

typedef struct PetVTBL{
	void (*eat)(Pet  const * const me);
}PetVTBL;

//Pet的成员函数
void Pet_ctor(Pet * const me, char *name);					//Pet构造函数（Constructor）
void Pet_rename(Pet * const me, char *new_name);			//Pet数据操作函数
char *Pet_getName(Pet const * const me);					//Pet接口函数

//加上 static 关键字，该函数只在当前 .c 文件可见 (将该头文件include进去的 .c 文件)
static inline void Pet_eat(Pet * const me){
	(*me->vptr->eat)(me);
}
#endif	//PET_H
```
#### `Pet.c`
```c
#include "Pet.h"
#include <assert.h>
#include <stdio.h>

//Pet的虚函数
//static void Pet_eat_(Pet * const me);
void Pet_eat_(Pet const * const me);

//构造函数
void Pet_ctor(Pet * const me, char *name){
	static PetVTBL const vtbl = {
		&Pet_eat_
	};
    me->name = name;
}
//Pet数据操作函数
void Pet_rename(Pet * const me, char *new_name){
    me->name = new_name;
}
//Pet接口函数
char *Pet_getName(Pet const * const me){
    return me->name;
}
//Pet的虚函数
void Pet_eat_(Pet const * const me){
	//assert(0);	//可以看作纯虚函数
	printf("Pet is eating...");
}
```
#### `Dog.h`
```c
#ifndef DOG_H
#define DOG_H
#include "Pet.h"

typedef struct{
	Pet super;
	//子类自己拥有的属性
	int color;		//毛色: 1:白色, 2:黑色, 3:黄色
}Dog;

void Dog_ctor(Dog * const me, char *name, int color);		//Dog构造函数
char *Dog_getColor(Dog const * const me);					//Dog接口函数
#endif	//DOG_H
```
#### `Dog.c`
```c
#include "Dog.h"
#include <stdlib.h>
#include <stdio.h>

static void Dog_eat_(Pet const * const me);					//Dog重载虚函数表

void Dog_ctor(Dog * const me, char *name, int color){
	//首先初始化父类对象
	static PetVTBL const vtbl= {
		&Dog_eat_
	};
	Pet_ctor((&me->super), name);

	me->super.vptr = &vtbl;									//重载vptr
	me->color = color;
}

char *Dog_getColor(Dog const * const me){
	char *temp = (char *)malloc(sizeof(char) * 5);
	switch(me->color){
		case 1:
			temp = "white";
			break;
		case 2:
			temp = "Black";
			break;
		case 3:
			temp = "Yellow";
			break;
	}
	return temp;
}

static void Dog_eat_(Pet const * const me){
	char *color = Dog_getColor((Dog const * const)me);
	printf("%s dog ,%s is eating...\n", color, Pet_getName(me));
	free(color);					//If not, it will be memory leaks.
}
```

我们再多写一个Cat类

#### `Cat.h`
```c
#ifndef CAT_H
#define CAT_H

#include "Pet.h"

typedef struct{
	Pet super;
	int color;						//1: 白色 2: 黑色
}Cat;

void Cat_ctor(Cat * const me, char *name, int color);
char *Cat_getColor(Cat const * const me);
#endif	//CAT_H
```
#### `Cat.c`
```c
#include "Cat.h"
#include <stdio.h>
#include <stdlib.h>

static void Cat_eat_(Pet const * const me);

void Cat_ctor(Cat * const me, char *name, int color){
	static PetVTBL vtbl = {
		&Cat_eat_
	};
	Pet_ctor(&me->super, name);
	me->super.vptr = &vtbl;				//重载vptr
	me->color = color;
}

char *Cat_getColor(Cat const * const me){
	char *temp = (char *)malloc(sizeof(char) * 5);
	switch(me->color){
		case 1:
			temp = "White";
			break;
		case 2:
			temp = "Black";
			break;
	}
	return temp;
}

static void Cat_eat_(Pet const * const me){
	char *color = Cat_getColor((Cat const * const)me);
	printf("%s cat ,%s is eating...\n", color,Pet_getName(me));
	free(color);					//If not, it will be memory leaks.
}
```
最后我们来写一下main函数，来检测一下:
#### `main_test_3.c`
```c
#include <stdio.h>
#include "Pet.h"
#include "Dog.h"
#include "Cat.h"

int main(){
	Dog dog;
	Cat cat;

	Dog_ctor(&dog, "来福", 2);		//黑色
	Cat_ctor(&cat, "鱼白", 1);		//白色

	printf("%s: %s\n", Pet_getName(&dog.super), Dog_getColor(&dog));
	printf("%s: %s\n", Pet_getName(&cat.super), Cat_getColor(&cat));

	Pet *p;

	p = (Pet *)&dog;
	Pet_eat(p);

	p = (Pet *)&cat;
	Pet_eat(p);

	return 0;
}
```

```c
Output:
来福: Black
鱼白: White
Black dog ,来福 is eating...
White cat ,鱼白 is eating...
```

我们可以看见**main函数**中**父类指针p**指向了**2个子类对象（cat，dog）**，并且完成了**Pet_eat()函数**的调用。实现了面向对象中的**多态**。

## 0xff.结语

在C语言中可以实现OOP。对于封装和继承的实现比较简单，对于多态来说，难点在于虚函数指针与虚函数表。而且调用成员函数时非常难受（对于我来说）。如果要大量使用OOP，建议还是用C++，Java等语言。它们将复杂的过程自动地帮我们实现了，我们就可以不用在语法上花太多时间，而是专心于算法和需求实现上。

我已经将代码上传至gitee: [C-Object-Oriented-Programming: Make C implement Object-Oriented Programming. (gitee.com)](https://gitee.com/mistgc/c-object-oriented-programming)

**Pause.**

