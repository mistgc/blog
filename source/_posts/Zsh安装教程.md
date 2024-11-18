---
title: Zsh食用教程
subtitle: 终极的shell...
date: 2022/2/25
tags: tech
---
# Zsh安装教程

## 下载

- [Zsh](https://github.com/zsh-users/zsh)
- [oh-my-zsh](https://ohmyz.sh/)
- [powerlevel10k](https://gitee.com/romkatv/powerlevel10k) (配置主题时使用)

我们首先查看系统是否拥有 **Zsh**。

```bash
cat /etc/shells
```

如果没有 **Zsh** , 使用不同平台提供的包管理工具进行下载安装：

```bash
# Ubuntu
apt update
apt install zsh
# CentenOS
yum install zsh
# ArchLinux
pacman -S zsh
```

这时再重新执行`cat /etc/shells`查看安装结果。

输出发现有`Zsh`的话，我们直接讲起设置为当前用户的默认shell。

```bash
chsh -s /bin/zsh
```

## 配置

### 自定义配置文件路径

我们需要先了解`Zsh`配置文件的加载顺序。

它默认有10个相关配置文件

```bash
/etc/zshenv
/etc/zprofile
/etc/zshrc
/etc/zlogin
/etc/zlogout
~/.zshenv
~/.zprofile
~/.zshrc
~/.zlogin
~/.zlogout
```

加载顺序为

```bash
/etc/zshenv
~/.zshenv
/etc/zprofile
~/.zprofile
/etc/zshrc
~/.zshrc
/etc/zlogin
~/.zlogin
~/.zlogout
/etc/zlogout
```

一般来说在`etc`文件夹下的配置文件，我们尽量不做修改。但是我们需要自定义用户的配置文件路径，所以我们需要对其进行一定的修改。我们这里只修改`/etc/.zshrc`文件。我们在最后添加一下命令：

```bash
HISTFILE=$HOME/.config/zsh/.zsh-history
source $HOME/.config/zsh/.zshrc
```

大家应该已经猜到了，当`zsh`加载时，会加载至系统配置文件`/etc/zshrc`，这时，我们在该配置文件中添加一句`source`命令，来加载我们自定义的`.zshrc`。

因为 **Zsh** 的内容比较多，为了方便我们进行配置，我们通过 **oh-my-zsh** 来配置 **zsh**。

```bash
# Download oh-my-zsh into the `.config` folder.
git clone git@gitee.com:mirrors/oh-my-zsh.git ~/.config
```

然后我们将`oh-my-zsh/templates/zshrc.zsh-template`中的内容复制到我们的 **zsh** 配置文件中去。

```bash
cp ~/.config/oh-my-zsh/templates/zshrc.zsh-template ~/.config/zsh/.zshrc
```

笔者这里配置主题使用 **powerlevel10k** 主题，所以我们将 `.zshrc` 文件中的`ZSH_THEME="robbyrussell"`  改为 `ZSH_THEME="powerlevel10k/powerlevel10k`。

然后我们去下载 **[powerlevel10k](https://gitee.com/romkatv/powerlevel10k)**。

```bash
# Manual
git clone --depth=1 https://gitee.com/romkatv/powerlevel10k.git ~/powerlevel10k
echo 'source ~/powerlevel10k/powerlevel10k.zsh-theme' >>~/.config/zsh/.zshrc
# Oh-My-Zsh
git clone --depth=1 https://gitee.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.config/oh-my-zsh/custom}/themes/powerlevel10k
```

第一次使用该主题时，需要手动配置。配置过程有引导界面，这里不再赘述。当主题配置完成时，它会自动在`~`目录下生成`.zshrc`和`.p10k.zsh`文件，我们不需要这个`.zshrc`文件了已经，我们将`~/.zshrc`中的内容直接复制到`~/.config/zsh/.zshrc`的后面。笔者这里推荐将`.p10k.zsh`文件留在`~`目录下，当然如果要更改的话，记得将`.zshrc`中的对应值也要改了，不然无法加载该主题。

在有些版本的 **Zsh** 中，第一次启动时，会让手动配置 **Zsh**。我们前面因为将`.zshrc`文件放到`~/.config/zsh`中了，所以每次 **Zsh** 启动时，检测发现没有`~/.zshrc`文件，就会一直让我们进行手动配置。我们可以将这个功能关闭。

```bash
rm /usr/share/zsh/version/scripts/zsh-newuser-install
```
注意：代码中的`version`是以下载的 **Zsh** 的版本号命名的文件夹。

这样就不会每次都让我们进行配置了。



