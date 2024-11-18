---
title: Change Default Applications
subtitle: xdg-mime default nvim.desktop text/plain
date: 2022/10/6 14:02
tags: tech
---
![pixiv/artworks/86432825](https://img1.imgtp.com/2022/10/06/TwONQ2G5.jpg)

# Default Applications

## Using CLI Utils

```bash
xdg-mime query filetype /path/to/file
xdg-mime default <application> <mimetype(s)>
```

We can get the file's mime-type through `xdg-mime query filetype /path/to/the/file`, and using `xdg-mime default <application> <mimetype(s)>` to set the default application for those files.

**Note: `application` is a .desktop file.**

## (Alternative) Edit mimeapps.list Manually

If you want to edit configuration manually or feel really desperate, just modify the `~/.config/mimeapps.list` manually.

Format:
```xml
application/TYPE=LAUNCHER.desktop
```

Just add it under the `[Default Applications]` section if you
want to be default, or under the `[Added Associations]`
section if it shouldn't be default.

**Note: Some programs still use the now deprecated `~/.local/share/applications/mimeapps.list`, however the best is to make that a symlink to `~/.config/mimeapps.list` to have a single config for this:**

```bash
cat ~/.local/share/applications/mimeapps.list >> ~/.config/mimeapps.list
rm ~/.local/share/applications/mimeapps.list
ln -s ~/.config/mimeapps.list  ~/.local/share/applications/mimeapps.list
```

### Reference
[Default Applications -- Arch Wiki](https://wiki.archlinux.org/title/default_applications#xdg-open)

[How do I set the default program?](https://askubuntu.com/questions/90214/how-do-i-set-the-default-program)
