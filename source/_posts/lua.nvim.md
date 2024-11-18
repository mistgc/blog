---
title: From init.vim to init.lua (1)
subtitle: Lua, Yes.
date: 2022/6/26 00:41:00
tags: tech
---
![Neovim by lua](https://img1.imgtp.com/2022/06/24/Tw2U0IQs.png)
# From init.vim to **init.lua**

## Preface
The nvim supported **lua** script since Nvim 0.5. And *lua* is faster than *vimscript*. Even so, Neovim is not deprecating Vimscript.

> **Will Neovim deprecate Vimscript?**
> 
> No. Lua is built-in, but Vimscript is supported with the **most advanced Vimscript engine in the world** (featuring an AST-producing [parser](https://neovim.io/doc/user/api.html#nvim_parse_expression())).

## Why Lua?
Lua is a powerful, efficient, lightweight, embeddable scripting language. It supports procedural programming, object-oriented programming, functional programming, data-driven programming, and data description. 

[Get started with lua in 30 minutes](https://www.bilibili.com/video/BV1vf4y1L7Rb)

## init.lua, Yes!
We need a lot of plugins when we use Neovim. So there is a plugin manager we need. Here I recommend '[packer.nvim](https://github.com/wbthomason/packer.nvim)'.

### Installation

##### Unix, Linux Installation
```zsh
git clone --depth 1 https://github.com/wbthomason/packer.nvim\
 ~/.local/share/nvim/site/pack/packer/start/packer.nvim
```

##### Windows Powershell Installation
```zsh
git clone https://github.com/wbthomason/packer.nvim "$env:LOCALAPPDATA\nvim-data\site\pack\packer\start\packer.nvim"
```

### Configuration & Usage
This is directory tree of my configuration:

```zsh
nvim
├── init.lua
├── lua
│   ├── conf
│   │   ├── plugin1_detaill-conf.lua
│   │   └── plugin2_detaill-conf.lua
│   ├── keymaps.lua
│   ├── localconf.lua
│   ├── lspconf.lua
│   └── plugins.lua
└── plugin
    └── packer_compiled.lua
```

The entrance file changed from `init.vim` to **`init.lua`**.

Let's look at the `init.lua`.

```lua
-- nvim/init.lua
require('plugins')
require('lspconf')
require('keymaps')
require('localconf')
```

The `require` function can invoke others `.lua` files.

> Why `require('lua/plugins')` ?
>
> Because the path `~/.config/nvim/lua` is contained in the search path by Lua that Nvim built-in.
>
> Let's omit it.

The extremely important file is `plugins.lua`. Ok, actually you can use any name if you like. So, here we call it `plugins.lua`.

Take a look:

```lua
-- nvim/lua/plugins.lua
vim.cmd [[packadd packer.nvim]]

return require('packer').startup(function()
	-- Packer can manage itself
	use 'wbthomason/packer.nvim'
	-- Lsp config
	use {
		'williamboman/nvim-lsp-installer',
		config = function()
			require('conf.nvim-lsp-installer-conf')
		end
	}
	use 'neovim/nvim-lspconfig'
	use 'hrsh7th/nvim-cmp'
	use 'hrsh7th/cmp-nvim-lsp'
	use 'saadparwaiz1/cmp_luasnip' -- Snippets source for nvim-cmp
	use 'L3MON4D3/LuaSnip' -- Snippets plugin
	use 'hrsh7th/cmp-buffer'
	use 'hrsh7th/cmp-path'
	use 'hrsh7th/cmp-nvim-lua'
    -- Markdown Preview
	use 'iamcco/markdown-preview.nvim'

    -- lspsaga beautify the windows of lsp
	use {
        'tami5/lspsaga.nvim',
        config = function()
            -- require('conf.lspsaga-conf')
            require('lspsaga').setup{}
        end
    } 

	-- Dashboard
	use {
		'goolord/alpha-nvim',
		requires = { 'kyazdani42/nvim-web-devicons' },
		config = function ()
			require'alpha'.setup(require'alpha.themes.startify'.config)
		end
	}

	-- Comment plugin
	use {
		'numToStr/Comment.nvim',
		config = function()
			require('Comment').setup{}
		end
  	}

	use 'xiyaowong/nvim-cursorword'

	use({
		"NTBBloodbath/galaxyline.nvim",
		-- your statusline
		config = function()
			require("galaxyline.themes.eviline")
		end,
		-- some optional icons
		requires = { "kyazdani42/nvim-web-devicons", opt = true }
	})

	use {'akinsho/bufferline.nvim',
		tag = "v2.*",
		requires = 'kyazdani42/nvim-web-devicons',
		config = function()
			require('bufferline').setup{}
		end
	}

  	-- Dirctory Tree
	use {
    	'kyazdani42/nvim-tree.lua',
    	requires = {
      		'kyazdani42/nvim-web-devicons', -- optional, for file icon
    	},
    	tag = 'nightly', -- optional, updated every week. (see issue #1193)
		config = function()
			require('nvim-tree').setup{}
		end
	}

	use({
		'NTBBloodbath/doom-one.nvim',
		config = function()
			require('doom-one').setup({
				cursor_coloring = false,
				terminal_colors = false,
				italic_comments = false,
				enable_treesitter = true,
				transparent_background = false,
				pumblend = {
					enable = true,
					transparency_amount = 20,
				},
				plugins_integrations = {
					neorg = true,
					barbar = true,
					bufferline = false,
					gitgutter = false,
					gitsigns = true,
					telescope = false,
					neogit = true,
					nvim_tree = true,
					dashboard = true,
					startify = true,
					whichkey = true,
					indent_blankline = true,
					vim_illuminate = true,
					lspsaga = false,
				},
			})
		end,
	})

    use {
        'nvim-telescope/telescope.nvim',
        requires = {
            "nvim-lua/plenary.nvim", -- Lua development module
            "BurntSushi/ripgrep", -- characters finding
            "sharkdp/fd" -- file search
        },
        config = function()
            require('telescope').setup{}
        end
    }
end)
```

Oh sh\*t. It's so loooooong. Let's simplify it a bit and explian it.

```Lua
-- nvim/lua/plugins.lua
vim.cmd [[packadd packer.nvim]]

return require('packer').startup(function()
    -- Packer can manage itself
    use 'wbthomason/packer.nvim'
    -- a plugin come from Github
    use 'user/plugin1'
    -- a plugin come from Github with some configuration
    use {
        'user/plugin2',
        -- some dependences
        requires = {...},
        -- configuration by anonymous funciton
        config = function()
            -- --- code ----
            ...
        end
    }
end)
```

Now, you can download all plugins that you like. [Awesome-Neovim](https://github.com/rockerBOO/awesome-neovim) has a lot of awesome plugins.

In the `nvim/lua/localconf.lua`:

```lua
-- nvim/lua/localconf.lua
vim.o.relativenumber = true
vim.o.number = true
vim.o.tabstop = 4
vim.o.shiftwidth = 4
vim.o.smartindent = true
vim.o.termguicolors = true
vim.o.cursorline = true
vim.o.mouse = 'a'
vim.o.scrolloff = 3
vim.o.expandtab = true
vim.o.wildmenu = true
vim.o.ignorecase = true
vim.o.swapfile = false

-- Markdown-Preview.nvim
vim.g.mkdp_filetypes = { 'markdown' }
vim.cmd('autocmd vimenter *.md exec ":MarkdownPreview"')
```

The global configuration of the editor has been changed in a new way.
If you have any unknown config items, you can use `:help ...` to query, such as `:help vim.o`.

Time to pause.
