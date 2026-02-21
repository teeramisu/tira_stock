# 把 stock_backtest 托管到 GitHub

## 1. 在 GitHub 上创建仓库

1. 打开 https://github.com/new  
2. **Repository name** 填：`stock_backtest`（或任意名称）  
3. 选择 **Public**，**不要**勾选 “Add a README file”  
4. 点击 **Create repository**  
5. 记下仓库地址，例如：`https://github.com/你的用户名/stock_backtest.git`

---

## 2. 在本地终端执行（在 stock_backtest 目录下）

当前状态：已执行过 `git init` 和 `git add -A`，文件已暂存，尚未提交。

**第一次提交**（若在 Cursor 里 commit 报错 `unknown option trailer`，请在本机终端执行）：

```bash
cd /home/tira/code/stock_backtest
git commit -m "Initial commit"
```

**添加远程并推送**（把 `你的用户名` 换成你的 GitHub 用户名）：

```bash
git remote add origin https://github.com/你的用户名/stock_backtest.git
git branch -M main
git push -u origin main
```

若 GitHub 使用 SSH，可改为：

```bash
git remote add origin git@github.com:你的用户名/stock_backtest.git
git branch -M main
git push -u origin main
```

---

## 3. 若尚未配置 Git 用户信息

首次在本机使用 Git 时可能需要：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

---

## 4. 之后日常更新到 GitHub

```bash
cd /home/tira/code/stock_backtest
git add -A
git commit -m "描述本次修改"
git push
```
