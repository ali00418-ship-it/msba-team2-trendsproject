# 🏦 Karen - AI Complaint Assistant
**MSBA Trends Project — Team 2**

> Turning large volumes of consumer banking complaints into clear, prioritized action items for product teams.

---

## 👥 Team Members
Mohameddeq Ali · Cora Goodwin · Midori Neaton · Raja Sori · Xupei Ye · Kyle Zhu

---

## 📌 Project Overview

Banks and fintech product teams receive too many complaints to review individually. The real challenge is figuring out **which pain points matter most**, which ones are getting worse, and what should be fixed first.

This project builds a **Banking Complaint Intelligence System** that helps product teams turn raw complaint data into prioritized insights — delivered through a dashboard or copilot-style interface.

**Outputs include:**
- Top consumer complaint themes
- Fast-growing issue areas
- Affected products and customer segments
- Priority scores with written rationale for each cluster
- NLP-generated summaries of complaint narratives

---

## 🗂️ Data Source

**CFPB Consumer Complaint Database (Public)**
[https://catalog.data.gov/dataset/consumer-complaint-database](https://catalog.data.gov/dataset/consumer-complaint-database)

---

## 🛠️ Tools & Technologies

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, PySpark |
| Querying | SQL |
| NLP / Clustering | BERTopic |
| Visualization | Tableau |
| AI / LLM | LLM-powered prioritization engine |

---

## 🚀 Getting Started — How to Collaborate

Follow the steps below **in order**. This guide is written for complete beginners — no prior Git experience required.

---

### Step 1: Install the Prerequisites

You'll need two things installed on your computer before anything else.

#### Install Git

**Mac:**
1. Open **Terminal** (press `Cmd + Space`, type "Terminal", hit Enter)
2. Type the following and press Enter:
   ```
   git --version
   ```
3. If Git is not installed, your Mac will prompt you to install it automatically. Follow the on-screen instructions.

**Windows:**
1. Go to [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Download and run the installer
3. Accept all default settings during installation
4. Open **Git Bash** (search for it in the Start menu) — use this instead of Command Prompt for all Git commands

#### Install Python

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download and run the installer for your OS
3. **Windows users:** On the first installer screen, check the box that says **"Add Python to PATH"** before clicking Install

---

### Step 2: Configure Git with Your Identity

This only needs to be done once per computer. Open Terminal (Mac) or Git Bash (Windows) and run these two commands, replacing the values with your own name and email:

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

Use the same email address associated with your GitHub account.

---

### Step 3: Clone the Repository

"Cloning" means downloading a copy of this project onto your computer so you can work on it.

1. Open Terminal (Mac) or Git Bash (Windows)
2. Navigate to the folder where you want to store the project. For example, to save it to your Desktop:

   **Mac:**
   ```bash
   cd ~/Desktop
   ```

   **Windows:**
   ```bash
   cd C:/Users/YourName/Desktop
   ```

3. Clone the repo by running:
   ```bash
   git clone https://github.com/ali00418-ship-it/msba-team2-trendsproject.git
   ```

4. Move into the project folder:
   ```bash
   cd msba-team2-trendsproject
   ```

You now have a full local copy of the project on your machine. ✅

---

### Step 4: Install Python Dependencies

Once inside the project folder, install the required Python libraries:

```bash
pip install -r requirements.txt
```

> ⚠️ If you get a "requirements.txt not found" error, this file hasn't been added yet. Check with the team or skip this step for now.

---

### Step 5: Create Your Own Branch

> 💡 **What is a branch?** Think of `main` as the official, shared version of the project. A branch is your own personal copy where you can make changes safely — without affecting anyone else's work — until you're ready to merge it in.

**Never work directly on `main`.** Always create your own branch first.

Name your branch after yourself or the feature you're working on. For example:

```bash
git checkout -b your-name
```

Real examples:
```bash
git checkout -b mohameddeq
git checkout -b cora-nlp-clustering
git checkout -b raja-dashboard
```

You only need to create your branch once. After that, Git will remember it.

To check which branch you're currently on at any time:
```bash
git branch
```

The branch with a `*` next to it is your active one.

---

### Step 6: Make Your Changes

Open the project in your preferred code editor (we recommend [VS Code](https://code.visualstudio.com/)). Make your edits to the relevant files — you're working safely on your own branch, so nothing affects `main` until you're ready.

---

### Step 7: Save and Upload Your Changes to Your Branch

After making changes, follow these three steps every time.

#### 7a — Stage your changes
This tells Git which files you want to save:
```bash
git add .
```

#### 7b — Commit your changes
This saves a snapshot of your work with a short description:
```bash
git commit -m "Brief description of what you changed"
```

For example:
```bash
git commit -m "Added data cleaning script for CFPB dataset"
```

#### 7c — Push your branch to GitHub
The first time you push a new branch, use this command (replace `your-name` with your actual branch name):
```bash
git push -u origin your-name
```

After the first push, you can just use:
```bash
git push
```

---

### Step 8: Open a Pull Request (PR)

Once your work is ready to be reviewed and merged into `main`, open a Pull Request on GitHub.

1. Go to the repo: [https://github.com/ali00418-ship-it/msba-team2-trendsproject](https://github.com/ali00418-ship-it/msba-team2-trendsproject)
2. You'll see a yellow banner saying **"Compare & pull request"** — click it
3. Add a short title and description of what you changed
4. Click **"Create pull request"**
5. Let a teammate know to review and merge it

> ⚠️ Do **not** merge your own PR without a teammate reviewing it first.

---

### Step 9: Pull the Latest Changes from Teammates

Before you start working each session, always download the latest changes from `main` into your branch to stay up to date:

```bash
git pull origin main
```

Make this a habit — **pull before you start, push when you're done.**

---

## ⚠️ Common Issues & Fixes

| Problem | Fix |
|---|---|
| `fatal: no email was given` | Run Step 2 above to set your Git identity |
| `error: src refspec main does not match any` | You haven't made a commit yet. Complete Step 7b first |
| `rejected — non-fast-forward` | Someone else pushed changes. Run `git pull origin main` first, then `git push` again |
| Accidentally working on `main` | Run `git checkout -b your-name` — your uncommitted changes carry over to the new branch automatically |
| `permission denied` | Make sure you're logged into GitHub in your terminal. Try `git push` and enter your GitHub credentials when prompted |
| `error: pathspec did not match` | Your branch does not exist yet. Run `git checkout -b your-branch-name` to create it |
| Python not found | Make sure Python is installed and added to PATH (see Step 1) |

---

## 📁 Project Structure

```
msba-team2-trendsproject/
│
├── MSBA_Market_Trends.py     # Main analysis script
├── README.md                 # This file
└── requirements.txt          # Python dependencies (to be added)
```

---

## 📬 Questions?

Reach out to any team member or open an [Issue](https://github.com/ali00418-ship-it/msba-team2-trendsproject/issues) on GitHub.
