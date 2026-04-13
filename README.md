Step 1: Install the Prerequisites
You'll need two things installed on your computer before anything else.
Install Git
Mac:

Open Terminal (press Cmd + Space, type "Terminal", hit Enter)
Type the following and press Enter:

   git --version

If Git is not installed, your Mac will prompt you to install it automatically. Follow the on-screen instructions.

Windows:

Go to https://git-scm.com/download/win
Download and run the installer
Accept all default settings during installation
Open Git Bash (search for it in the Start menu) — use this instead of Command Prompt for all Git commands

Install Python

Go to https://www.python.org/downloads/
Download and run the installer for your OS
Windows users: On the first installer screen, check the box that says "Add Python to PATH" before clicking Install


Step 2: Configure Git with Your Identity
This only needs to be done once per computer. Open Terminal (Mac) or Git Bash (Windows) and run these two commands, replacing the values with your own name and email:
bashgit config --global user.name "Your Name"
git config --global user.email "your@email.com"
Use the same email address associated with your GitHub account.

Step 3: Clone the Repository
"Cloning" means downloading a copy of this project onto your computer so you can work on it.

Open Terminal (Mac) or Git Bash (Windows)
Navigate to the folder where you want to store the project. For example, to save it to your Desktop:
Mac:

bash   cd ~/Desktop
Windows:
bash   cd C:/Users/YourName/Desktop

Clone the repo by running:

bash   git clone https://github.com/ali00418-ship-it/msba-team2-trendsproject.git

Move into the project folder:

bash   cd msba-team2-trendsproject
You now have a full local copy of the project on your machine. ✅

Step 4: Install Python Dependencies
Once inside the project folder, install the required Python libraries:
bashpip install -r requirements.txt

⚠️ If you get a "requirements.txt not found" error, this file hasn't been added yet. Check with the team or skip this step for now.


Step 5: Make Your Changes
Open the project in your preferred code editor (we recommend VS Code). Make your edits to the relevant files.

Step 6: Save and Upload Your Changes
After making changes, follow these three steps every time to save and share your work with the team.
6a — Stage your changes
This tells Git which files you want to save:
bashgit add .
6b — Commit your changes
This saves a snapshot of your work with a short description:
bashgit commit -m "Brief description of what you changed"
For example:
bashgit commit -m "Added data cleaning script for CFPB dataset"
6c — Push your changes to GitHub
This uploads your saved snapshot to the shared repository so everyone can see it:
bashgit push

Step 7: Pull the Latest Changes from Teammates
Before you start working each session, always download your teammates' latest changes first to avoid conflicts:
bashgit pull
Make this a habit — pull before you start, push when you're done.

⚠️ Common Issues & Fixes
ProblemFixfatal: no email was givenRun Step 2 above to set your Git identityerror: src refspec main does not match anyYou haven't made a commit yet. Complete Step 6b firstrejected — non-fast-forwardSomeone else pushed changes. Run git pull first, then git push againpermission deniedMake sure you're logged into GitHub in your terminal. Try git push and enter your GitHub credentials when promptedPython not foundMake sure Python is installed and added to PATH (see Step 1)

📁 Project Structure
msba-team2-trendsproject/
│
├── MSBA_Market_Trends.py     # Main analysis script
├── README.md                 # This file
└── requirements.txt          # Python dependencies (to be added)

