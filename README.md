# Message Based Bitcoin Scam Classification using Linguistic and Transaction Pattern Analysis


## What is this project?

This project helps people identify if a message about Bitcoin is a scam or legitimate. Imagine you're getting a text or email about investing in Bitcoin – this tool analyzes the message to tell you if it's safe or not. It's like having a smart friend who checks for red flags in Bitcoin offers.

## How does it work?

The system uses two main ways to spot scams:

### 1. Understanding the Words (Linguistic Analysis)
- It reads the message and breaks it down into words.
- Then, it turns those words into numbers using a technique called Word2Vec. Think of it like translating words into a secret code that computers can understand.
- This helps the system learn patterns in scam messages, like urgent language or promises of huge returns.

### 2. Looking at Patterns (Transaction Pattern Analysis)
- It checks for things like:
  - Does the message have a link? (Scams often do)
  - Is there a Bitcoin address? (Legit ones might, but in suspicious ways)
  - How long is the message?
  - What's the tone? (Happy or scary?)
  - What platform is it from?
  - How urgent does it sound?
- These patterns help spot common scam tricks.

### Putting it Together
- The system combines word analysis and pattern checking.
- It uses smart computer programs (called machine learning models) that learn from thousands of examples.
- Models like XGBoost, LightGBM, and others are trained to make predictions.
- Once trained, it can check new messages and say "scam" or "legit."

## How to Use It

This project comes with a simple desktop app. Here's how to get started:

### Step 1: Install What You Need
- Make sure you have Python installed on your computer.
- Open a command window and type: `pip install -r requirements.txt`

### Step 2: Run the App
- Type: `python app.py`
- A window will open.

### Step 3: Login
- Use these to get in:
  - For admin: username `admin`, password `admin`
  - For user: username `user`, password `user`

### Step 4: Train the System (Admin Only)
- Upload a dataset of messages.
- Prepare the data.
- Split it for training.
- Train the models (it might take a few minutes).

### Step 5: Check Messages (Anyone)
- Upload new messages to test.
- The app will predict if they're scams or not.
- Look at graphs to see how well it's working.

## Files in This Project
- `app.py`: The main app you run.
- `Dataset/`: Example data files.
- `models/`: Where trained models are saved.
- `results/`: Charts and graphs from testing.

## Important Notes
- Always be careful with real money – this is a tool to help, not a guarantee.
- The system learns from data, so the more examples it sees, the better it gets.
- If something doesn't work, check the guide in the app.

This project makes Bitcoin safer by using smart technology to spot scams before they trick you!
