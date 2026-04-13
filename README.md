# Message Based Bitcoin Scam Classification using Linguistic and Transaction Pattern Analysis

A desktop dashboard built with Python and Tkinter to detect scam vs legit Bitcoin-related messages using multiple ML models.

## Features

- Role-based login (`ADMIN` / `USER`)
- Guided workflow with in-app Guide window
- Light/Dark theme toggle
- Upload, preprocess, split, train, and predict flow
- Multiple models:
  - XGBoost
  - LightGBM
  - AdaBoost
  - Stacking (SGD + PAC)
- Graph viewer with:
  - Flash Graphs dropdown
  - Model-specific graph dropdown on hover
  - `<` / `>` buttons
  - Left/Right keyboard arrow navigation

## Project Structure

- `app.py` - Main application
- `requirements.txt` - Python dependencies
- `models/` - Saved encoders/models (generated at runtime)
- `results/` - Generated plots (generated at runtime)
- `Dataset/` - Input datasets (local, ignored in Git)

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Default Credentials

- Admin:
  - Username: `admin`
  - Password: `admin`
- User:
  - Username: `user`
  - Password: `user`

## Recommended Usage Flow

### Admin

1. Login as `ADMIN`
2. Click `Upload Dataset`
3. Click `Preprocess Dataset`
4. Click `Dataset Splitting`
5. Train models (`Train XGBoost`, `Train LightGBM`, `Train AdaBoost`, `Train Stacking`)
6. View graphs from:
   - `Flash Graphs`
   - Hovering on model train buttons

### User

1. Login as `USER`
2. Click `Predict on Test Data`
3. Use graph controls to view performance graphs

## Notes

- If training is clicked before required steps, the app shows: `Read the Guide`.
- Generated artifacts are ignored via `.gitignore`.
- Use `Logout` to return to the login screen.
