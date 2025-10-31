# Shadowfox
# autocorrect_keyboard_gui.py
# Simple GUI Autocorrect & Next-Word Prediction Keyboard using Tkinter

import tkinter as tk
from collections import defaultdict, Counter

# -----------------------------
# 1. Training Data (tiny sample)
# -----------------------------
corpus = [
    "i am happy",
    "i am sad",
    "i love python",
    "you are happy",
    "she is kind",
    "he is smart",
    "they are playing",
    "i am learning python",
    "we are going to school",
    "i love coding",
]

# Build bigram model
bigrams = defaultdict(Counter)
vocab = set()
for sentence in corpus:
    words = sentence.lower().split()
    for w1, w2 in zip(words, words[1:]):
        bigrams[w1][w2] += 1
        vocab.update([w1, w2])

# -----------------------------
# 2. Helper Functions
# -----------------------------
def edit_distance(a, b):
    dp = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )
    return dp[-1][-1]

def autocorrect(word):
    if not word:
        return word
    return min(vocab, key=lambda w: edit_distance(word, w))

def predict_next(word):
    if word in bigrams:
        return [w for w, _ in bigrams[word].most_common(3)]
    return []

# -----------------------------
# 3. Tkinter GUI
# -----------------------------
root = tk.Tk()
root.title("Autocorrect & Next Word Predictor")
root.geometry("450x300")
root.config(bg="#f0f0f0")

label = tk.Label(root, text="Type below:", font=("Arial", 12), bg="#f0f0f0")
label.pack(pady=5)

text_box = tk.Text(root, height=5, width=50, font=("Arial", 12))
text_box.pack(pady=5)

suggestion_label = tk.Label(root, text="Suggestions:", font=("Arial", 11, "bold"), bg="#f0f0f0")
suggestion_label.pack()

buttons_frame = tk.Frame(root, bg="#f0f0f0")
buttons_frame.pack(pady=5)

buttons = []
for i in range(3):
    b = tk.Button(buttons_frame, text="", font=("Arial", 11), width=12, bg="white", relief="ridge")
    b.grid(row=0, column=i, padx=5)
    buttons.append(b)

# -----------------------------
# 4. Core Logic
# -----------------------------
def update_suggestions(event=None):
    content = text_box.get("1.0", tk.END).strip().lower()
    words = content.split()
    if not words:
        for b in buttons:
            b.config(text="")
        return
    last_word = words[-1]

    corrected = autocorrect(last_word)
    if corrected != last_word:
        text_box.delete("end-2c wordstart", "end-1c")
        text_box.insert(tk.END, corrected + " ")

    preds = predict_next(corrected)
    for i, b in enumerate(buttons):
        b.config(text=preds[i] if i < len(preds) else "")

def insert_suggestion(idx):
    word = buttons[idx].cget("text")
    if word:
        text_box.insert(tk.END, word + " ")
        update_suggestions()

for i in range(3):
    buttons[i].config(command=lambda i=i: insert_suggestion(i))

text_box.bind("<space>", update_suggestions)
text_box.bind("<KeyRelease>", update_suggestions)

root.mainloop()


# -------------------------------
# Store Sales and Profit Analysis
# -------------------------------

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

# Optional: to display plots inline (for Jupyter)
# %matplotlib inline

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
# Replace with your file path or Google Drive share link
file_path = "your_dataset.csv"  # Example: 'store_sales.csv'
data = pd.read_csv(file_path)

# -------------------------------
# Step 2: Basic Exploration
# -------------------------------
print("First 5 rows of data:\n", data.head())
print("\nData Info:\n")
print(data.info())
print("\nMissing Values:\n", data.isnull().sum())

# -------------------------------
# Step 3: Data Cleaning
# -------------------------------
# Drop duplicates if any
data = data.drop_duplicates()

# Fill missing numeric values with median
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    data[col] = data[col].fillna(data[col].median())

# -------------------------------
# Step 4: Summary Statistics
# -------------------------------
print("\nSummary Statistics:\n", data.describe())

# -------------------------------
# Step 5: Sales and Profit Analysis
# -------------------------------
# Total Sales and Profit
total_sales = data['Sales'].sum()
total_profit = data['Profit'].sum()
print(f"\nTotal Sales: ₹{total_sales:,.2f}")
print(f"Total Profit: ₹{total_profit:,.2f}")

# Profit Margin
data['Profit Margin (%)'] = (data['Profit'] / data['Sales']) * 100

# -------------------------------
# Step 6: Category-wise Analysis
# -------------------------------
category_summary = data.groupby('Category')[['Sales', 'Profit']].sum().sort_values('Sales', ascending=False)
print("\nCategory-wise Summary:\n", category_summary)

# Plot
category_summary.plot(kind='bar', figsize=(8,5))
plt.title('Sales and Profit by Category')
plt.ylabel('Amount')
plt.xlabel('Category')
plt.grid(True)
plt.show()

# -------------------------------
# Step 7: Region-wise Analysis
# -------------------------------
region_summary = data.groupby('Region')[['Sales', 'Profit']].sum()
print("\nRegion-wise Summary:\n", region_summary)

region_summary.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Region-wise Sales and Profit')
plt.ylabel('Amount')
plt.xlabel('Region')
plt.show()

# -------------------------------
# Step 8: Top 10 Products by Profit
# -------------------------------
top_products = data.groupby('Product Name')['Profit'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Products by Profit:\n", top_products)

top_products.plot(kind='barh', color='green')
plt.title('Top 10 Products by Profit')
plt.xlabel('Profit')
plt.show()

# -------------------------------
# Step 9: Correlation Analysis
# -------------------------------
plt.figure(figsize=(6,4))
plt.matshow(data[['Sales', 'Profit', 'Quantity', 'Discount']].corr(), cmap='coolwarm', fignum=1)
plt.xticks(range(4), ['Sales', 'Profit', 'Quantity', 'Discount'], rotation=45)
plt.yticks(range(4), ['Sales', 'Profit', 'Quantity', 'Discount'])
plt.colorbar()
plt.title('Correlation Heatmap', pad=20)
plt.show()