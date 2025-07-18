import tkinter as tk
from tkinter import filedialog, Text, Scrollbar
from fraud_predictor import train_model, predict
from model_trainer import split_data
from data_loader import load_dataset
from graph_visualizer import plot_graph
from sklearn.metrics import accuracy_score

def run_app():
    app = tk.Tk()
    app.title("Credit Card Fraud Detection")
    app.geometry("1200x700")
    app.config(bg='LightSkyBlue')

    font_title = ('times', 16, 'bold')
    font_label = ('times', 12, 'bold')

    tk.Label(app, text='Credit Card Fraud Detection Using Random Forest', bg='greenyellow',
             fg='dodger blue', font=font_title, height=3, width=80).pack()

    output = Text(app, height=20, width=140, font=font_label)
    output.pack(pady=20)
    Scrollbar(output).pack(side=tk.RIGHT, fill=tk.Y)

    state = {'model': None, 'X_test': None, 'y_test': None, 'total': 0, 'clean': 0, 'fraud': 0}

    def upload():
        path = filedialog.askopenfilename(initialdir="dataset")
        output.insert(tk.END, f"{path} loaded\n")
        data = load_dataset(path)
        X_train, X_test, y_train, y_test = split_data(data)
        model = train_model(X_train, y_train)
        state['model'] = model
        state['X_test'] = X_test
        state['y_test'] = y_test
        output.insert(tk.END, f"Train/Test model generated\nTrain Size: {len(X_train)}\nTest Size: {len(X_test)}\n")

    def evaluate():
        pred = predict(state['model'], state['X_test'])
        acc = accuracy_score(state['y_test'], pred) * 100
        output.insert(tk.END, f"Accuracy: {acc:.2f}%\n")

    def test_data():
        path = filedialog.askopenfilename(initialdir="dataset")
        output.insert(tk.END, f"{path} test file loaded\n")
        test_data = load_dataset(path).values[:, 0:29]
        preds = predict(state['model'], test_data)
        total = len(preds)
        fraud = sum(preds == 1)
        clean = total - fraud
        state.update({'total': total, 'fraud': fraud, 'clean': clean})
        for i, val in enumerate(preds):
            label = "FRAUD" if val == 1 else "CLEAN"
            output.insert(tk.END, f"Transaction {i+1}: {label}\n")

    def show_graph():
        plot_graph(state['total'], state['clean'], state['fraud'])

    tk.Button(app, text="Upload Dataset", font=font_label, command=upload).pack()
    tk.Button(app, text="Evaluate Model", font=font_label, command=evaluate).pack()
    tk.Button(app, text="Detect Fraud from Test Data", font=font_label, command=test_data).pack()
    tk.Button(app, text="Visualize Result", font=font_label, command=show_graph).pack()
    tk.Button(app, text="Exit", font=font_label, command=app.destroy).pack(pady=10)

    app.mainloop()

if __name__ == "__main__":
    run_app()
