import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def pt2inch(pt):
    return pt / 72.27

def update_rcParams(fontsize=10):
    plt.rcParams.update({
        "figure.figsize": (pt2inch(237), pt2inch(158)),
        "font.family": "Arial",
        "font.size": fontsize,
        "xtick.labelsize": 'small',
        "ytick.labelsize": 'small',
        "legend.fontsize": 'small',
        "axes.labelsize": 'medium',
        "lines.linewidth": 2,
        "lines.markersize": 3
    })

class TimeManager:
    def __init__(self):
        self.start = time.time()
        self.prev = self.start

    def print_time(self):
        now = time.time()
        print(f"Time: {now - self.start:.2f} s (section: {now - self.prev:.2f} s)")
        self.prev = now

class DataManager:
    @staticmethod
    def open_json(path):
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def get_json_value(json_data, key):
        if key in json_data:
            return json_data[key]
        raise KeyError(f"Key not found in JSON object: {key}")

    @staticmethod
    def extract_freqs(data, freq_range):
        freq_indices = [f-4 for f in freq_range]
        return data[:, freq_indices]

class MLAnalyzer:
    MATERIAL_COLOR = {
        "ABS5": "tab:blue",
        "PE10": "tab:orange",
        "PMMA2": "tab:green",
        "PMMA3": "tab:red",
        "POM10": "tab:purple",
        "PP10": "tab:brown",
        "PVC3": "tab:pink",
        "Towel": "tab:cyan",
        "Paper": "tab:gray",
        "Wax": "tab:olive",
        "Reference": "tab:gray",
        "Mirror": "tab:gray",
    }

    materials = [
		"ABS5",
		"PE10",
		"PMMA2",
		"PMMA3",
		"POM10",
		"PP10",
		"PVC3"
	]

    def __init__(self, id, save_results=True):
        self.id = id
        self.tm = TimeManager()
        self.save_results = save_results
        self.label = np.load(f'dataset/{self.id}/label.npy')

    def cross_validation(self, dataset_name, freqs_mhz=None):
        print(f"Cross-validation for dataset: {dataset_name}")
        
        data = np.load(f'dataset/{self.id}/{dataset_name}.npy')

        if freqs_mhz:
            data = DataManager.extract_freqs(data, freqs_mhz)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        classifiers = {
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(n_neighbors=1),
            "Linear SVM": LinearSVC(),
            "RBF SVM": SVC(kernel='rbf'),
            "Random Forest": RandomForestClassifier(n_estimators=20),
            "AdaBoost": AdaBoostClassifier(),
            "LDA": LDA(),
            "QDA": QuadraticDiscriminantAnalysis()
        }
        
        results = []
        for name, clf in classifiers.items():
            scores = cross_val_score(clf, data, self.label, cv=skf)
            results.append([name, *scores, np.mean(scores)])
            print(f"{name}: mean accuracy {np.mean(scores):.4f}")
            self.tm.print_time()
        
        if self.save_results:
            os.makedirs(f'results/cross_validation', exist_ok=True)
            results_path = f'results/cross_validation/{self.id}_cross_validation_{dataset_name}.csv'
            pd.DataFrame(results, columns=['Model', 'Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5', 'Mean']).to_csv(results_path, index=False)
            print(f"Results saved to {results_path}")
    
    # Plot 2D visualization of dataset with LDA
    def plot_2d_lda(self, dataset_name, freqs_mhz=None):
        print(f"Performing LDA for {dataset_name}")
        
        data = np.load(f'dataset/{self.id}/{dataset_name}.npy')

        if freqs_mhz is None:
            freqs_mhz = range(4, 101)
        else:
            data = DataManager.extract_freqs(data, freqs_mhz)

        data = StandardScaler().fit_transform(data)
        lda = LDA()
        lda.fit(data, self.label)
        data_lda = lda.transform(data)

        plt.figure(figsize=(4, 4))
        num_per_material = len(data) // len(self.materials)

        for i, material in enumerate(self.materials):
            plt.scatter(
                data_lda[i * num_per_material:(i + 1) * num_per_material, 0],
                data_lda[i * num_per_material:(i + 1) * num_per_material, 1],
                label=material,
                color = self.MATERIAL_COLOR[material]
            )

        plt.xlabel('LD1', fontsize=18)
        plt.ylabel('LD2', fontsize=18)
        plt.tick_params(labelsize=15)
        plt.tight_layout()
        plt.xlim(-12, 12)
        plt.ylim(-12, 12)

        if self.save_results:
            os.makedirs(f'results/lda2d', exist_ok=True)
            path = f'results/lda2d/notitle_LDA_{self.id}_{dataset_name}_{len(freqs_mhz)}.png'
            plt.savefig(path, bbox_inches='tight')
            print(f"Saved LDA plot to {path}")
        
        plt.show()
        plt.close()
    
    def compute_confusion_matrix(self, dataset_name):
        print(f"Computing confusion matrix for {dataset_name}")
        
        data = np.load(f'dataset/{self.id}/{dataset_name}.npy')
        
        X_train, X_test, y_train, y_test = train_test_split(data, self.label, test_size=0.2, random_state=42)
        
        lda = LDA()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred, labels=self.materials)
        cm_norm = confusion_matrix(y_test, y_pred, labels=self.materials, normalize='true')
        cm_norm = np.floor(cm_norm * 1000) / 1000
        
        if self.save_results:
            os.makedirs(f'results/confusion_matrix/{self.id}', exist_ok=True)
            pd.DataFrame(cm, index=self.materials, columns=self.materials).to_csv(f'results/confusion_matrix/{self.id}/confusion_matrix_{dataset_name}.csv')
            pd.DataFrame(cm_norm, index=self.materials, columns=self.materials).to_csv(f'results/confusion_matrix/{self.id}/confusion_matrix_{dataset_name}_norm.csv')
        
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=self.materials, xticks_rotation='vertical', normalize='true', include_values=False, values_format='.3g', cmap='Blues', colorbar=True)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('')
        plt.ylabel('')
        
        if self.save_results:
            plt.savefig(f'results/confusion_matrix/{self.id}/confusion_matrix_{dataset_name}_norm.png', bbox_inches='tight')
        plt.close()

    def compute_contribution_scores(self, dataset_name):
        print(f"Compute contribution scores for {dataset_name}")

        data = np.load(f'dataset/{self.id}/{dataset_name}.npy')

        # Fit LDA
        data = StandardScaler().fit_transform(data)
        lda = LDA()
        lda.fit(data, self.label)
        features = lda.transform(data)  # Not strictly needed if you only want the contribution scores

        # Compute contribution scores
        # NOTE: In scikit-learn’s LDA, S_ and Vt_ are not public by default.
        #       You must modify _solve_svd(self, X, y) in the LDA implementation.
        #       Right after:
        #           U, S, Vt = svd(X, full_matrices=False)
        #       add:
        #           self.S_ = S
        #           self.Vt_ = Vt
        #       to make these attributes accessible for frequency-domain analysis.
        repetition = data.shape[1] // 97

        x = list(range(4, 101))  # Frequency axis from 4 MHz to 100 MHz

        # Absolute scalings for each frequency component
        scalings = np.abs(lda.Vt_ * lda.S_.reshape(-1, 1))
        print(f"scalings shape: {scalings.shape}")

        # Sum over each block of 97 frequencies (if your dataset is structured that way)
        scalings_data = None
        for i in range(repetition):
            scaling_tmp = scalings[:, i * 97:(i + 1) * 97]
            # Normalize rows
            scaling_tmp /= np.sum(scaling_tmp, axis=1).reshape(-1, 1)
            if i == 0:
                scalings_data = scaling_tmp
            else:
                scalings_data += scaling_tmp

        # Collapse down to a single 1D array of scores (size 97)
        scalings_data = np.sum(scalings_data, axis=0)
        # Sort frequencies by contribution score
        sorted_index = np.argsort(scalings_data)[::-1] + 4
        print(f"Sorted frequency indices: {sorted_index}")

        # (Optionally) save numerical results
        if self.save_results:
            os.makedirs(f'results/contribution_scores/{self.id}', exist_ok=True)
            np.save(f'results/contribution_scores/{self.id}/sorted_index_{dataset_name}.npy', sorted_index)

        # Plot the contribution scores
        plt.figure(figsize=(6, 4))
        plt.plot(x, scalings_data)
        plt.xlabel("Frequency [MHz]", fontsize=18)
        plt.ylabel("Contribution Score", fontsize=18)
        plt.tick_params(labelsize=15)
        plt.grid()

        # (Optionally) save the plot
        if self.save_results:
            name = f'results/contribution_scores/{self.id}/contribution_score_{dataset_name}'
            plt.savefig(name + '.svg', bbox_inches='tight')
            plt.savefig(name + '.png', bbox_inches='tight')
            plt.savefig(name + '.pdf', bbox_inches='tight')

        plt.close()

if __name__ == '__main__':
    update_rcParams()
    
    # dataset id
    dataset_id = '202411_polarimetricToF'

    analyzer = MLAnalyzer(dataset_id, save_results=True)

    # Dataset names: 'phase_amp', 'phase', 'amp'
    dataset_name = 'phase_amp'

    # Using frequencies in the range of 4-100 MHz
    # e.g. freqs = [100, 87, 74]
    freqs_mhz = list(range(4, 101))

    analyzer.plot_2d_lda(dataset_name, freqs_mhz)
    analyzer.cross_validation(dataset_name, freqs_mhz)
    analyzer.compute_confusion_matrix(dataset_name)
    analyzer.compute_contribution_scores(dataset_name)
