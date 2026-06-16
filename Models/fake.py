import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


def plot_features(df, features, title_suffix=''):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f'Features vs Genuine {title_suffix}', fontsize=16)
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        genuine = df[df['is_genuine'] == True][feature]
        fake = df[df['is_genuine'] == False][feature]
        
        axes[i].scatter(genuine, [1] * len(genuine), color='blue', label='Genuine', alpha=0.5)
        axes[i].scatter(fake, [0] * len(fake), color='red', label='Fake', alpha=0.5)
        
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('is_genuine')
        axes[i].set_yticks([0, 1])
        axes[i].set_yticklabels(['Fake', 'Genuine'])
        axes[i].legend()
        axes[i].set_title(f'{feature} vs Genuine')
    
    plt.tight_layout()
    plt.show(block=False)

def main():
    df = pd.read_csv('../Data/fake_bills.csv', sep=';')

    df = df.dropna()
    
    # Feature engineering
    df['area_left'] = df['height_left'] * df['length']
    df['area_right'] = df['height_right'] * df['length']
    
    features = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length', 'area_left', 'area_right']
    
    # Plot unnormalized
    #plot_features(df, features, title_suffix='(Unnormalized)')
    
    # Normalize
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Plot normalized
    plot_features(df, features, title_suffix='(Normalized)')
    
    plt.pause(0.1)
    
    # Logistic Regression on length and area_left
    X = df[['margin_low', 'length']]
    y = df['is_genuine']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print("\n--- Random Forest Results ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions))

    plot_decision_boundary(model, X_test, y_test, title='Random Forest Decision Boundary')
    plt.pause(0.1)

    input("Press Enter to close...")



def plot_decision_boundary(model, X_test, y_test, title=''):
    x_min, x_max = X_test.iloc[:, 0].min() - 0.5, X_test.iloc[:, 0].max() + 0.5
    y_min, y_max = X_test.iloc[:, 1].min() - 0.5, X_test.iloc[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z == True).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, colors=['red', 'blue'])
    
    genuine = y_test == True
    fake = y_test == False
    
    plt.scatter(X_test[genuine].iloc[:, 0], X_test[genuine].iloc[:, 1], 
                color='blue', label='Genuine', edgecolors='black', alpha=0.7)
    plt.scatter(X_test[fake].iloc[:, 0], X_test[fake].iloc[:, 1], 
                color='red', label='Fake', edgecolors='black', alpha=0.7)
    
    plt.xlabel('length (normalized)')
    plt.ylabel('area_left (normalized)')
    plt.title(title)
    plt.legend()
    plt.show(block=False)

if __name__ == '__main__':
    main()