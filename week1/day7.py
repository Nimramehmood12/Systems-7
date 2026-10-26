import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    raw = {'Age':[22,25,np.nan,30,45,np.nan,29,38,41,np.nan,23,35], 'Score':[78,85,90,np.nan,62,74,np.nan,88,95,80,np.nan,89], 'Category':['A','B','A','B',np.nan,'A','A','B','B','A',np.nan,'B']}
    dirty = pd.DataFrame(raw); clean = dirty.copy()
    clean['Age'] = clean['Age'].fillna(clean['Age'].median())
    clean['Score'] = clean['Score'].fillna(clean['Score'].mean())
    clean['Category'] = clean['Category'].fillna(clean['Category'].mode()[0])
    print('Missing before:\n', dirty.isnull().sum()); print('\nMissing after:\n', clean.isnull().sum()); print('\nCleaned data:\n', clean)
    fig, axes = plt.subplots(2,2,figsize=(13,8))
    sns.heatmap(dirty.isnull(), cbar=False, cmap='magma', ax=axes[0,0]); axes[0,0].set_title('Before')
    sns.heatmap(clean.isnull(), cbar=False, cmap='magma', ax=axes[0,1]); axes[0,1].set_title('After')
    sns.kdeplot(dirty['Age'], label='Before', ax=axes[1,0], fill=True); sns.kdeplot(clean['Age'], label='After', ax=axes[1,0], fill=True); axes[1,0].legend()
    sns.boxplot(data=pd.DataFrame({'Before':dirty['Score'], 'After':clean['Score']}), ax=axes[1,1])
    plt.tight_layout(); plt.savefig('cleaning_comparison.png'); plt.show()

if __name__ == '__main__': main()
