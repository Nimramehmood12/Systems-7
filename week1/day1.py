import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    data = {'Department':['Engineering','Marketing','Engineering','HR','Marketing','Engineering','HR','Engineering'],'Salary':[85000,62000,92000,50000,67000,105000,54000,78000],'Exp':[3,2,5,2,4,7,1,3],'Perf':[88,79,94,82,85,96,75,87]}
    df = pd.DataFrame(data)
    df.to_csv('employee_data.csv', index=False)
    loaded_df = pd.read_csv('employee_data.csv')
    print('---SUMMARY STATISTICS---')
    print(loaded_df.describe())
    print('\n---Grouped salary by department---')
    print(loaded_df.groupby('Department')['Salary'].mean())
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=loaded_df, x='Department', y='Salary', hue='Department', legend=False, errorbar=None, ax=axes[0], palette='Blues_d')
    axes[0].set_title('Average Salary by Department')
    sns.scatterplot(data=loaded_df, x='Exp', y='Salary', hue='Perf', size='Perf', sizes=(50, 200), palette='viridis', ax=axes[1])
    axes[1].set_title('Salary vs Experience')
    plt.tight_layout()
    plt.savefig('employee_analysis.png')
    plt.show()

if __name__ == '__main__':
    main()
